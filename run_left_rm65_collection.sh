#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="/home/dora/anaconda3/envs/xr-rm65-collection/bin/python"
XR_BINDING_DIR="${PROJECT_DIR}/dependencies/XRoboToolkit-PC-Service-Pybind"

reject_existing_left_rm65_connection() {
    local connections
    connections="$(ss -Htnp 2>/dev/null | grep -E '192\.168\.1\.18:8080' || true)"
    if [[ -z "${connections}" ]]; then
        return 0
    fi
    echo "左臂启动前检查失败：已有程序占用 192.168.1.18:8080。" >&2
    echo "请先正常停止下面的进程，再重新启动采集：" >&2
    printf '%s\n' "${connections}" >&2
    echo "提示：不要同时运行左臂遥操作、双臂遥操作或双臂采集。" >&2
    return 1
}

select_xr_sdk_library_dir() {
    local candidate
    local ldd_output
    local candidates=()
    if [[ -n "${XR_SDK_LIBRARY_DIR:-}" ]]; then
        candidates+=("${XR_SDK_LIBRARY_DIR}")
    fi
    candidates+=(
        "${XR_BINDING_DIR}/lib"
        "/home/dora/XRoboToolkit-Teleop-Sample-Python/dependencies/XRoboToolkit-PC-Service-Pybind/lib"
        "/home/dora/XRoboToolkit-Teleop-Sample-Python2/dependencies/XRoboToolkit-PC-Service-Pybind/lib"
    )
    for candidate in "${candidates[@]}"; do
        if [[ -f "${candidate}/libPXREARobotSDK.so" ]] \
            && ldd_output="$(ldd "${candidate}/libPXREARobotSDK.so" 2>&1)" \
            && [[ "${ldd_output}" != *"not found"* ]]; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    done
    return 1
}

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "固定采集环境不存在或不可执行: ${PYTHON_BIN}" >&2
    exit 1
fi
if ! "${PYTHON_BIN}" -c \
    'import sys, cv2, lerobot, numpy; assert sys.executable == "/home/dora/anaconda3/envs/xr-rm65-collection/bin/python"; assert lerobot.__version__ == "0.3.3"' \
    >/dev/null 2>&1; then
    echo "固定采集环境验证失败（需要 LeRobot 0.3.3、OpenCV 和 NumPy）。" >&2
    exit 1
fi
if ! XR_SDK_LIB_DIR="$(select_xr_sdk_library_dir)"; then
    echo "找不到与本机系统兼容的 libPXREARobotSDK.so，采集模式拒绝启动。" >&2
    echo "可用 XR_SDK_LIBRARY_DIR 指向兼容 SDK 的 lib 目录。" >&2
    exit 1
fi

cd "${PROJECT_DIR}"
export LD_LIBRARY_PATH="${XR_SDK_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export XR_ROBOT_SDK_LIB="${XR_SDK_LIB_DIR}/libPXREARobotSDK.so"
export PYTHONPATH="${XR_BINDING_DIR}:${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

if ! "${PYTHON_BIN}" -c \
    'from xrobotoolkit_teleop.common.xr_atomic import resolve_sdk_library; resolve_sdk_library()' \
    >/dev/null 2>&1; then
    echo "XR SDK 原生回调库验证失败，采集模式拒绝启动。" >&2
    exit 1
fi

reject_existing_left_rm65_connection

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
exec "${PYTHON_BIN}" scripts/hardware/supervise_dual_rm65_collection.py \
    --startup-timeout 120 \
    --heartbeat-timeout 300 \
    -- \
    "${PYTHON_BIN}" scripts/hardware/collect_left_rm65_lerobot.py "$@"
