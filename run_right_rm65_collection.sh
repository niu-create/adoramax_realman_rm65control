#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="/home/dora/anaconda3/envs/xr-rm65-collection/bin/python"
XR_BINDING_DIR="${PROJECT_DIR}/dependencies/XRoboToolkit-PC-Service-Pybind"

if [[ ! -x "${PYTHON_BIN}" ]]; then echo "固定采集环境不存在: ${PYTHON_BIN}" >&2; exit 1; fi
if ss -Htnp 2>/dev/null | grep -q '192\.168\.1\.19:8080'; then
  echo "右臂 .19 已被其他程序占用；请先停止旧遥操作进程。" >&2; exit 1
fi

for candidate in "${XR_SDK_LIBRARY_DIR:-}" \
  "/home/dora/XRoboToolkit-Teleop-Sample-Python2/dependencies/XRoboToolkit-PC-Service-Pybind/lib" \
  "/home/dora/XRoboToolkit-Teleop-Sample-Python/dependencies/XRoboToolkit-PC-Service-Pybind/lib" \
  "${XR_BINDING_DIR}/lib"; do
  [[ -n "${candidate}" && -f "${candidate}/libPXREARobotSDK.so" ]] || continue
  if ldd "${candidate}/libPXREARobotSDK.so" 2>&1 | grep -q 'not found'; then continue; fi
  XR_SDK_LIB_DIR="${candidate}"; break
done
if [[ -z "${XR_SDK_LIB_DIR:-}" ]]; then echo "找不到兼容 XR SDK。" >&2; exit 1; fi

cd "${PROJECT_DIR}"
export LD_LIBRARY_PATH="${XR_SDK_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export XR_ROBOT_SDK_LIB="${XR_SDK_LIB_DIR}/libPXREARobotSDK.so"
export PYTHONPATH="${XR_BINDING_DIR}:${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
exec "${PYTHON_BIN}" scripts/hardware/supervise_dual_rm65_collection.py --startup-timeout 120 --heartbeat-timeout 15 -- \
  "${PYTHON_BIN}" scripts/hardware/collect_right_rm65_lerobot.py "$@"
