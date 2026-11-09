#!/usr/bin/env python3
"""Supervise collection outside native XR and RM65 SDK calls."""

import argparse

from xrobotoolkit_teleop.collection.supervisor import run_supervised


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--startup-timeout", type=float, default=120.0)
    parser.add_argument("--heartbeat-timeout", type=float, default=15.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        parser.error("a worker command is required after --")
    return run_supervised(
        command,
        startup_timeout_s=args.startup_timeout,
        heartbeat_timeout_s=args.heartbeat_timeout,
    )


if __name__ == "__main__":
    raise SystemExit(main())
