"""Hold the machine awake through long unattended runs. Windows Modern Standby only.

WHY POWERCFG IS NOT ENOUGH, measured the hard way twice on 1-2 Sep 2026. This machine reports

    Standby (S0 Low Power Idle) Network Connected     available
    Standby (S1) / (S2) / (S3)                        NOT available

`powercfg /change standby-timeout-ac 0` and `hibernate-timeout-ac 0` were both correctly set
to 0, and `SUB_BUTTONS LIDACTION` to 0, and the machine STILL suspended the training fleet:
processes launched at 03:54 had accumulated 3,432 CPU-seconds by 10:55, i.e. 13.6% of wall
clock. **S0 idle is not the S3 timeout and does not obey it.** It throttles and suspends
processes when the system decides it is idle, and a long-running compute job that never
touches the input stack looks exactly like idle.

The supported way to say "do not do that" is `SetThreadExecutionState`, which is what media
players use to stop a film pausing. ES_CONTINUOUS makes the assertion persist until it is
cleared or the process exits, rather than resetting a one-shot timer.

    ES_SYSTEM_REQUIRED    the system may not sleep
    ES_AWAYMODE_REQUIRED  on Modern Standby, keep computing rather than entering away mode
    ES_CONTINUOUS         hold the assertion for the life of this process

The DISPLAY flag is deliberately NOT set: the screen should still be allowed to switch off,
which saves power and does not affect compute.

The assertion dies with the process, which is the desired behaviour -- kill this and the
machine returns to normal power management with nothing left behind to clean up.

    .venv/bin/python scripts/keep_awake.py            # hold until killed
    .venv/bin/python scripts/keep_awake.py --hours 8  # hold, then release and exit
"""
from __future__ import annotations

import argparse
import ctypes
import sys
import time

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_AWAYMODE_REQUIRED = 0x00000040


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hours", type=float, default=None,
                    help="release after this long; default is to hold until killed")
    args = ap.parse_args(argv)

    if not sys.platform.startswith("win"):
        print("not Windows; nothing to do")
        return 0

    kernel32 = ctypes.windll.kernel32
    flags = ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED
    if kernel32.SetThreadExecutionState(flags) == 0:
        # Away mode is refused on some configurations; the system-required assertion alone
        # is still worth having, so fall back rather than leaving the machine unprotected.
        flags = ES_CONTINUOUS | ES_SYSTEM_REQUIRED
        if kernel32.SetThreadExecutionState(flags) == 0:
            print("SetThreadExecutionState FAILED -- machine is NOT held awake")
            return 1
        print("holding awake (system-required only; away mode refused)", flush=True)
    else:
        print("holding awake (system-required + away mode)", flush=True)

    deadline = None if args.hours is None else time.time() + args.hours * 3600
    try:
        while deadline is None or time.time() < deadline:
            # Re-assert periodically. ES_CONTINUOUS should persist on its own, but a cheap
            # re-assert costs nothing and survives anything that clears it out from under us.
            kernel32.SetThreadExecutionState(flags)
            time.sleep(60)
    except KeyboardInterrupt:
        pass
    finally:
        kernel32.SetThreadExecutionState(ES_CONTINUOUS)   # release
        print("released", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
