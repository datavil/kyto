"""Run a script as __main__, then print its process's peak resident memory in MB."""

import resource
import runpy
import sys

if __name__ == "__main__":
    script_path = sys.argv[1]
    sys.argv = sys.argv[1:]
    runpy.run_path(script_path, run_name="__main__")
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is in bytes on macOS and in kilobytes on Linux.
    peak_mb = peak / 1e6 if sys.platform == "darwin" else peak / 1e3
    print(f"peak_rss_mb={peak_mb:.0f}")
