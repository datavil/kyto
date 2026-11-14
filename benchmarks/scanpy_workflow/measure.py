"""Run a script as __main__, then print its process's peak resident memory in MB."""

import resource
import runpy
import sys

if __name__ == "__main__":
    script_path = sys.argv[1]
    sys.argv = sys.argv[1:]
    runpy.run_path(script_path, run_name="__main__")
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"peak_rss_mb={peak_mb:.0f}")
