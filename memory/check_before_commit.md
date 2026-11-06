# Never let a failing check reach a commit

(from ggrust)

**Broke (in ggrust):** `make check 2>&1 | grep ...; git add -A && git commit ...`
committed and pushed a lint failure: the pipe hid the check's exit status and `;` ran
the commit anyway.

**Works:** gate the commit on the check's own status:
`pre-commit run --all-files > <scratch>/check.log 2>&1 && git add -A && git commit ...`
(then inspect the log), or `set -o pipefail`. If a bad commit slipped out, fix forward
in a new commit and say so.
