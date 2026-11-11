# GIL, progress and Ctrl-C: a worker thread and a polling Python thread (task 5.7)

**Works:** run the plan (and drain its stream) on a worker thread; the calling Python
thread loops `py.detach(|| receiver.recv_timeout(20 ms))`, then, holding the GIL,
`py.check_signals()` (runs Python's SIGINT handler: `KeyboardInterrupt`) and the
progress callback. On an exception: `RunControl::cancel()`, join the worker detached,
re-raise. The executor checks the cancel flag before each block read and each block
computed (`Windows::next_window`), so statistics passes stop too. Callbacks run on the
calling thread only: no GIL taken from executor threads, no callback re-entrancy.

**Watch out:**
- `py.detach` needs `Ungil` (= `Send` for the closure in the extension build, but not
  checked by a plain `cargo clippy`): a `mpsc::Receiver` is not `Sync`, so `&receiver`
  fails only under `uv sync`. Move the receiver into the closure and back out.
- `check_signals` does nothing off the main thread; Python signal handlers only run
  there. Tests simulate Ctrl-C with `_thread.interrupt_main(signal.SIGINT)` from another
  thread (portable) or `signal.raise_signal` inside the callback (raises there), and set
  `signal.default_int_handler` in a fixture.
- A scan of 10 million entries runs in ~0.15 s on one thread (M-series): a cancel test
  needs a store far above the 20 ms tick (the test uses 20 million entries, ~0.3 s) or
  the run ends before the cancel lands. Assert the last progress `done < total`.
- The reader reads a window of blocks at once, so after a cancel at most that window's
  reads finish; count reads per block from a full run before asserting a bound.
