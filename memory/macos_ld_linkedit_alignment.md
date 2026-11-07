# Release extension fails to load on macOS 27: build against the 26.5 SDK (task 2.7)

**Broke (Nov 7, 2026, macOS 27.0.1, Xcode / CLT 27 with ld-27037, rustc 1.97.1):** the
release `_kyto.abi3.so` that `uv sync` builds fails `import kyto` with
`dlopen ... (mis-aligned LINKEDIT string pool, fileOffset=0x...4)`. HEAD of 2.6 fails the
same way when rebuilt, so it is the toolchain, not a task. `otool -l` shows `LC_SYMTAB
stroff` 4 bytes off an 8-byte boundary (the string pool follows the 4-byte indirect
symbol table unpadded). The debug build loads. Not fixed by `-Wl,-ld_classic`,
`-Wl,-no_fixup_chains`, `strip = "symbols"` or `opt-level = 2`. rustup's `rust-lld`
cannot read the 27 SDK's `libSystem.tbd` (`unknown architecture arm64e.x1-macos`).

**Works:** link against the older SDK that the Command Line Tools still ship, with the
normal `ld`:

```bash
export SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk
```

With it, the release build loads and the whole local gate passes. It is a machine
setting, not repo config: the path exists only where those CLT are installed, and CI's
macOS runners use their own Xcode. Keep it in the shell profile of this machine. After
changing `SDKROOT`, touch a kyto-py source so cargo relinks the extension.

**Also:** `rust-lld` (`-C link-arg=-fuse-ld=<rustup>/gcc-ld/ld64.lld`) with the same
`SDKROOT` loads too; not needed. Drop this entry once an Xcode update fixes ld for the
27 SDK (check: unset `SDKROOT`, rebuild, `import kyto`).
