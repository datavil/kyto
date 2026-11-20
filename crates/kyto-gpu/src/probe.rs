//! Finding this machine's devices.
//!
//! Does: ask each built-in [`crate::Backend`] for its device once per process and keep
//! the answer ([`probe`]): the hardware does not change while kyto runs, and asking the
//! drivers again would cost every `explain()` tens of milliseconds. The answer is a fact
//! about the machine, not a setting (R7): no function changes it, and plans take it as
//! an argument (kyto-plan's `PhysicalOptions::devices`), so tests plan for any machine.
//! Does not: choose a device for a plan (kyto-plan).

use std::sync::OnceLock;

use crate::backend::{Cuda, Wgpu};
use crate::device::{Device, Devices};
use crate::{Backend, WgpuBackend};

/// The devices on this machine that this build can use: the CPU always, a CUDA GPU with
/// the `cuda` feature, a wgpu adapter (Metal, Vulkan, DX12) with the `wgpu` feature.
/// Asked once per process, then the same answer.
pub fn probe() -> Devices {
    static DEVICES: OnceLock<Devices> = OnceLock::new();
    *DEVICES.get_or_init(|| {
        let cuda = Cuda.detect() == Some(Device::Cuda);
        let wgpu = match Wgpu.detect() {
            Some(Device::Wgpu(backend)) => Some(backend),
            _ => None::<WgpuBackend>,
        };
        Devices::new(cuda, wgpu)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_finds_only_what_is_built_and_answers_the_same() {
        let devices = probe();
        assert!(devices.has(Device::Cpu));
        if !Cuda.built() {
            assert!(!devices.cuda());
        }
        if !Wgpu.built() {
            assert_eq!(devices.wgpu(), None);
        }
        assert_eq!(probe(), devices);
    }

    /// On an Apple silicon Mac built with `wgpu`, the probe finds Metal. Ignored: CI's
    /// macOS runners are virtual machines without a GPU (run on purpose:
    /// `cargo test -p kyto-gpu --features wgpu -- --ignored`).
    #[test]
    #[ignore = "needs a Metal GPU and the wgpu feature"]
    fn finds_metal_on_apple_silicon() {
        assert!(Wgpu.built(), "build with --features wgpu");
        assert_eq!(probe().wgpu(), Some(WgpuBackend::Metal));
    }
}
