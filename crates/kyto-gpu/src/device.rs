//! The devices a plan's nodes run on, and the set of them found on a machine.
//!
//! Does: name each device as `explain` shows it ([`Device::name`]), and hold what a
//! probe found ([`Devices`], plain data: tests build it by hand to plan for a machine
//! with or without a GPU).
//! Does not: probe the machine (`probe.rs`) or say which kernels a device has
//! (`backend.rs`).

use std::fmt;

/// The graphics API a wgpu adapter runs on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum WgpuBackend {
    /// Apple's Metal (macOS, Apple silicon and AMD GPUs).
    Metal,
    /// Vulkan (Linux, Windows; AMD, Intel and NVIDIA GPUs).
    Vulkan,
    /// Direct3D 12 (Windows).
    Dx12,
}

/// Where a node runs (R7: chosen from an argument, never a global setting).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Device {
    /// The CPU: the streaming executor of 5.x. Every operation has a kernel here.
    #[default]
    Cpu,
    /// An NVIDIA GPU through CUDA (CubeCL's CUDA runtime, cuSPARSE and cuBLAS).
    Cuda,
    /// A GPU through wgpu (CubeCL's wgpu runtime) on the given graphics API.
    Wgpu(WgpuBackend),
}

impl Device {
    /// The name `explain` shows: `cpu`, `cuda`, `metal`, `vulkan` or `dx12`.
    pub fn name(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
            Self::Wgpu(WgpuBackend::Metal) => "metal",
            Self::Wgpu(WgpuBackend::Vulkan) => "vulkan",
            Self::Wgpu(WgpuBackend::Dx12) => "dx12",
        }
    }

    /// Whether it is a GPU.
    pub fn is_gpu(self) -> bool {
        self != Self::Cpu
    }
}

impl fmt::Display for Device {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.name())
    }
}

/// The devices found on a machine: always the CPU, and at most one CUDA GPU and one wgpu
/// adapter (the first GPU of each, task 9.1: one GPU per run).
///
/// [`Default`] is the CPU alone, the same on every machine; [`crate::probe`] fills it
/// from this one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Devices {
    cuda: bool,
    wgpu: Option<WgpuBackend>,
}

impl Devices {
    /// The CPU alone (a machine without a GPU, or a build without GPU features).
    pub const CPU_ONLY: Self = Self {
        cuda: false,
        wgpu: None,
    };

    /// The CPU, with a CUDA GPU if `cuda` and a wgpu adapter on `wgpu` if given.
    pub const fn new(cuda: bool, wgpu: Option<WgpuBackend>) -> Self {
        Self { cuda, wgpu }
    }

    /// Whether `device` was found (the CPU always is).
    pub fn has(self, device: Device) -> bool {
        match device {
            Device::Cpu => true,
            Device::Cuda => self.cuda,
            Device::Wgpu(backend) => self.wgpu == Some(backend),
        }
    }

    /// Whether a CUDA GPU was found.
    pub fn cuda(self) -> bool {
        self.cuda
    }

    /// The wgpu adapter's graphics API, if one was found.
    pub fn wgpu(self) -> Option<WgpuBackend> {
        self.wgpu
    }

    /// The GPUs found, in the order `device="auto"` tries them: CUDA first, then the
    /// wgpu adapter.
    pub fn gpus(self) -> impl Iterator<Item = Device> {
        let cuda = self.cuda.then_some(Device::Cuda);
        cuda.into_iter().chain(self.wgpu.map(Device::Wgpu))
    }
}

/// `cpu`, then the GPUs found: `cpu, cuda, metal`.
impl fmt::Display for Devices {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("cpu")?;
        for gpu in self.gpus() {
            write!(formatter, ", {gpu}")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_and_order() {
        let devices = Devices::new(true, Some(WgpuBackend::Metal));
        assert_eq!(devices.to_string(), "cpu, cuda, metal");
        assert_eq!(
            devices.gpus().collect::<Vec<_>>(),
            [Device::Cuda, Device::Wgpu(WgpuBackend::Metal)]
        );
        assert!(devices.has(Device::Cpu));
        assert!(!devices.has(Device::Wgpu(WgpuBackend::Vulkan)));
        assert_eq!(Devices::CPU_ONLY.to_string(), "cpu");
        assert_eq!(Devices::default(), Devices::CPU_ONLY);
        assert!(!Device::Cpu.is_gpu());
        assert_eq!(Device::Wgpu(WgpuBackend::Dx12).name(), "dx12");
    }
}
