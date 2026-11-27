//! The `Backend` interface: one implementation per runtime (CPU, CUDA, wgpu).
//!
//! Does: say whether a runtime is built in, find its device on this machine, and say
//! which operations it has a kernel for, by the plan method that names the step
//! (`log1p`, `pca`, as `explain` and `KY-E501` name them).
//! Does not: run kernels (`launch.rs`, `kernels.rs`) or choose among devices
//! (kyto-plan's `device_placement.rs`).
//!
//! **Adding a kernel** (9.2, 9.3, 9.8): implement it, add its operation to the
//! runtime's list below ([`CUDA_KERNELS`], [`WGPU_KERNELS`]), and dispatch the step to
//! it in kyto-exec (`Pipeline::new` refuses a GPU step it cannot dispatch with
//! `KY-E501`); the physical plan then places that step on the GPU, and `explain` shows
//! it.

use crate::device::{Device, WgpuBackend};

/// A family of devices that share their kernels: CubeCL compiles one kernel for every
/// wgpu graphics API, so Metal, Vulkan and DX12 share [`Runtime::Wgpu`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Runtime {
    /// The CPU executor.
    Cpu,
    /// CUDA (NVIDIA).
    Cuda,
    /// wgpu (Metal, Vulkan, DX12).
    Wgpu,
}

impl Runtime {
    /// The name users pass as `device=` for it: `cpu`, `cuda` or `wgpu`.
    pub fn name(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
            Self::Wgpu => "wgpu",
        }
    }

    /// The runtime of `device`.
    pub fn of(device: Device) -> Self {
        match device {
            Device::Cpu => Self::Cpu,
            Device::Cuda => Self::Cuda,
            Device::Wgpu(_) => Self::Wgpu,
        }
    }

    /// Its [`Backend`] implementation.
    pub fn backend(self) -> &'static dyn Backend {
        match self {
            Self::Cpu => &Cpu,
            Self::Cuda => &Cuda,
            Self::Wgpu => &Wgpu,
        }
    }
}

/// One runtime kyto can run on: whether it is built in, where its device is, and what
/// it runs.
pub trait Backend: Sync {
    /// The runtime it implements.
    fn runtime(&self) -> Runtime;

    /// The cargo feature that builds it in (`None` for the CPU, always built).
    fn feature(&self) -> Option<&'static str>;

    /// Whether this build has it (its [`Backend::feature`] was on).
    fn built(&self) -> bool;

    /// Its device on this machine, if found. Asks the driver: call it through
    /// [`crate::probe`], which asks once per process.
    fn detect(&self) -> Option<Device>;

    /// Whether it has a kernel for `operation` (a plan method such as `log1p`).
    fn has_kernel(&self, operation: &str) -> bool;
}

/// The operations with a CUDA kernel: the per-cell maps of 9.2 (CubeCL, `kernels.rs`),
/// the per-cell QC metrics of 9.10 (9.2's row sums and nnz; named by the expression
/// leaf, as a `with_cells` or `filter_cells` step reads them, see kyto-plan's
/// `device_placement.rs`), and `pca` and exact `neighbors` through 9.8's CubeCL matrix
/// kernels (`kernels_matrix.rs`) until 9.3 makes cuSPARSE and cuBLAS CUDA's path for
/// `pca` (the same entry points, `launch_matrix.rs`). Approximate neighbours
/// ([`APPROXIMATE_NEIGHBORS`]) have no GPU kernel.
pub const CUDA_KERNELS: &[&str] = &[
    "normalize_total",
    "log1p",
    "total_counts",
    "genes_detected",
    "total_counts_in",
    "percent_counts_in",
    "pca",
    "neighbors",
];

/// The operations with a wgpu kernel (Metal, Vulkan, DX12): the same CubeCL kernels as
/// CUDA's, 9.2's maps, 9.10's QC metrics and 9.8's `pca` and exact `neighbors`.
pub const WGPU_KERNELS: &[&str] = &[
    "normalize_total",
    "log1p",
    "total_counts",
    "genes_detected",
    "total_counts_in",
    "percent_counts_in",
    "pca",
    "neighbors",
];

/// The operation of a `neighbors` step with `search="approximate"` (the HNSW index,
/// 6.17): no GPU has a kernel for it, so `auto` keeps it on the CPU and a named GPU
/// refuses it (`KY-E501`). Exact search is `neighbors`.
pub const APPROXIMATE_NEIGHBORS: &str = "neighbors(search=approximate)";

/// How the result of an operation on a GPU compares with the CPU's: the note `explain`
/// shows on a GPU step (kyto-plan's `Placement::note`). Ordered from the closest: a step
/// of several operations shows its loosest.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Agreement {
    /// The CPU's result exactly: exact `neighbors` (9.8: the GPU proposes candidates,
    /// the host ranks them with the CPU's distances and searches a row again when the
    /// candidates cannot be shown to hold its neighbours).
    Equal,
    /// Within 1e-6 relative of the CPU (`f32` arithmetic): 9.2's maps and 9.10's QC
    /// metrics.
    Close,
    /// Within the PCA parity tolerances of the CPU (`tests/parity/tolerances.toml`,
    /// `pca.*`: 1e-3 relative on variances, loadings and embedding): 9.8's `pca`, whose
    /// products are `f32` (compensated sums) where the CPU's are `f64`.
    PcaTolerance,
}

impl Agreement {
    /// The agreement of `operation` on a GPU.
    pub fn of(operation: &str) -> Self {
        match operation {
            "neighbors" => Self::Equal,
            "pca" => Self::PcaTolerance,
            _ => Self::Close,
        }
    }

    /// The note `explain` shows: `equal to cpu`, `within 1e-6 of cpu`, `within 1e-3 of
    /// cpu`.
    pub fn note(self) -> &'static str {
        match self {
            Self::Equal => "equal to cpu",
            Self::Close => "within 1e-6 of cpu",
            Self::PcaTolerance => "within 1e-3 of cpu",
        }
    }
}

/// Every backend, in the order `device="auto"` prefers their GPUs after the CPU's
/// place: CPU, CUDA, wgpu.
pub const BACKENDS: [&dyn Backend; 3] = [&Cpu, &Cuda, &Wgpu];

/// The CPU: always built, always found, a kernel for everything.
#[derive(Debug, Clone, Copy)]
pub struct Cpu;

impl Backend for Cpu {
    fn runtime(&self) -> Runtime {
        Runtime::Cpu
    }

    fn feature(&self) -> Option<&'static str> {
        None
    }

    fn built(&self) -> bool {
        true
    }

    fn detect(&self) -> Option<Device> {
        Some(Device::Cpu)
    }

    fn has_kernel(&self, _operation: &str) -> bool {
        true
    }
}

/// An NVIDIA GPU through the CUDA driver (feature `cuda`).
#[derive(Debug, Clone, Copy)]
pub struct Cuda;

impl Backend for Cuda {
    fn runtime(&self) -> Runtime {
        Runtime::Cuda
    }

    fn feature(&self) -> Option<&'static str> {
        Some("cuda")
    }

    fn built(&self) -> bool {
        cfg!(feature = "cuda")
    }

    fn detect(&self) -> Option<Device> {
        detect_cuda()
    }

    fn has_kernel(&self, operation: &str) -> bool {
        CUDA_KERNELS.contains(&operation)
    }
}

/// A GPU through wgpu: Metal, Vulkan or DX12 (feature `wgpu`).
#[derive(Debug, Clone, Copy)]
pub struct Wgpu;

impl Backend for Wgpu {
    fn runtime(&self) -> Runtime {
        Runtime::Wgpu
    }

    fn feature(&self) -> Option<&'static str> {
        Some("wgpu")
    }

    fn built(&self) -> bool {
        cfg!(feature = "wgpu")
    }

    fn detect(&self) -> Option<Device> {
        detect_wgpu().map(Device::Wgpu)
    }

    fn has_kernel(&self, operation: &str) -> bool {
        WGPU_KERNELS.contains(&operation)
    }
}

impl Device {
    /// Whether this device has a kernel for `operation` (a plan method such as `log1p`).
    pub fn has_kernel(self, operation: &str) -> bool {
        Runtime::of(self).backend().has_kernel(operation)
    }
}

/// A CUDA GPU, if the driver library loads, initializes and counts at least one device.
#[cfg(feature = "cuda")]
fn detect_cuda() -> Option<Device> {
    use cudarc::driver::{result, sys};
    // SAFETY: only tries to open the driver library by its usual names (`libcuda.so`,
    // `nvcuda.dll`) and drops it; no symbol is called. Checking first keeps cudarc from
    // panicking below on a machine without the driver.
    if !unsafe { sys::is_culib_present() } {
        return None;
    }
    result::init().ok()?;
    let count = result::device::get_count().ok()?;
    (count > 0).then_some(Device::Cuda)
}

/// Built without the `cuda` feature: no CUDA.
#[cfg(not(feature = "cuda"))]
fn detect_cuda() -> Option<Device> {
    None
}

/// The graphics API of the best wgpu GPU adapter: a discrete GPU before an integrated
/// one before a virtual one, then Metal, Vulkan, DX12. Software (CPU) adapters do not
/// count.
#[cfg(feature = "wgpu")]
fn detect_wgpu() -> Option<WgpuBackend> {
    use wgpu::{Backends, DeviceType, Instance, InstanceDescriptor, InstanceFlags};
    let backends = Backends::METAL | Backends::VULKAN | Backends::DX12;
    let instance = Instance::new(InstanceDescriptor {
        backends,
        // No validation layers or debug labels: the probe only lists adapters.
        flags: InstanceFlags::empty(),
        ..InstanceDescriptor::new_without_display_handle()
    });
    let adapters = pollster::block_on(instance.enumerate_adapters(backends));
    adapters
        .iter()
        .filter_map(|adapter| {
            let info = adapter.get_info();
            let kind = match info.device_type {
                DeviceType::DiscreteGpu => 0,
                DeviceType::IntegratedGpu => 1,
                DeviceType::VirtualGpu => 2,
                DeviceType::Other => 3,
                DeviceType::Cpu => return None,
            };
            let backend = match info.backend {
                wgpu::Backend::Metal => WgpuBackend::Metal,
                wgpu::Backend::Vulkan => WgpuBackend::Vulkan,
                wgpu::Backend::Dx12 => WgpuBackend::Dx12,
                _ => return None,
            };
            Some((kind, backend))
        })
        .min()
        .map(|(_, backend)| backend)
}

/// Built without the `wgpu` feature: no wgpu adapter.
#[cfg(not(feature = "wgpu"))]
fn detect_wgpu() -> Option<WgpuBackend> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_has_every_kernel_and_gpus_the_9_2_maps_9_10_qc_and_9_8_matrix_steps() {
        assert!(Device::Cpu.has_kernel("pca"));
        for gpu in [Device::Cuda, Device::Wgpu(WgpuBackend::Metal)] {
            assert!(gpu.has_kernel("log1p") && gpu.has_kernel("normalize_total"));
            for metric in [
                "total_counts",
                "genes_detected",
                "total_counts_in",
                "percent_counts_in",
            ] {
                assert!(gpu.has_kernel(metric), "{metric}");
            }
            assert!(gpu.has_kernel("pca") && gpu.has_kernel("neighbors"));
            assert!(!gpu.has_kernel(APPROXIMATE_NEIGHBORS) && !gpu.has_kernel("scale"));
            assert!(!gpu.has_kernel("with_cells") && !gpu.has_kernel("filter_cells"));
        }
        for backend in BACKENDS {
            assert_eq!(backend.runtime().backend().runtime(), backend.runtime());
        }
        assert!(Cpu.built() && Cpu.feature().is_none());
        assert_eq!(
            (Cuda.built(), Cuda.feature()),
            (cfg!(feature = "cuda"), Some("cuda"))
        );
        assert_eq!(
            (Wgpu.built(), Wgpu.feature()),
            (cfg!(feature = "wgpu"), Some("wgpu"))
        );
    }

    #[test]
    fn agreements_order_from_equal_to_the_pca_tolerance() {
        assert_eq!(Agreement::of("neighbors"), Agreement::Equal);
        assert_eq!(Agreement::of("log1p").note(), "within 1e-6 of cpu");
        let loosest = ["neighbors", "log1p", "pca"]
            .map(Agreement::of)
            .into_iter()
            .max();
        assert_eq!(loosest, Some(Agreement::PcaTolerance));
    }

    #[test]
    fn runtimes_of_devices() {
        assert_eq!(
            Runtime::of(Device::Wgpu(WgpuBackend::Vulkan)),
            Runtime::Wgpu
        );
        assert_eq!(Runtime::of(Device::Cuda).name(), "cuda");
    }
}
