//! Which device each node runs on (task 9.1): the `device="auto"` rules.
//!
//! Does: turn the device a plan asks for ([`DeviceRequest`], set by `on(device=...)`),
//! the devices found on the machine ([`Devices`], an argument: `PhysicalOptions`) and
//! the kernels each device has (`kyto_gpu::Device::has_kernel`) into a
//! [`Placement`] per node, which `explain` shows. The rules, fixed and in this order:
//!
//! 1. **Host steps run on the CPU**, whatever was asked: steps that move or select data
//!    without arithmetic on the matrix (`scan`, `with_default_layer`, `on`,
//!    `select_genes`, `take_cells`, a sink, a filter or computed column on metadata
//!    only). They have no device kernel to miss.
//! 2. **`device="cpu"`** (and a plan without `on`): every step on the CPU.
//! 3. **`device="auto"`**: each step on the first GPU found, in the order CUDA, then the
//!    wgpu adapter (Metal, Vulkan, DX12), that has a kernel for it (a fused pass: for all
//!    of its maps); otherwise on the CPU, and `explain` says so: `cpu (auto: no gpu
//!    found)` or `cpu (auto: no gpu kernel)`. Never silent.
//! 4. **`device="cuda"`, `"metal"` or `"wgpu"`** (any wgpu adapter): every step that is
//!    not a host step on that device. If the machine (or the build) has no such device,
//!    the step shows `(not found)` and the run raises `KY-E504`; if the device has no
//!    kernel for the step, it shows `(no kernel)` and the run raises `KY-E501`, before
//!    anything is read (task 9.6 keeps these rules).
//!
//! The data size plays no part, so the same plan, request and devices always give the
//! same placement.
//! Does not: find the devices (`kyto_gpu::probe`), or move data between devices (9.2).

use std::fmt;

use kyto_error::KytoError;
use kyto_gpu::{Device, Devices, WgpuBackend};
use serde::{Deserialize, Serialize};

use crate::filter::Predicate;
use crate::node::LogicalNode;
use crate::optimized_plan::{Operation, OptimizedNode};
use crate::plan::Plan;
use crate::with_columns::Columns;

/// The device a plan asks for (`on(device=...)`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DeviceRequest {
    /// The CPU (the default).
    #[default]
    Cpu,
    /// Each step on the first GPU found with a kernel for it, else the CPU (rule 3).
    Auto,
    /// An NVIDIA GPU through CUDA.
    Cuda,
    /// A GPU through wgpu: Metal, Vulkan or DX12, whichever was found.
    Wgpu,
    /// An Apple GPU through wgpu's Metal backend.
    Metal,
}

/// The names `on(device=...)` accepts, as the error lists them.
pub const DEVICE_NAMES: &str = "'cpu', 'auto', 'cuda', 'wgpu' or 'metal'";

impl DeviceRequest {
    /// The name users pass: `cpu`, `auto`, `cuda`, `wgpu` or `metal`.
    pub fn name(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Auto => "auto",
            Self::Cuda => "cuda",
            Self::Wgpu => "wgpu",
            Self::Metal => "metal",
        }
    }

    /// The request named `name` (case-sensitive).
    ///
    /// # Errors
    /// `KY-E215` for any other name, listing the accepted ones.
    pub fn parse(name: &str) -> Result<Self, KytoError> {
        match name {
            "cpu" => Ok(Self::Cpu),
            "auto" => Ok(Self::Auto),
            "cuda" => Ok(Self::Cuda),
            "wgpu" => Ok(Self::Wgpu),
            "metal" => Ok(Self::Metal),
            _ => Err(KytoError::InvalidArgument {
                function: "on".to_owned(),
                parameter: "device".to_owned(),
                value: format!("'{name}'"),
                problem: "it is not a device kyto knows".to_owned(),
                accepted: DEVICE_NAMES.to_owned(),
            }),
        }
    }

    /// The device an explicit request names among `devices`, `None` if it was not found
    /// (or for `cpu` and `auto`, which name no single GPU).
    fn named_gpu(self, devices: Devices) -> Option<Device> {
        match self {
            Self::Cpu | Self::Auto => None,
            Self::Cuda => devices.cuda().then_some(Device::Cuda),
            Self::Wgpu => devices.wgpu().map(Device::Wgpu),
            Self::Metal => (devices.wgpu() == Some(WgpuBackend::Metal))
                .then_some(Device::Wgpu(WgpuBackend::Metal)),
        }
    }

    /// Why `devices` has no device for this explicit request.
    fn not_found(self) -> String {
        let (runtime, feature) = match self {
            Self::Cuda => (kyto_gpu::Runtime::Cuda, "cuda"),
            _ => (kyto_gpu::Runtime::Wgpu, "wgpu"),
        };
        if runtime.backend().built() {
            match self {
                Self::Cuda => "no NVIDIA GPU or CUDA driver was found".to_owned(),
                Self::Metal => "no Metal GPU was found".to_owned(),
                _ => "no Metal, Vulkan or DX12 GPU was found".to_owned(),
            }
        } else {
            format!("this kyto was built without the '{feature}' feature")
        }
    }
}

impl fmt::Display for DeviceRequest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.name())
    }
}

/// Where one node runs, and why when it is not where it was asked to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Placement {
    /// On this device: asked for, picked by `auto`, or a host step's CPU (rules 1-4).
    On(Device),
    /// On the CPU because `auto` found no GPU with a kernel for it (rule 3);
    /// `gpu_found` says whether any GPU was found at all.
    Fallback {
        /// Whether the machine has a GPU (without a kernel for this step).
        gpu_found: bool,
    },
    /// The device asked for has no kernel for `operation`: the run raises `KY-E501`.
    NoKernel {
        /// The device asked for.
        device: Device,
        /// The step without a kernel (a plan method, such as `log1p`).
        operation: &'static str,
    },
    /// The device asked for was not found: the run raises `KY-E504`.
    NotFound(DeviceRequest),
}

impl Placement {
    /// The device it runs on, `None` when the run is refused (`KY-E501`, `KY-E504`).
    pub fn device(self) -> Option<Device> {
        match self {
            Self::On(device) => Some(device),
            Self::Fallback { .. } => Some(Device::Cpu),
            Self::NoKernel { .. } | Self::NotFound(_) => None,
        }
    }

    /// The device name `explain` shows: where it runs, or the device asked for.
    pub fn name(self) -> &'static str {
        match self {
            Self::On(device) | Self::NoKernel { device, .. } => device.name(),
            Self::Fallback { .. } => Device::Cpu.name(),
            Self::NotFound(request) => request.name(),
        }
    }

    /// Why it runs elsewhere than asked, or cannot run (`None` when it runs as asked).
    pub fn note(self) -> Option<&'static str> {
        match self {
            Self::On(_) => None,
            Self::Fallback { gpu_found: false } => Some("auto: no gpu found"),
            Self::Fallback { gpu_found: true } => Some("auto: no gpu kernel"),
            Self::NoKernel { .. } => Some("no kernel: KY-E501 when run"),
            Self::NotFound(_) => Some("not found: KY-E504 when run"),
        }
    }

    /// The error a run of this node raises, if it cannot run as asked.
    pub fn error(self, devices: Devices) -> Option<KytoError> {
        match self {
            Self::On(_) | Self::Fallback { .. } => None,
            Self::NoKernel { device, operation } => Some(KytoError::MissingKernel {
                operation: operation.to_owned(),
                device: device.name().to_owned(),
            }),
            Self::NotFound(request) => Some(KytoError::DeviceNotFound {
                device: request.name().to_owned(),
                problem: request.not_found(),
                found: devices.to_string(),
            }),
        }
    }
}

/// `metal`, or `cpu (auto: no gpu kernel)` with the note.
impl fmt::Display for Placement {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.name())?;
        if let Some(note) = self.note() {
            write!(formatter, " ({note})")?;
        }
        Ok(())
    }
}

/// The placement of a step doing `operations` (empty for a host step) when `request`
/// is asked for on a machine with `devices`, where `has_kernel` says which device has
/// a kernel for which operation (the module's rules).
pub fn place(
    request: DeviceRequest,
    devices: Devices,
    operations: &[&'static str],
    has_kernel: impl Fn(Device, &str) -> bool,
) -> Placement {
    let runs_on = |device: Device| {
        operations
            .iter()
            .all(|&operation| has_kernel(device, operation))
    };
    if operations.is_empty() {
        return Placement::On(Device::Cpu);
    }
    match request {
        DeviceRequest::Cpu => Placement::On(Device::Cpu),
        DeviceRequest::Auto => devices.gpus().find(|&gpu| runs_on(gpu)).map_or(
            Placement::Fallback {
                gpu_found: devices.gpus().next().is_some(),
            },
            Placement::On,
        ),
        DeviceRequest::Cuda | DeviceRequest::Wgpu | DeviceRequest::Metal => {
            match request.named_gpu(devices) {
                None => Placement::NotFound(request),
                Some(device) => match operations
                    .iter()
                    .find(|&&operation| !has_kernel(device, operation))
                {
                    Some(&operation) => Placement::NoKernel { device, operation },
                    None => Placement::On(device),
                },
            }
        }
    }
}

/// The operations of an optimized node that need a device kernel, by plan method
/// (`log1p`, `pca`); empty for a host step (rule 1).
pub(crate) fn operations(node: &OptimizedNode) -> Vec<&'static str> {
    match node.operation() {
        Operation::Scan { .. } => Vec::new(),
        Operation::Fused { maps, .. } => maps.iter().map(LogicalNode::method).collect(),
        Operation::Node(plan) => {
            let host = match plan.as_ref() {
                Plan::Scan(_)
                | Plan::WithDefaultLayer(_)
                | Plan::OnDevice(_)
                | Plan::SelectGenes(_)
                | Plan::TakeCells(_)
                | Plan::Sink(_) => true,
                Plan::Filter(filter) => match filter.params().predicate() {
                    Predicate::Cells(predicate) => predicate.is_metadata_only(),
                    Predicate::Genes(predicate) => predicate.is_metadata_only(),
                },
                Plan::WithColumns(with_columns) => match with_columns.params().columns() {
                    Columns::Cells(columns) => columns.iter().all(|(_, e)| e.is_metadata_only()),
                    Columns::Genes(columns) => columns.iter().all(|(_, e)| e.is_metadata_only()),
                },
                Plan::MapElementwise(_)
                | Plan::CrossCell(_)
                | Plan::Neighbors(_)
                | Plan::Cluster(_)
                | Plan::Umap(_) => false,
            };
            if host {
                Vec::new()
            } else {
                vec![plan.node().method()]
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const METAL: Device = Device::Wgpu(WgpuBackend::Metal);
    const NONE: Devices = Devices::CPU_ONLY;
    const MAC: Devices = Devices::new(false, Some(WgpuBackend::Metal));
    const BOTH: Devices = Devices::new(true, Some(WgpuBackend::Vulkan));

    /// A stand-in kernel table: CUDA runs `log1p` and `pca`, wgpu only `log1p`.
    fn table(device: Device, operation: &str) -> bool {
        match device {
            Device::Cpu => true,
            Device::Cuda => matches!(operation, "log1p" | "pca"),
            Device::Wgpu(_) => operation == "log1p",
        }
    }

    #[test]
    fn host_steps_and_cpu_stay_on_the_cpu() {
        for request in [
            DeviceRequest::Cuda,
            DeviceRequest::Auto,
            DeviceRequest::Metal,
        ] {
            assert_eq!(place(request, NONE, &[], table), Placement::On(Device::Cpu));
        }
        let cpu = place(DeviceRequest::Cpu, BOTH, &["log1p"], table);
        assert_eq!(cpu, Placement::On(Device::Cpu));
        assert_eq!(cpu.to_string(), "cpu");
    }

    #[test]
    fn auto_prefers_cuda_then_wgpu_then_falls_back_out_loud() {
        let auto = |devices, operations: &[&'static str]| {
            place(DeviceRequest::Auto, devices, operations, table)
        };
        assert_eq!(auto(BOTH, &["pca"]), Placement::On(Device::Cuda));
        assert_eq!(auto(MAC, &["log1p"]), Placement::On(METAL));
        assert_eq!(auto(MAC, &["log1p"]).to_string(), "metal");
        // A fused pass goes to a GPU only if it has every map's kernel.
        assert_eq!(
            auto(MAC, &["normalize_total", "log1p"]).to_string(),
            "cpu (auto: no gpu kernel)"
        );
        assert_eq!(auto(MAC, &["pca"]).device(), Some(Device::Cpu));
        assert_eq!(
            auto(NONE, &["log1p"]).to_string(),
            "cpu (auto: no gpu found)"
        );
        // The real table has no GPU kernel before 9.2: everything falls back.
        let real = place(DeviceRequest::Auto, MAC, &["log1p"], Device::has_kernel);
        assert_eq!(real, Placement::Fallback { gpu_found: true });
    }

    #[test]
    fn an_explicit_device_is_used_or_refused() {
        assert_eq!(
            place(DeviceRequest::Cuda, BOTH, &["pca"], table),
            Placement::On(Device::Cuda)
        );
        assert_eq!(
            place(DeviceRequest::Wgpu, BOTH, &["log1p"], table),
            Placement::On(Device::Wgpu(WgpuBackend::Vulkan))
        );
        let missing = place(DeviceRequest::Metal, MAC, &["pca"], table);
        assert_eq!(
            missing,
            Placement::NoKernel {
                device: METAL,
                operation: "pca"
            }
        );
        assert_eq!(missing.to_string(), "metal (no kernel: KY-E501 when run)");
        assert_eq!(missing.device(), None);
        let error = missing.error(MAC).map(|error| error.code());
        assert_eq!(error, Some("KY-E501"));
        // Metal is not Vulkan: `metal` on a Linux GPU is not found.
        let absent = place(DeviceRequest::Metal, BOTH, &["log1p"], table);
        assert_eq!(absent, Placement::NotFound(DeviceRequest::Metal));
        assert_eq!(absent.to_string(), "metal (not found: KY-E504 when run)");
        let error = absent.error(BOTH).map(|error| error.to_string());
        assert!(
            error.is_some_and(|text| text.contains("cpu, cuda, vulkan")),
            "the error lists what was found"
        );
        assert_eq!(
            place(DeviceRequest::Cuda, NONE, &["pca"], table).name(),
            "cuda"
        );
    }

    #[test]
    fn requests_parse_and_print() {
        for name in ["cpu", "auto", "cuda", "wgpu", "metal"] {
            assert_eq!(
                DeviceRequest::parse(name).map(DeviceRequest::name).ok(),
                Some(name)
            );
        }
        let error = DeviceRequest::parse("gpu").err().map(|error| error.code());
        assert_eq!(error, Some("KY-E215"));
        assert_eq!(DeviceRequest::default(), DeviceRequest::Cpu);
    }
}
