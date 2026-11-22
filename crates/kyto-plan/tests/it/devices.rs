//! Task 9.1 Done when: `explain()` names the device per node on a machine with and
//! without a GPU.
//!
//! The devices found are an argument of the physical plan (`PhysicalOptions::devices`),
//! so one test plans for every machine: no GPU, an Apple GPU (Metal), an NVIDIA GPU
//! (CUDA), and both kinds. Each optimized node's line names its device; host steps stay
//! on the CPU; `auto` puts the fused `normalize_total` + `log1p` pass on the GPU (its 9.2
//! kernels, noted as not bit-identical to the CPU) and falls back to the CPU out loud for
//! `pca` (no GPU kernel before 9.3/9.8); an explicit GPU is refused before the run with
//! `KY-E504` (not found) or `KY-E501` (no kernel). The `on` step round-trips through the
//! plan JSON.
//!
//! Task 9.6 Done when: `device="cuda"` with a missing kernel raises `KY-E501`, and
//! `device="auto"` falls back to the CPU and says so in `explain()`, now that some steps
//! have kernels: a chain whose steps alternate between a GPU and the CPU (each step's
//! input computed on another device) shows the CPU steps' fallback notes and each GPU
//! step's block moves (`moves=cpu->metal->cpu`, rule 5: blocks live on the host between
//! steps), in text and JSON; `cuda` refuses the same chain with `KY-E501` naming the
//! first step without a kernel. The runs of such chains are in kyto-exec's
//! `pipeline.rs` (`a_chain_alternating_devices_...`) and `tests/test_devices.py`.
#![allow(clippy::unwrap_used)]

use crate::support;

use std::sync::Arc;

use kyto_core::{BinaryOp, CellExpr, Expr, Literal, QcMetric};
use kyto_plan::{
    Device, DeviceRequest, Devices, Explain, PhysicalOptions, PhysicalPlan, Plan, WgpuBackend,
};

use support::scan;

const NO_GPU: Devices = Devices::CPU_ONLY;
const MAC: Devices = Devices::new(false, Some(WgpuBackend::Metal));
const CUDA: Devices = Devices::new(true, None);
const BOTH: Devices = Devices::new(true, Some(WgpuBackend::Vulkan));

/// A metadata filter (a host step), a fused `normalize_total` + `log1p` pass and `pca`
/// (compute steps), then `on(device=request)`.
fn pipeline(request: DeviceRequest) -> Arc<Plan> {
    let score = Expr::binary(
        CellExpr::column("score"),
        BinaryOp::Gt,
        Literal::Float(-2.0),
    );
    scan()
        .filter_cells(&score.unwrap())
        .unwrap()
        .normalize_total(Some(1e4), Some("counts"), "norm")
        .unwrap()
        .log1p(Some("norm"), "logn")
        .unwrap()
        .pca(Some("logn"), "X_new", 2, 0, None)
        .unwrap()
        .on_device(request)
        .unwrap()
}

fn physical(plan: &Arc<Plan>, devices: Devices) -> PhysicalPlan {
    plan.physical(PhysicalOptions {
        devices,
        ..PhysicalOptions::default()
    })
}

/// The `[device=...` part of each optimized node line, by node name (top down).
fn devices_per_node(plan: &Arc<Plan>, devices: Devices) -> Vec<(String, String)> {
    let text = Explain::new(plan, &physical(plan, devices)).to_string();
    let optimized = text.split("optimized plan").nth(1).unwrap();
    optimized
        .lines()
        .filter(|line| line.trim_start().starts_with('#'))
        .map(|line| {
            let name = line.split_whitespace().nth(1).unwrap().to_owned();
            let device = line.split("[device=").nth(1).unwrap();
            let device = device.split(" passes=").next().unwrap().to_owned();
            (name, device)
        })
        .collect()
}

fn device_of<'a>(nodes: &'a [(String, String)], name: &str) -> &'a str {
    &nodes
        .iter()
        .find(|(node, _)| node == name)
        .unwrap_or_else(|| panic!("no {name} in {nodes:?}"))
        .1
}

#[test]
fn every_node_names_its_device_with_and_without_a_gpu() {
    for (devices, maps, auto) in [
        (
            NO_GPU,
            "cpu (auto: no gpu found)",
            "cpu (auto: no gpu found)",
        ),
        (
            MAC,
            "metal (within 1e-6 of cpu) moves=cpu->metal->cpu",
            "cpu (auto: no gpu kernel)",
        ),
        (
            CUDA,
            "cuda (within 1e-6 of cpu) moves=cpu->cuda->cpu",
            "cpu (auto: no gpu kernel)",
        ),
        (
            BOTH,
            "cuda (within 1e-6 of cpu) moves=cpu->cuda->cpu",
            "cpu (auto: no gpu kernel)",
        ),
    ] {
        let nodes = devices_per_node(&pipeline(DeviceRequest::Auto), devices);
        assert_eq!(nodes.len(), 4, "{nodes:?}");
        // Host steps run on the CPU, without a note.
        for host in ["FILTER_CELLS", "SCAN"] {
            assert_eq!(device_of(&nodes, host), "cpu", "{devices}: {host}");
        }
        // Compute steps: the maps on the first GPU (CUDA first), PCA on the CPU out loud.
        assert_eq!(device_of(&nodes, "FUSED_ELEMENTWISE"), maps, "{devices}");
        assert_eq!(device_of(&nodes, "PCA"), auto, "{devices}");
        // The header names the request and the devices found.
        let text = Explain::new(
            &pipeline(DeviceRequest::Auto),
            &physical(&pipeline(DeviceRequest::Auto), devices),
        )
        .to_string();
        assert!(
            text.contains(&format!("\n  device auto (found: {devices})\n")),
            "{text}"
        );
    }
}

#[test]
fn explicit_gpus_are_marked_and_refused_before_the_run() {
    let cases = [
        (
            DeviceRequest::Metal,
            MAC,
            "metal (no kernel: KY-E501 when run)",
            "KY-E501",
        ),
        (
            DeviceRequest::Wgpu,
            MAC,
            "metal (no kernel: KY-E501 when run)",
            "KY-E501",
        ),
        (
            DeviceRequest::Cuda,
            CUDA,
            "cuda (no kernel: KY-E501 when run)",
            "KY-E501",
        ),
        (
            DeviceRequest::Wgpu,
            BOTH,
            "vulkan (no kernel: KY-E501 when run)",
            "KY-E501",
        ),
        (
            DeviceRequest::Cuda,
            MAC,
            "cuda (not found: KY-E504 when run)",
            "KY-E504",
        ),
        (
            DeviceRequest::Metal,
            BOTH,
            "metal (not found: KY-E504 when run)",
            "KY-E504",
        ),
        (
            DeviceRequest::Wgpu,
            NO_GPU,
            "wgpu (not found: KY-E504 when run)",
            "KY-E504",
        ),
    ];
    for (request, devices, shown, code) in cases {
        let plan = pipeline(request);
        let nodes = devices_per_node(&plan, devices);
        assert_eq!(device_of(&nodes, "PCA"), shown, "{request} on {devices}");
        if code == "KY-E501" {
            // The maps have their 9.2 kernels: on the GPU asked for.
            let gpu = shown.split(' ').next().unwrap();
            let maps = format!("{gpu} (within 1e-6 of cpu) moves=cpu->{gpu}->cpu");
            assert_eq!(device_of(&nodes, "FUSED_ELEMENTWISE"), maps);
        } else {
            assert_eq!(device_of(&nodes, "FUSED_ELEMENTWISE"), shown);
        }
        assert_eq!(device_of(&nodes, "SCAN"), "cpu");
        let error = physical(&plan, devices).device_error().unwrap();
        assert_eq!(error.code(), code, "{request} on {devices}");
        let message = error.to_string();
        if code == "KY-E501" {
            // The first step (from the scan up) without a kernel: PCA.
            assert!(message.contains("'pca'"), "{message}");
        } else {
            assert!(
                message.contains(&format!("(found: {devices})")),
                "{message}"
            );
            assert!(
                message.contains(&format!("device '{request}'")),
                "{message}"
            );
        }
    }
}

#[test]
fn cpu_plans_run_and_print_as_before() {
    for devices in [NO_GPU, MAC, BOTH] {
        for plan in [
            pipeline(DeviceRequest::Cpu),
            pipeline(DeviceRequest::Cpu).input().cloned().unwrap(),
        ] {
            let physical = physical(&plan, devices);
            assert!(physical.device_error().is_none());
            assert!(
                physical
                    .executions()
                    .iter()
                    .all(|execution| execution.device() == Some(Device::Cpu))
            );
            let text = Explain::new(&plan, &physical).to_string();
            // No device line for a CPU plan, and no notes.
            assert!(!text.contains("\n  device "), "{text}");
            assert!(!text.contains("(auto"), "{text}");
        }
        // auto falls back but still runs: no error.
        assert!(
            physical(&pipeline(DeviceRequest::Auto), devices)
                .device_error()
                .is_none()
        );
    }
    assert_eq!(scan().device_request(), DeviceRequest::Cpu);
}

#[test]
fn the_last_on_decides_for_the_whole_plan() {
    let plan = scan()
        .on_device(DeviceRequest::Cuda)
        .unwrap()
        .log1p(Some("counts"), "logn")
        .unwrap()
        .on_device(DeviceRequest::Auto)
        .unwrap();
    assert_eq!(plan.device_request(), DeviceRequest::Auto);
    // A step added after `on` keeps the plan's request.
    let later = plan.log1p(Some("logn"), "again").unwrap();
    assert_eq!(later.device_request(), DeviceRequest::Auto);
    let physical = physical(&later, MAC);
    assert_eq!(physical.device_request(), DeviceRequest::Auto);
    assert!(physical.device_error().is_none());
    assert_eq!(physical.estimate().device(), DeviceRequest::Auto);
    // `on` after a sink is refused like any step.
    let sunk = scan().sink(kyto_plan::Sink::Zarr {
        path: "out.kyto".to_owned(),
    });
    let refused = sunk.unwrap().on_device(DeviceRequest::Auto).err().unwrap();
    assert_eq!(refused.code(), "KY-E216");
}

#[test]
fn the_on_step_round_trips_through_the_plan_json() {
    for request in [
        DeviceRequest::Cpu,
        DeviceRequest::Auto,
        DeviceRequest::Cuda,
        DeviceRequest::Wgpu,
        DeviceRequest::Metal,
    ] {
        let plan = pipeline(request);
        let json = plan.to_json();
        assert!(
            json.contains(r#""method": "on""#) || json.contains(r#""method":"on""#),
            "{json}"
        );
        let read = Plan::from_json(&json).unwrap();
        assert_eq!(read.to_json(), json);
        assert_eq!(read.device_request(), request);
        assert!(kyto_plan::validate_plan_json(&json).is_empty());
    }
    let json = pipeline(DeviceRequest::Auto)
        .to_json()
        .replace(r#""device": "auto""#, r#""device": "gpu""#)
        .replace(r#""device":"auto""#, r#""device":"gpu""#);
    let error = Plan::from_json(&json).err().unwrap();
    assert_eq!(error.code(), "KY-E215");
    assert!(error.to_string().contains("'gpu'"), "{error}");
}

#[test]
fn an_on_between_maps_does_not_stop_their_fusion() {
    let fused = |plan: Arc<Plan>| {
        let physical = physical(&plan, MAC);
        let names: Vec<&str> = physical
            .optimized()
            .nodes()
            .iter()
            .map(|node| node.name())
            .collect();
        (names, physical.estimate().passes())
    };
    let middle = scan()
        .normalize_total(Some(1e4), Some("counts"), "norm")
        .unwrap()
        .on_device(DeviceRequest::Auto)
        .unwrap()
        .log1p(Some("norm"), "logn")
        .unwrap();
    let end = scan()
        .normalize_total(Some(1e4), Some("counts"), "norm")
        .unwrap()
        .log1p(Some("norm"), "logn")
        .unwrap()
        .on_device(DeviceRequest::Auto)
        .unwrap();
    assert_eq!(fused(middle.clone()), fused(end));
    assert_eq!(fused(middle).0, ["SCAN", "FUSED_ELEMENTWISE"]);
}

/// A chain alternating devices under `auto` on a GPU: the fused `normalize_total` +
/// `log1p` pass (GPU kernels), a cell filter on the data (no GPU kernel: the CPU), a
/// second `normalize_total` (GPU) reading the filter's output, then `pca` (CPU).
fn alternating(request: DeviceRequest) -> Arc<Plan> {
    let total = CellExpr::qc(Some("logn".to_owned()), QcMetric::TotalCounts).unwrap();
    let expressed = Expr::binary(total, BinaryOp::Gt, Literal::Float(0.0)).unwrap();
    scan()
        .normalize_total(Some(1e4), Some("counts"), "norm")
        .unwrap()
        .log1p(Some("norm"), "logn")
        .unwrap()
        .filter_cells(&expressed)
        .unwrap()
        .normalize_total(Some(1e4), Some("logn"), "renorm")
        .unwrap()
        .pca(Some("renorm"), "X_new", 2, 0, None)
        .unwrap()
        .on_device(request)
        .unwrap()
}

#[test]
fn auto_moves_blocks_between_devices_and_says_so_at_each_step() {
    let gpu = |name: &str| format!("{name} (within 1e-6 of cpu) moves=cpu->{name}->cpu");
    for (devices, on_gpu, on_cpu) in [
        (MAC, gpu("metal"), "cpu (auto: no gpu kernel)"),
        (CUDA, gpu("cuda"), "cpu (auto: no gpu kernel)"),
        (BOTH, gpu("cuda"), "cpu (auto: no gpu kernel)"),
    ] {
        let plan = alternating(DeviceRequest::Auto);
        let nodes = devices_per_node(&plan, devices);
        let expected = [
            ("PCA", on_cpu),
            ("NORMALIZE_TOTAL", on_gpu.as_str()),
            ("FILTER_CELLS", on_cpu),
            ("FUSED_ELEMENTWISE", on_gpu.as_str()),
            ("SCAN", "cpu"),
        ];
        let shown: Vec<(&str, &str)> = nodes
            .iter()
            .map(|(name, device)| (name.as_str(), device.as_str()))
            .collect();
        assert_eq!(shown, expected, "{devices}");
        // Each compute step's input was computed on another device, and it runs.
        let physical = physical(&plan, devices);
        assert!(physical.device_error().is_none(), "{devices}");
        let optimized = physical.optimized().nodes();
        let device_of_node = |index: usize| physical.executions()[index].device().unwrap();
        let mut crossings = 0;
        for (index, node) in optimized.iter().enumerate() {
            let Some(input) = node.input() else { continue };
            if node.name() != "SCAN" && device_of_node(index) != device_of_node(input) {
                crossings += 1;
            }
        }
        // scan -> gpu maps -> cpu filter -> gpu normalize -> cpu pca.
        assert_eq!(crossings, 4, "{devices}");
        // The JSON says the same: moves on the GPU steps, `null` on the CPU ones.
        let json: serde_json::Value =
            serde_json::from_str(&Explain::new(&plan, &physical).to_json()).unwrap();
        let moves: Vec<(&str, Option<&str>)> = json["optimized"]["nodes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|node| (node["name"].as_str().unwrap(), node["moves"].as_str()))
            .collect();
        let gpu_name = on_gpu.split(' ').next().unwrap();
        let moved = format!("cpu->{gpu_name}->cpu");
        assert_eq!(
            moves,
            [
                ("PCA", None),
                ("NORMALIZE_TOTAL", Some(moved.as_str())),
                ("FILTER_CELLS", None),
                ("FUSED_ELEMENTWISE", Some(moved.as_str())),
                ("SCAN", None),
            ],
            "{devices}"
        );
    }
    // No GPU: every compute step on the CPU, said so, and nothing moves.
    let nodes = devices_per_node(&alternating(DeviceRequest::Auto), NO_GPU);
    for (name, device) in &nodes {
        let expected = if name == "SCAN" {
            "cpu"
        } else {
            "cpu (auto: no gpu found)"
        };
        assert_eq!(device, expected, "{name}");
    }
}

#[test]
fn a_named_gpu_refuses_a_chain_with_a_step_it_has_no_kernel_for() {
    for (request, devices, gpu) in [
        (DeviceRequest::Cuda, CUDA, "cuda"),
        (DeviceRequest::Cuda, BOTH, "cuda"),
        (DeviceRequest::Metal, MAC, "metal"),
        (DeviceRequest::Wgpu, BOTH, "vulkan"),
    ] {
        let plan = alternating(request);
        let nodes = devices_per_node(&plan, devices);
        let refused = format!("{gpu} (no kernel: KY-E501 when run)");
        let runs = format!("{gpu} (within 1e-6 of cpu) moves=cpu->{gpu}->cpu");
        assert_eq!(device_of(&nodes, "FILTER_CELLS"), refused);
        assert_eq!(device_of(&nodes, "PCA"), refused);
        assert_eq!(device_of(&nodes, "FUSED_ELEMENTWISE"), runs);
        assert_eq!(device_of(&nodes, "NORMALIZE_TOTAL"), runs);
        // Never a silent fall back to the CPU: the whole run is refused, naming the
        // first step (from the scan up) the GPU has no kernel for.
        let error = physical(&plan, devices).device_error().unwrap();
        assert_eq!(error.code(), "KY-E501", "{request} on {devices}");
        let message = error.to_string();
        assert!(message.contains("'filter_cells'"), "{message}");
        assert!(message.contains(gpu), "{message}");
        assert!(!message.contains("'pca'"), "{message}");
    }
}
