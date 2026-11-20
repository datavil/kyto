//! kyto-gpu: the GPU backend.
//!
//! Does: name the devices kyto runs on ([`Device`]), find them on this machine
//! ([`probe`], [`Devices`]) and say which operations each one has a kernel for
//! ([`Device::has_kernel`]), through one [`Backend`] interface with a CPU, a CUDA and a
//! wgpu (Metal, Vulkan, DX12) implementation (task 9.1). CubeCL kernels (9.2) and
//! cuSPARSE / cuBLAS through cudarc (9.3) arrive later; until then no GPU has a kernel.
//! Does not: choose devices for plans (the `device="auto"` rules live in kyto-plan's
//! `device_placement.rs`) or run CPU code.
//!
//! **Features.** The GPU libraries are opt-in, so the default build and wheel link none
//! of them and the probe finds only the CPU: `wgpu` finds a Metal, Vulkan or DX12
//! adapter, `cuda` an NVIDIA GPU through the driver library loaded at run time (no CUDA
//! toolkit needed to build).
//!
//! `unsafe` is allowed here, each block with a `// SAFETY:` note.

mod backend;
mod device;
mod probe;

pub use backend::{BACKENDS, Backend, Cpu, Cuda, Runtime, Wgpu};
pub use device::{Device, Devices, WgpuBackend};
pub use probe::probe;
