//! kyto-gpu: the GPU backend.
//!
//! Does: CubeCL kernels (CUDA, ROCm, wgpu) and cuSPARSE / cuBLAS through cudarc.
//! Does not: choose devices for plans (see kyto-plan) or run CPU code.
//!
//! `unsafe` is allowed here, each block with a `// SAFETY:` note.
