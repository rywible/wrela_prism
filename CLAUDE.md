# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Wrela Prism is a real-time 3D renderer in Rust implementing a Nanite-style visibility buffer pipeline. It renders procedurally-generated redwood trees with meshlet LOD, atmospheric effects, and post-processing, running on WebGPU via wgpu.

## Build & Run Commands

```bash
cargo build                  # Debug build (opt-level 1, deps at opt-level 3)
cargo build --release        # Release build (fat LTO, single codegen unit, O3)
cargo run --release          # Run interactive viewer
cargo test                   # Run all tests
cargo test <test_name>       # Run a single test by name
cargo clippy --all-targets   # Lint
cargo fmt --check            # Check formatting
```

**CLI arguments:**
```bash
--preset {hero|low_angle|silhouette|neutral_debug}  # Camera preset
--seed <u64>                                         # Tree generation seed
--capture <path.png>                                 # Capture frame to PNG and exit
--capture-size WIDTHxHEIGHT                          # Output resolution
--camera-position x,y,z                              # Camera world position
--camera-yaw <degrees>                               # Horizontal rotation
--camera-pitch <degrees>                             # Vertical rotation
```

**Logging:** `RUST_LOG=wrela_prism=debug cargo run --release` (default filter: `wrela_prism=info,wgpu=warn`)

## Architecture

### Rendering Pipeline (`src/pipeline/`)

A multi-pass visibility buffer pipeline orchestrated by `VisbufPipeline`:

1. **Cull** — Compute shader instance culling via bounding sphere projection
2. **HW Raster** — Hardware rasterization to visibility buffer (R32Uint: meshlet_id << 8 | tri_id)
3. **SW Raster** — Software rasterization fallback
4. **Shadow** — Directional shadow map (depth-only from light view)
5. **HZB** — Hierarchical Z-buffer construction
6. **Material** — Visibility buffer resolve: unpack IDs → fetch attributes → shade
7. **SSAO** — Screen-space ambient occlusion
8. **Sky** — Atmospheric scattering (Rayleigh + Mie) + sun disk
9. **Sun Shafts** — God rays via occlusion sampling
10. **Bloom** — Threshold → blur → composite
11. **Tonemap** — Final tone mapping + gamma correction

### Scene Flow

`SourceScene` → `SceneCompiler::compile()` → `RuntimeScene` → GPU upload → render

- **SourceScene** (`src/source_scene.rs`): Declarative scene definition with nodes (geometry + transform + material)
- **SceneCompiler** (`src/compiler/`): Realizes geometry, assigns spatial chunks, builds meshlet DAGs
- **RuntimeScene** (`src/runtime_scene.rs`): GPU-ready scene with prototypes, instances, chunks, and spatial index

### Meshlet LOD DAG (`src/meshlet/`)

Hierarchical LOD via directed acyclic graph of meshlets:
- `simplify.rs`: Builds DAG levels — partition mesh into meshlets, group & simplify, repeat
- `partition.rs`: Meshlet partitioning with max triangle constraints
- `gpu_buffers.rs`: GPU memory layout for meshlet data
- Uses `meshopt` crate for mesh simplification

### Procedural Geometry (`src/subjects/`)

- `redwood_growth.rs`: Deterministic tree generation (Splitmix64 PRNG) — trunk capsules, recursive branches, foliage anchors, root flares
- `foliage_billboards.rs`: Billboard quads at foliage anchor points with alpha masking
- `tube_mesh.rs`: Branch/trunk capsule mesh generation
- `ground_slab.rs`: Ground plane geometry

### Key Types

- **`GpuContext`** (`src/gpu/`): wgpu device/queue/surface wrapper
- **`CameraState`** (`src/camera.rs`): Position, yaw/pitch, view-projection, frustum; WASD + mouse controls
- **`SceneSettings`** / **`LightingUniforms`** (`src/scene/mod.rs`): Sun, fog, sky, atmosphere, wind parameters → GPU uniform buffer
- **`Vertex`** (`src/scene/mod.rs`): Position, normal, material ID, UV, AO
- **`Soundstage`** (`src/soundstage/`): Camera presets with lighting configs (hero, low_angle, silhouette, neutral_debug)

### Shaders

All shaders are WGSL files in `shaders/`. They are included at compile time via `include_str!` / `include_wgsl!`. Changes to `.wgsl` files require recompilation.

## Key Dependencies

| Crate | Purpose |
|-------|---------|
| `wgpu` 28 | WebGPU graphics API |
| `winit` 0.30 | Window & event loop |
| `glam` 0.29 | SIMD math (vectors, matrices) |
| `bytemuck` 1 | Zero-copy GPU data casting |
| `meshopt` 0.6 | Mesh simplification for LOD |
| `image` 0.25 | PNG frame capture |

## Interactive Controls

- **WASD** — Move, **QE** — Vertical, **Shift** — Sprint
- **Mouse** — Look (after left-click to capture cursor)
- **F1-F4** — Debug overlays (structure-only, canopy-only, wind magnitude, LOD heatmap)
- **ESC** — Release cursor
