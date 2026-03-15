# Art Direction System — Design Spec

## Overview

A global art direction system for Wrela Prism that enables runtime-morphable visual styles across the full rendering stack. The system treats style as a gameplay variable — game events (combat flow state, death, biome transitions) drive visual style transitions in real-time.

**Core philosophy:** High-fidelity PBR simulation runs fully, then style transforms apply controlled degradation on top. All defaults produce the current photorealistic output (identity transform). Turning knobs moves away from realism.

**First target style:** Demon Slayer (ufotable) — soft gradient banding, purple-shifted shadows, strong rim lights, curated palette, variable-weight ink outlines. Chosen because it validates the "high fidelity then desample" approach — ufotable renders full 3D CG then applies stylization.

## Architecture: Three-Layer Style Transform Stack

### Layer 1: Style Axes (User/Game-Facing)

~8-12 high-level, intuitive knobs. These are what artists tune and what gameplay code drives. All are `f32` values, all interpolatable.

```rust
pub struct StyleAxes {
    pub softness: f32,             // 0 = sharp/crisp, 1 = soft/dreamy
    pub exaggeration: f32,         // 0 = natural proportions, 1 = bold/stylized
    pub shadow_graphicness: f32,   // 0 = smooth PBR shadows, 1 = hard graphic bands
    pub palette_warmth: f32,       // 0 = cool, 0.5 = neutral, 1 = warm
    pub atmospheric_depth: f32,    // 0 = clear, 1 = heavy atmosphere/fog
    pub surface_detail: f32,       // 0 = flat/simplified, 1 = full texture detail
    pub outline_presence: f32,     // 0 = no outlines, 1 = full ink outlines
    pub color_discipline: f32,     // 0 = raw object colors, 1 = fully palette-mapped
}
```

Default: all zeros = current photorealistic output (except `palette_warmth` which defaults to 0.5 = neutral).

**Example mapping functions** (axes → GPU params):

```rust
fn map_axes_to_uniforms(axes: &StyleAxes) -> ArtDirectionUniforms {
    // shadow_graphicness drives both band count and softness inversely
    shading_bands: axes.shadow_graphicness * 4.0,           // 0→0 (smooth), 1→4 bands
    band_softness: (1.0 - axes.shadow_graphicness) * 0.3,   // 0→0.3 (soft), 1→0.0 (hard)

    // exaggeration drives rim lights and silhouette inflate together
    rim_light_strength: axes.exaggeration * 1.5,
    silhouette_inflate: axes.exaggeration * 0.15,

    // softness drives specular flattening and detail suppression
    specular_flattening: axes.softness * 0.8,
    detail_suppression: axes.softness * 0.6,

    // ... etc — each axis influences multiple GPU params for cohesive effect
}
```

These are hand-authored and tuned by experimentation. The mapping layer is where artistic judgment lives — it's the reason users get 8 intuitive knobs instead of 25 technical ones.

### Layer 2: GPU Parameters (Derived)

~25 raw parameters in `ArtDirectionUniforms`, derived from style axes via hand-authored mapping functions. Uploaded as a separate `wgpu::Buffer` (not extending `LightingUniforms`). Bound to all shaders that need it.

```rust
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ArtDirectionUniforms {
    // Tier 1: Geometry Deformation (read by material resolve — see Pipeline Integration)
    pub silhouette_inflate: f32,
    pub normal_smoothing: f32,
    pub detail_suppression: f32,
    pub _pad0: f32,

    // Tier 2: Shading Transforms (read by material resolve)
    pub shading_bands: f32,              // 0 = smooth PBR, 3-4 = anime banding
    pub band_softness: f32,              // smoothstep width between bands
    pub specular_flattening: f32,        // 0 = full PBR spec, 1 = fully matte
    pub rim_light_strength: f32,         // 0 = PBR default, 1+ = exaggerated

    pub shadow_color_shift: [f32; 4],    // xyz = hue offset for shadowed regions, w = pad
    pub rim_light_color: [f32; 4],       // xyz = tint for rim lighting, w = bark_detail_blend

    pub palette_saturation: f32,         // 1 = no change, >1 = boosted
    pub palette_hue_shift: f32,          // degrees, rotate entire palette
    pub palette_value_contrast: f32,     // light/dark spread
    pub transmission_boost: f32,         // backlit foliage glow multiplier

    // Tier 3: Post-Processing (read by outline/tonemap/bloom)
    pub outline_strength: f32,           // 0 = off, 1 = full
    pub outline_thickness: f32,          // pixels
    pub outline_depth_sensitivity: f32,
    pub outline_normal_sensitivity: f32,

    pub outline_color: [f32; 4],         // xyz = typically near-black, w = pad

    pub color_grade_tint: [f32; 4],      // xyz = overall scene tint, w = color_grade_strength

    pub bloom_tint: [f32; 4],            // xyz = override bloom color, w = bloom_softness

    pub detail_fade_distance: f32,       // where objects flatten to background
    pub _pad3: [f32; 3],
}
```

All vec3 fields are padded to `[f32; 4]` to match WGSL `vec4<f32>` alignment, consistent with the existing `LightingUniforms` convention in `src/scene/mod.rs`.

### Layer 3: Semantic Channels (Per-Object)

Objects expose abstract properties via per-vertex data. Style transforms act on these channels, making styles portable across object types.

Four channels packed into a single `u32` (each as `u8`, normalized to 0.0-1.0 in shader):

| Channel | Description | Trunk | Foliage | Ground |
|---------|-------------|-------|---------|--------|
| `curvature` | Surface curvedness | Ring radius changes | 0 (flat billboards) | Terrain slope |
| `edge_sharpness` | Nearby edge sharpness | High at junctions/flares | High (leaf outlines) | Low (soft ground) |
| `surface_noise` | High-frequency detail level | Bark texture frequency | Moderate (vein detail) | Ripple amplitude |
| `importance` | Visual weight | Size-based | Cluster size | Low (background) |

```rust
pub fn pack_semantic_channels(
    curvature: f32,
    edge_sharpness: f32,
    surface_noise: f32,
    importance: f32,
) -> u32 {
    let c = (curvature.clamp(0.0, 1.0) * 255.0) as u32;
    let e = (edge_sharpness.clamp(0.0, 1.0) * 255.0) as u32;
    let s = (surface_noise.clamp(0.0, 1.0) * 255.0) as u32;
    let i = (importance.clamp(0.0, 1.0) * 255.0) as u32;
    c | (e << 8) | (s << 16) | (i << 24)
}
```

**Vertex format change:** Current `Vertex` struct gains one additional `u32` field for packed semantic channels (+4 bytes per vertex). This is a breaking change that cascades across the codebase — the following must be updated in lock-step:

- `src/scene/mod.rs` — `Vertex` struct definition and `Vertex::layout()`
- `shaders/meshlet_hw_vis_fallback.wgsl` — WGSL `Vertex` struct and field access
- `shaders/material_resolve.wgsl` — WGSL `Vertex` struct and field access
- `src/meshlet/gpu_buffers.rs` — `GpuMeshletBuffers::from_dag` vertex stride
- `src/subjects/tube_mesh.rs` — trunk vertex creation
- `src/subjects/foliage_billboards.rs` — foliage vertex creation
- `src/subjects/ground_slab.rs` — ground vertex creation
- Any tests that construct `Vertex` values directly

**Migration order:** Update the Rust `Vertex` struct first, then fix all subject generators to compute and pack semantic channels, then update both WGSL shader structs, then update GPU buffer code. Run `cargo build` after each step to catch mismatches.

Generators should also pre-compute a smoothed normal and store it alongside the geometric normal. This is needed for `normal_smoothing` in the material resolve shader — a per-vertex smoothed normal is computed as the area-weighted average of face normals sharing each vertex position. This adds another `[f32; 3]` to the vertex (or can be packed as a `u32` using octahedral encoding to save space). The `normal_smoothing` uniform then lerps between geometric normal and smoothed normal in the shader.

## Palette System

A discrete color remapping mechanism that forces all object colors into a curated family. This is the single biggest lever for anime cohesion.

```rust
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct StylePalette {
    // Tonal slots
    pub shadow: [f32; 4],    // darkest tone (w = padding)
    pub dark: [f32; 4],      // dark midtone
    pub mid: [f32; 4],       // primary midtone
    pub light: [f32; 4],     // highlight
    pub accent: [f32; 4],    // pop color (rim lights, backlit leaves)

    // Material-hint slots
    pub bark: [f32; 4],      // wood/trunk color family
    pub foliage: [f32; 4],   // leaf/green color family
    pub earth: [f32; 4],     // ground/soil color family
    pub sky_tint: [f32; 4],  // atmosphere contribution
}
```

**Shader remapping logic:**

1. Compute raw base color from existing material logic
2. Convert to luminance → determines tonal slot (shadow/dark/mid/light)
3. Blend raw hue toward palette's material-hint slot (bark/foliage/earth based on material ID)
4. `color_discipline` style axis controls blend strength: 0.0 = raw colors, 1.0 = fully palette-mapped

Tonal structure (light/dark relationships) is preserved — the palette remaps hue and saturation, not value.

Palette slots are vec3s (padded to vec4) — interpolatable for smooth morphing between palettes.

## Pipeline Integration

### Modified Passes

**HW Raster Pass** — Unchanged in stage 1. See note on geometry deformation below.

**Material Resolve Pass** — The primary integration point. After full PBR lighting computes `raw_light`, apply in order:

0. **Geometry deformation** (shading-only) — after reconstructing world position and normal from the visibility buffer, apply deformations to the interpolated normal before lighting:
   - `silhouette_inflate` × vertex `curvature` → bias normal outward (fattens shading, not silhouette)
   - `normal_smoothing` × vertex `edge_sharpness` → blend normal toward a smoothed average (pre-computed, see Semantic Channels)
   - `detail_suppression` × vertex `surface_noise` → dampen normal perturbation from bark texture
1. **Light banding** — quantize `raw_light` into `shading_bands` steps with `band_softness` smoothstep
2. **Shadow color shift** — where light < threshold, blend toward `shadow_color_shift`
3. **Rim lighting** — fresnel-based edge glow scaled by `rim_light_strength`, tinted by `rim_light_color`
4. **Palette remap** — base_color through palette LUT, blended by `color_discipline`
5. **Specular flatten** — lerp specular toward zero by `specular_flattening`
6. **Transmission boost** — scale backlit foliage glow by `transmission_boost`

Order matters: PBR first (high fidelity), then style transforms (controlled degradation).

**Why deformation is in material resolve, not HW raster:** The visibility buffer pipeline has a fundamental split — HW raster writes vis IDs using projected vertices, then material resolve reconstructs geometry from the *original* vertex data. If we deformed vertices in HW raster, the silhouettes would change but material resolve would reconstruct from un-deformed data, producing incorrect barycentrics. For stage 1, we apply deformation to the *interpolated attributes* in material resolve. This changes shading (normals, lighting response) but not silhouettes. For Demon Slayer's needs (bolder shading, not dramatically different silhouettes), this is sufficient. Silhouette deformation (requiring deformation in both passes or a pre-deformation compute pass) is deferred to a future stage.

**Bloom Pass** — Read `bloom_tint` and `bloom_softness` from art direction uniforms.

**Tonemap Pass** — Apply `color_grade_tint` × `color_grade_strength` post-tonemap.

### New Pass

**Outline Pass** (inserted between Sun Shafts and Bloom):
- Screen-space edge detection via Sobel/Roberts on depth buffer + normal buffer (both already available)
- `outline_depth_sensitivity` — edge weight from depth discontinuities (silhouettes)
- `outline_normal_sensitivity` — edge weight from normal discontinuities (surface detail)
- `outline_thickness` — kernel size / dilation
- `outline_strength` — final composite blend (0 = invisible, 1 = full ink)
- `outline_color` — typically near-black for anime
- When `outline_strength` = 0, early-out (no-op)
- **Render target:** Composites directly onto the scene color target using `LoadOp::Load` with alpha blending, consistent with how sun shafts already composite.

### New Module

**`src/art_direction.rs`** — Contains:
- `StyleAxes` struct and defaults
- `ArtDirectionUniforms` struct
- `StylePalette` struct
- Mapping functions (axes → GPU params)
- `ArtDirection` runtime struct with transition engine
- `StyleSnapshot` for saved styles
- GPU buffer creation and upload
- Named style constants (e.g., `DEMON_SLAYER`)

**Bind group strategy:** Art direction uniforms and palette use **bind group 3** (currently unused). This avoids rebuilding the existing material pass pipeline layout (bind groups 0-2). The art direction bind group layout contains two bindings:
- Binding 0: `ArtDirectionUniforms` (uniform buffer)
- Binding 1: `StylePalette` (uniform buffer)

## Runtime API

### Two Consumers

**Gameplay (Programmatic):**

```rust
impl ArtDirection {
    /// Transition to a target style over a duration with an easing curve.
    /// E.g., player enters flow state → watercolor style over 2 seconds.
    fn transition_to(&mut self, target: &StyleSnapshot, duration: Duration, curve: EaseCurve);

    /// Weighted blend from multiple simultaneous style sources (owned for persistence across frames).
    /// E.g., biome drives palette while combat intensity drives shadow_graphicness.
    fn set_blend(&mut self, sources: Vec<(Arc<StyleSnapshot>, f32)>);

    /// Bind a single axis to a game variable.
    /// E.g., health_normalized drives desaturation.
    fn drive_axis(&mut self, axis: Axis, value: f32);

    /// Snapshot current state for saving/restoring.
    fn snapshot(&self) -> StyleSnapshot;

    /// Called each frame — advances transitions, recomputes uniforms, uploads to GPU.
    fn update(&mut self, dt: Duration, queue: &wgpu::Queue);
}

pub enum EaseCurve {
    Linear,
    EaseIn,
    EaseOut,
    EaseInOut,
    // Custom curves can be added as named variants as needed.
    // Avoids fn pointer which prevents Clone/Debug/serialization.
}
```

**Per-frame update cycle:**
1. Evaluate all active transitions (advance `t`, apply ease curve)
2. Collect all blend sources with their weights
3. Weighted-average all `StyleAxes` values
4. Run mapping functions → `ArtDirectionUniforms`
5. Weighted-average all `StylePalette` slots
6. Upload both to GPU

**Harness (Authoring/Tuning):**
- `F5` — cycle through saved style snapshots (F1-F4 are reserved for existing debug overlays)
- `F6` — toggle style inspector overlay showing current axis values
- `[` / `]` — blend slider between current and next snapshot
- `0` — snap to photorealistic (identity)
- CLI: `--style demon_slayer`, `--style-blend 0.5`, `--softness 0.7`

### Per-Object Overrides

**Deferred to stage 1b.** The Demon Slayer style applies uniformly to all objects — per-object overrides are not needed to validate the system. Deferring this removes significant complexity from the initial implementation (prototype-indexed buffer, shader branching, GPU representation of optional values).

When implemented, the GPU representation will use a sentinel-value approach (`f32::NAN` = use global):

```rust
// CPU-side API (ergonomic)
pub struct ArtDirectionOverrides {
    pub silhouette_inflate: Option<f32>,
    pub shading_bands: Option<f32>,
    pub rim_light_strength: Option<f32>,
    // ... etc
}

// GPU-side representation (Pod/Zeroable, uploaded per prototype)
#[repr(C)]
pub struct GpuArtDirectionOverrides {
    pub silhouette_inflate: f32,  // NAN = use global
    pub shading_bands: f32,       // NAN = use global
    pub rim_light_strength: f32,  // NAN = use global
    // ... etc
}
// Shader: let value = select(global.shading_bands, override.shading_bands, !isNan(override.shading_bands));
```

Stored per prototype in a storage buffer indexed by prototype ID. The material resolve shader reads global, then conditionally overrides per-field.

### Named Style Snapshots

Stored as Rust constants to start (can move to data files later):

```rust
pub const DEMON_SLAYER: StyleSnapshot = StyleSnapshot {
    axes: StyleAxes {
        softness: 0.3,
        exaggeration: 0.7,
        shadow_graphicness: 0.8,
        palette_warmth: 0.6,
        atmospheric_depth: 0.5,
        surface_detail: 0.4,
        outline_presence: 0.7,
        color_discipline: 0.8,
    },
    palette: StylePalette { /* curated demon slayer colors */ },
};
```

## Scope

### In Stage 1

- Three-layer architecture (axes → params → semantic channels)
- ArtDirectionUniforms + StylePalette as separate GPU buffers in bind group 3
- Semantic channels packed into vertex data (+ smoothed normals)
- Palette system with shader remapping
- Material resolve shading-only geometry deformation (normal modification, no silhouette change)
- Material resolve post-PBR transforms (banding, shadow shift, rim, palette, spec flatten)
- Outline pass (screen-space edge detection, compositing onto scene color)
- Bloom tint + tonemap color grading
- Transition API with ease curves and weighted blending
- Harness controls (F5/F6/brackets/0/CLI)
- One complete style: Demon Slayer

### Stage 1b (Near-Term Follow-Up)

- Per-object overrides via prototype-indexed buffer (sentinel-value GPU representation)
- Silhouette deformation (requires deformation in both HW raster and material resolve, or pre-deformation compute pass)

### Deferred (Future Stages)

- Spatial style fields (style varying across the world)
- Style inheritance and sharing ecosystem
- Composition-layer influence (density, spacing, rhythm)
- Latent style manifold / learned mappings
- AI-assisted style generation
- GUI slider panel
- SW raster deformation path
- Additional style targets beyond Demon Slayer

## Testing Strategy

- **Unit tests:** Mapping functions (axes → params) produce expected values for known inputs. Semantic channel packing/unpacking roundtrips correctly.
- **Snapshot tests:** Capture frame at identity (all zeros) and verify pixel-identical to current renderer output — guarantees no regression.
- **Style capture tests:** `--style demon_slayer --capture test.png` produces deterministic output for a given seed.
- **Interpolation tests:** Verify that lerping between two StyleSnapshots produces valid intermediate uniform values (no NaN, no out-of-range).
