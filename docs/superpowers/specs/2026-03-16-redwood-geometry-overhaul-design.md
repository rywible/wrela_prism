# Redwood Geometry Overhaul — Design Spec

**Date**: 2026-03-16
**Status**: Draft
**Goal**: Make procedural redwood trees structurally read as photorealistic coast redwoods — columnar trunk, proper branch hierarchy, opaque geometry foliage, conical crown silhouette. Art direction system handles downstream stylization.

## Context

The current redwood tree has several fidelity problems:

1. **Trunk is a cone** — real redwoods are columnar (stay thick most of their height)
2. **Branch structure is sparse** — 14 primary branches, max depth 3, reads as a hat rack
3. **Foliage is billboard cards with alpha masks** — visible as discrete rectangles, expensive for the visibility buffer pipeline (alpha evaluation per-pixel, HZB poisoning from discarded fragments, dedicated alpha fixup compute pass)
4. **Crown is flat-topped** — should be a narrow, irregular conical spire

The rendering pipeline (visibility buffer, meshlet LOD, frustum culling, HZB occlusion) is well-suited for high triangle counts with fully opaque materials. Following Nanite's production approach: replace alpha-masked billboards with dense opaque geometry and let the meshlet DAG handle LOD.

## Approach

Geometry-first overhaul. Replace billboard foliage with opaque needle spray geometry. Retune branch architecture parameters and extend zone logic for redwood morphology. Fix trunk proportions. Add area preservation to the meshlet simplifier. Remove the alpha mask pipeline entirely.

---

## 1. Trunk Geometry

### Problem
Current trunk tapers from `base_radius: 3.1` to `tip_radius: 0.36` over 42 units using `t.powf(1.72)` — a smooth cone. Real coast redwoods are columnar.

### Design

**Columnar taper profile**: Replace the single power-curve taper with a two-phase profile:
- **Lower `columnar_fraction` (70%)**: Very gradual taper using `radius = base_radius * lerp(1.0, 0.85, smoothstep(0.0, columnar_fraction, t))`. The "telephone pole" look that defines redwoods.
- **Upper 30%**: Power-curve taper from the columnar-end radius down to `tip_radius`, using `(t_upper).powf(2.0)` where `t_upper` is normalized 0→1 within the upper section.

**Base fluting**: Remove `grow_buttresses` (which creates discrete `BranchKind::Buttress` skeleton nodes and separate tube meshes). Replace with radial fluting as a ring-level perturbation in `append_tube_along_path` — a sinusoidal radius modulation at low `height_ratio` values that creates vertical ridges/channels in the lower 10-15% of the trunk. This integrates fluting into the trunk mesh itself rather than separate geometry. Note: this is an **addition** to the existing `bark_profile` function, which provides fine-detail bark texture at all heights. Fluting is a larger-scale modulation that fades out above the base zone, while bark profile continues to operate on top of it.

**Increased dimensions**: Bump tree height from 42 to ~65 units with proportionally wider base (~4.5 units radius) to improve the aspect ratio and sell old-growth scale.

### Parameters Changed
- `trunk_height`: 42 → 65 (tunable)
- `base_radius`: 3.1 → 4.5
- `tip_radius`: 0.36 → 0.25 (narrower leader)
- New: `columnar_fraction: 0.70` — fraction of height with minimal taper
- New: `flute_count: 6-10` — number of radial ridges at base
- New: `flute_depth: 0.15-0.25` — how deep the channels cut (fraction of radius)
- Remove: `root_flare` parameter (replaced by fluting)

---

## 2. Branch Architecture

### Problem
14 primary branches, all clustered in top 30%, max depth 3. Too sparse, wrong placement.

### What Already Exists
The codebase already has the right zone infrastructure: `PrimaryStyle::{LowerScaffold, UpperSecondary, FillerLive, DeadStub}` with corresponding `BranchKind` variants, lobe assignments, droop/lift per-segment parameters, and recursive sub-branching via `grow_live_sub_branches`. The primary changes are **parameter retuning** and **extending the zone ranges**, not a structural rewrite of the branching logic.

### Design

Retune the existing zone system with these target distributions and ranges. Note: current actual counts are constrained by `branch_count: 14` even though `desired_lower` is 6-8 and `desired_upper` is 14-18 in the code. With `branch_count: 65`, these desired counts will actually be achievable.

1. **Dead zone (0-60% height)**: Expand dead stub range from current `0.64-0.82` down to `0.15-0.60`. Increase `desired_stub` from 3-4 to 15. Stubs should be short broken-off nubs angled slightly downward.

2. **Lower scaffold zone (60-75%)**: Increase `desired_lower` from 6-8 to 12. These carry the bulk of foliage. Retune existing `LowerScaffold` parameters:
   - Increase `seg_count` from 6 to 8-10 for smoother droop-then-upturn curves
   - Increase `base_radius_factor` from 0.10 to 0.15-0.20 for thicker limbs
   - Increase branch length to reach 8-12 units outward (longest in the tree)

3. **Mid-crown zone (75-90%)**: Shift `UpperSecondary` range down from current `0.84-0.97` to `0.75-0.90`. Increase `desired_upper` from 14-18 to 25. Progressively shorter as height increases (driven by crown envelope function).

4. **Upper crown / leader zone (90-100%)**: 12 small branches at steep upward angles. Crown narrows to irregular spire. A few asymmetric branches near the very top for a scraggly look.

**Branching depth**: Increase `max_branch_depth` from 3 to 5. More recursive sub-branching creates the fine twig structure that carries foliage.

**Total**: ~65 primary attachment points. Recursive sub-branching produces several hundred total branch segments.

### Parameters Changed
- `branch_count`: 14 → 65 (drives total primary attachments; existing `build_primary_branch_specs` distributes across zones)
- `max_branch_depth`: 3 → 5
- `crown_start_frac`: 0.70 → 0.60 (bottom of live crown; dead stubs placed below this)
- Retune per-style parameters within `grow_primary_branch` (seg counts, radius factors, lengths)
- Dead stub range: `0.64-0.82` → `0.15-0.60` (below `crown_start_frac`)

---

## 3. Foliage Geometry

### Problem
Billboard cards with alpha masks. Visible as individual rectangles. Expensive in visibility buffer pipeline (per-pixel alpha eval, HZB poisoning, dedicated alpha fixup pass). Fighting the architecture.

### Design

**Replace billboards with opaque needle spray geometry.**

Each "leaf unit" is a spray of 4-8 flat needle-shaped polygons (thin elongated quads, no alpha) arranged in a fan/radial pattern around a twig axis:
- Fully opaque, `MATERIAL_FOLIAGE`
- Each spray: ~16-32 triangles
- No alpha testing anywhere in the pipeline

**Placement**: Sprays attach to the last 2-3 segments of every live branch at depth 3+. Oriented along branch direction with random rotation around the axis. Dense enough that adjacent sprays overlap and merge into a continuous mass.

**Density**: ~30-60 sprays per terminal branch cluster. Thousands of sprays total per tree. At 16-32 triangles each: roughly 200k-500k foliage triangles per tree. The meshlet DAG collapses these aggressively at distance.

**Orientation variation**: Each spray gets random tilt (±30°) and twist to break uniformity. Slight upward bias to catch light, matching how real redwood foliage presents flat sprays to the sun.

**Canopy reads as solid mass** through sheer density and overlap — no individual spray is meant to look convincing alone. The aggregate is what matters.

**Minimum-viable foliage shading**: The current `MATERIAL_FOLIAGE` shader path is tuned for billboard cards (procedural normal modulation, subsurface approximation for thin translucent cards, screen-facing normals). With opaque needle sprays, normals come from actual geometry. As part of this overhaul:
- Remove alpha test / discard logic from `MATERIAL_FOLIAGE` path
- Keep the existing diffuse albedo and fog attenuation
- Let geometry normals drive lighting naturally (no procedural normal override)
- Subsurface scattering approximation can be simplified to a basic wrap-lighting term
- Full shading redesign deferred to a follow-up pass

**Build time note**: 200k-500k foliage triangles per tree through the meshlet DAG builder (`compile_chunk`) will increase build time. The existing DAG builder processes all geometry in a single pass per chunk. This should be monitored during implementation — if build times exceed ~2s per tree, consider splitting foliage and wood into separate chunks that get separate DAG builds.

### New File
- `needle_sprays.rs` — replaces `foliage_billboards.rs`

---

## 4. Crown Silhouette & Envelope

### Problem
Current crown is a flat-topped blob. Real redwoods have a narrow, conical, irregular spire.

### Design

**Conical envelope function**: Branch length is driven by a crown envelope. Longest branches at the bottom of the live crown (`crown_start_frac = 0.60`), progressively shorter toward the top. Envelope shape is a narrow cone, producing the classic redwood silhouette but with irregularity.

**Lobe structure with gaps**: The existing lobe system (6-7 azimuthal lobes with gap angles) is a good foundation. Make the effect more pronounced — heavy foliage clusters separated by thinner windows where sky is visible through the crown. Real redwood crowns look ragged, not perfectly filled.

**Irregular apex**: The leader (topmost trunk extension) pokes above the last branches. A few small asymmetric branches near the very top create a scraggly spire. Optional: multiple competing leaders (fork at top).

**Crown depth**: Inner branches with sparser foliage create parallax and shadow. Not just a shell of foliage on the outside. The SSAO pass picks this up naturally.

**Proportions** (at `trunk_height: 65`):
- Live crown: top 40% of tree height (starts at 60% = 39 units up)
- Below: bare trunk with dead stubs
- Crown width at widest (bottom of live crown): ~16-22 units diameter (~25-35% of height)
- Narrow and tall, not wide and round

---

## 5. Meshlet LOD & Area Preservation

### Problem
When the meshlet DAG simplifies thin foliage geometry at distance, needle sprays vanish — creating "forests of sticks." This is the exact problem Nanite's Preserve Area feature solves.

### Design

**Area preservation as a post-process on `simplify_mesh` output** (`src/meshlet/simplify.rs`):

After `meshopt::simplify` produces the simplified vertex/index buffer for a meshlet group:

1. Compute total triangle surface area of the input geometry (sum of `0.5 * |cross(e1, e2)|` per triangle).
2. Compute total triangle surface area of the simplified output.
3. If `output_area / input_area < 0.85`, compute a scale factor `sqrt(input_area / output_area)`.
4. For each vertex in the simplified output, displace it outward from the group centroid by `(position - centroid) * scale_factor`. This dilates the surviving geometry to maintain perceived coverage.
5. Clamp the scale factor to a maximum of 2.0 to prevent extreme distortion at very aggressive simplification levels.

This is applied in **world space** as a post-process on the `simplify_mesh` output, not as a modification to meshopt's simplification target. It works because at the LOD distances where this activates, the geometric distortion is below perceptual threshold.

**Foliage-aware meshlet grouping**: Before calling `meshopt::clusterize::partition_clusters_with_positions`, offset foliage vertex positions by a small artificial separation (e.g., add `material_id * 0.001` to Y) so that spatial partitioning naturally keeps foliage and branch geometry in separate groups. After partitioning, restore original positions. This ensures foliage clusters simplify together into broader leaf masses rather than getting merged with wood geometry and disappearing.

**No special LOD path**: The existing DAG builder structure is unchanged. The cull pass, HW/SW raster, and material resolve all work as-is.

---

## 6. Alpha Pipeline Removal

Removing billboard foliage eliminates the need for the entire alpha mask pipeline. This touches more files than just the foliage generators:

### Rust Files
| File | Change |
|------|--------|
| `src/subjects/foliage_billboards.rs` | **Removed** — replaced by `needle_sprays.rs` |
| `src/subjects/foliage_atlas.rs` | **Removed** — no more atlas textures |
| `src/subjects/alpha_mask.rs` | **Removed** — no more alpha mask generation |
| `src/subjects/mod.rs` | Remove `pub mod foliage_billboards`, `pub mod foliage_atlas`, `pub mod alpha_mask`; add `pub mod needle_sprays` |
| `src/source_scene.rs` | Remove `alpha_tested: bool` field from `SourceNode` and its default in `add` method |
| `src/pipeline/visbuf_pipeline.rs` | Remove alpha fixup pass setup, remove `alpha_mask` references |
| `src/pipeline/material_pass.rs` | Remove `alpha_mask` texture binding and references |
| `src/material/bark_bake.rs` | Remove `alpha_mask` references |
| `src/runtime_scene.rs` | Remove `alpha_tested` field from surface/material structs |
| `src/compiler/mod.rs` | Remove `alpha_tested` propagation through compilation |

### Shader Files
| File | Change |
|------|--------|
| `shaders/alpha_fixup.wgsl` | **Removed** — entire compute pass no longer needed |
| `shaders/material_resolve.wgsl` | Remove `foliage_alpha_mask` function, remove alpha mask texture bindings (group 2, binding 3-4), remove discard logic in `MATERIAL_FOLIAGE` path |
| `shaders/shadow_pass.wgsl` | Remove `foliage_alpha_mask_shadow` function and foliage discard logic |

### Bind Group Layout
Removing the alpha mask texture bindings from group 2 means updating the bind group layout in the pipeline setup code. All bind group indices above the removed bindings may need renumbering.

---

## 7. Summary of Geometry/Growth Changes

| File | Change |
|------|--------|
| `redwood_growth.rs` | Retune trunk taper (columnar profile), retune branch zone parameters (counts, ranges, depths), remove `grow_buttresses`, update crown envelope |
| `tube_mesh.rs` | Add fluting perturbation to trunk ring extrusion at low `height_ratio`, handle higher branch counts |
| `needle_sprays.rs` | **New** — opaque needle spray geometry generator |
| `simplify.rs` | Add area preservation post-process, foliage-aware partition trick |
| `RedwoodParams` | New/changed fields for columnar taper, fluting, branch zone counts, foliage density |
| `redwood.rs` | Update `Default` impl and public API wrappers to match new params |

### Not Changed
- Rendering pipeline structure (cull, HW raster, SW raster, HZB, SSAO, sky, bloom, tonemap)
- Meshlet DAG builder structure (only simplification post-process added)
- Ground slab geometry

### Follow-Up Work (Not In This Spec)
- **Soundstage preset retuning**: Tree dimensions change from 42→65 height and 3.1→4.5 base radius. Camera positions in hero/low_angle/silhouette/neutral_debug presets will need adjustment for good framing. Do this after geometry lands.
- **Wind animation**: Current wind vertex displacement is tuned for billboard cards. Opaque needle sprays need different wind response (vertex displacement on dense meshes vs. billboard orientation). Accept static foliage as a temporary regression; design wind in a follow-up.
- **Full foliage shading redesign**: The minimum-viable shading in this spec (geometry normals, basic wrap lighting) is functional but not final. A dedicated shading pass for subsurface scattering, translucency, and color variation comes later.
- **Forest instancing**: This spec covers single-tree fidelity. Forest-scale variation, placement, and LOD is a separate future spec.
