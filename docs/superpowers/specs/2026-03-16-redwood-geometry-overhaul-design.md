# Redwood Geometry Overhaul — Design Spec

**Date**: 2026-03-16
**Status**: Draft
**Goal**: Make procedural redwood trees structurally read as photorealistic coast redwoods — columnar trunk, proper branch hierarchy, opaque geometry foliage, conical crown silhouette. Art direction system handles downstream stylization.

## Context

The current redwood tree has several fidelity problems:

1. **Trunk is a cone** — real redwoods are columnar (stay thick most of their height)
2. **Branch structure is sparse** — 14 primary branches, max depth 3, reads as a hat rack
3. **Foliage is billboard cards with alpha masks** — visible as discrete rectangles, expensive for the visibility buffer pipeline (alpha evaluation per-pixel, HZB poisoning)
4. **Crown is flat-topped** — should be a narrow, irregular conical spire

The rendering pipeline (visibility buffer, meshlet LOD, frustum culling, HZB occlusion) is well-suited for high triangle counts with fully opaque materials. Following Nanite's production approach: replace alpha-masked billboards with dense opaque geometry and let the meshlet DAG handle LOD.

## Approach

Geometry-first overhaul. Replace billboard foliage with opaque needle spray geometry. Rebuild branch architecture to match redwood morphology. Fix trunk proportions. Add area preservation to the meshlet simplifier.

No changes to the rendering pipeline (cull, raster, HZB, shadow, material resolve, post-processing). This is a content/geometry change plus a simplifier improvement.

---

## 1. Trunk Geometry

### Problem
Current trunk tapers from `base_radius: 3.1` to `tip_radius: 0.36` over 42 units using `t.powf(1.72)` — a smooth cone. Real coast redwoods are columnar.

### Design

**Columnar taper profile**: Replace the single power-curve taper with a two-phase profile:
- **Lower 70%**: Very gradual taper. Radius stays within 80-90% of base radius. This gives the "telephone pole" look that defines redwoods vs. other conifers.
- **Upper 30%**: Accelerating taper toward a narrow leader/spire tip.

**Base fluting**: Rework buttresses as radial fluting integrated into the trunk ring geometry at the base. Irregular vertical ridges/channels carved into the lower 10-15% of the trunk that flow into the ground. Not discrete separate capsule objects — part of the trunk mesh itself.

**Increased dimensions**: Bump tree height from 42 to ~60-70 units with proportionally wider base (~4-5 units radius) to improve the aspect ratio and sell old-growth scale.

### Parameters Changed
- `trunk_height`: 42 → 65 (tunable)
- `base_radius`: 3.1 → 4.5
- `tip_radius`: 0.36 → 0.25 (narrower leader)
- New: `columnar_fraction: 0.70` — fraction of height with minimal taper
- New: `flute_count: 6-10` — number of radial ridges at base
- New: `flute_depth: 0.15-0.25` — how deep the channels cut (fraction of radius)

---

## 2. Branch Architecture

### Problem
14 primary branches, all clustered in top 30%, max depth 3. Too sparse, wrong placement.

### Design

**Four vertical zones on the trunk:**

1. **Dead zone (0-55% height)**: Bare trunk. Only dead stubs (broken-off branch nubs). 10-20 stubs, short, angled slightly downward. This is the self-pruning zone — redwoods shed lower branches as the canopy rises.

2. **Lower scaffold zone (55-75%)**: 10-15 massive horizontal scaffold limbs. These are the oldest surviving branches:
   - Thick: radius ~15-20% of trunk radius at attachment point
   - Long: reach 8-12 units outward
   - Droop-then-upturn profile: 8-10 segment paths that sag from gravity then curve upward at tips
   - Carry the bulk of foliage mass
   - Longest branches in the tree (widest point of crown envelope)

3. **Mid-crown zone (75-90%)**: 20-30 secondary branches. Shorter than lower scaffolds, angled slightly upward, medium thickness. Progressively shorter as height increases (conical envelope).

4. **Upper crown / leader zone (90-100%)**: 10-15 small branches at steep upward angles, short. Crown narrows to irregular spire. A few asymmetric branches near the very top for a scraggly look. Optional competing leader (fork).

**Branching depth**: Increase `max_branch_depth` from 3 to 4-5. Each scaffold limb → secondary → tertiary → foliage-bearing twigs. This recursive hierarchy is what makes a tree read as a tree.

**Total branch count**: ~60-80 primary attachment points. Recursive sub-branching produces several hundred total branch segments.

### Parameters Changed
- `branch_count`: 14 → 65 (tunable)
- `max_branch_depth`: 3 → 5
- `crown_start_frac`: 0.70 → 0.55 (for dead stubs); live crown starts ~0.55-0.60
- New: `dead_stub_count: 15`
- New: `scaffold_count: 12`
- New: `secondary_count: 25`
- New: `upper_count: 12`

---

## 3. Foliage Geometry

### Problem
Billboard cards with alpha masks. Visible as individual rectangles. Expensive in visibility buffer pipeline (per-pixel alpha eval, HZB poisoning). Fighting the architecture.

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

### Files Changed
- `foliage_billboards.rs` — **replaced entirely** with new needle spray generator (e.g., `needle_sprays.rs`)
- `foliage_atlas.rs` — **removed** (no more alpha mask textures)
- `alpha_mask.rs` — **removed** (no more alpha testing)
- Material shader — remove `MATERIAL_FOLIAGE` alpha test path; foliage becomes fully opaque shading

---

## 4. Crown Silhouette & Envelope

### Problem
Current crown is a flat-topped blob. Real redwoods have a narrow, conical, irregular spire.

### Design

**Conical envelope function**: Branch length is driven by a crown envelope. Longest branches at the bottom of the live crown, progressively shorter toward the top. Envelope shape is a narrow cone, producing the classic redwood silhouette but with irregularity.

**Lobe structure with gaps**: The existing lobe system (6-7 azimuthal lobes with gap angles) is a good foundation. Make the effect more pronounced — heavy foliage clusters separated by thinner windows where sky is visible through the crown. Real redwood crowns look ragged, not perfectly filled.

**Irregular apex**: The leader (topmost trunk extension) pokes above the last branches. A few small asymmetric branches near the very top create a scraggly spire. Optional: multiple competing leaders (fork at top).

**Crown depth**: Inner branches with sparser foliage create parallax and shadow. Not just a shell of foliage on the outside. The SSAO pass picks this up naturally.

**Proportions**:
- Live crown: top 35-40% of total tree height
- Below: bare trunk with dead stubs
- Crown width at widest (bottom of live crown): ~25-35% of tree height
- Narrow and tall, not wide and round

---

## 5. Meshlet LOD & Area Preservation

### Problem
When the meshlet DAG simplifies thin foliage geometry at distance, needle sprays vanish — creating "forests of sticks." This is the exact problem Nanite's Preserve Area feature solves.

### Design

**Area preservation in the simplifier** (`src/meshlet/simplify.rs`):

After each simplification pass, compare input vs. output surface area per meshlet group. If the area ratio drops below a threshold (~0.85), scale surviving vertex positions outward from the cluster centroid to compensate. This keeps perceived canopy density stable across LOD levels.

**Foliage-aware meshlet grouping**: When partitioning meshlets, prefer keeping foliage geometry grouped with nearby foliage rather than mixed with branch geometry. Foliage clusters simplify together into broader leaf masses rather than getting merged with wood and disappearing.

**No special LOD path**: The existing DAG builder handles this. The cull pass, HW/SW raster, and material resolve all work unchanged. This is purely a tuning of simplification behavior.

---

## 6. Summary of Codebase Changes

| File | Change |
|------|--------|
| `redwood_growth.rs` | Major rework: trunk taper curve, branch zones/counts/hierarchy, crown envelope, increased dimensions |
| `tube_mesh.rs` | Moderate: buttress fluting in trunk rings, handle higher branch counts |
| `foliage_billboards.rs` | **Replaced** with new `needle_sprays.rs` |
| `foliage_atlas.rs` | **Removed** |
| `alpha_mask.rs` | **Removed** |
| `simplify.rs` | Add area preservation logic |
| `RedwoodParams` | New/changed fields for columnar taper, crown envelope, branch zone counts, foliage density |
| Material shader (WGSL) | Remove MATERIAL_FOLIAGE alpha test; foliage becomes opaque shading |
| `redwood.rs` | Update public API to match new params |
| `mod.rs` (subjects) | Update module declarations |

### Not Changed
- Rendering pipeline (cull, HW raster, SW raster, shadow, HZB, material resolve, SSAO, sky, bloom, tonemap)
- Meshlet DAG builder structure (only simplification tuning)
- Camera, scene settings, soundstage presets
- Ground slab geometry

---

## Open Questions

1. **Foliage color/shading**: Current `MATERIAL_FOLIAGE` shading will need adjustment for opaque geometry (no more alpha-driven transparency). Should we design new shading in this pass or defer?
2. **Wind animation**: The current system has wind parameters in `SceneSettings`. Should foliage sprays respond to wind, or defer to a later pass?
3. **Forest instancing**: This spec covers single-tree fidelity. Forest-scale instancing (variation, placement, LOD) is a separate future spec.
