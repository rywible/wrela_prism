//! Biological redwood growth simulation.
//!
//! Three layers:
//! 1. CPU growth sim — year-by-year biological processes
//! 2. GPU light field — 3D voxel volume for light availability
//! 3. Skeleton conversion — outputs same TreeSkeleton as existing pipeline

pub mod biology;
pub mod light_field;
pub mod skeleton;
pub mod tree;

use glam::Vec3;
use tree::{GrowthSegment, GrowthTree, TreeRng};

// ──────────────────────── Species Parameters ────────────────────────

/// Biologically-meaningful parameters for coast redwood (Sequoia sempervirens).
#[derive(Clone, Debug)]
pub struct RedwoodSpecies {
    // Geometry
    pub max_height: f32,
    pub base_diameter: f32,

    // Growth rates
    pub leader_extension_rate: f32,
    pub lateral_extension_rate: f32,

    // Branching
    pub phyllotactic_angle: f32,
    pub bud_activation_probability: f32,
    pub auxin_decay_per_meter: f32,
    pub auxin_suppression_threshold: f32,
    pub leader_auxin_strength: f32,
    pub min_bud_age: u16,
    pub insertion_angle: f32,

    // Hydraulics
    pub water_stress_per_meter: f32,
    pub fog_drip_coefficient: f32,
    pub stomatal_closure_threshold: f32,

    // Photosynthesis
    pub photosynthetic_efficiency: f32,
    pub maintenance_respiration: f32,

    // Mechanical
    pub wood_density: f32,
    pub droop_rate: f32,
    pub reaction_wood_strength: f32,
    pub reaction_wood_decay: f32,
    pub sapwood_area_per_leaf: f32,

    // Reiteration
    pub reiteration_min_age: u32,
    pub reiteration_min_radius: f32,
    pub reiteration_probability: f32,

    // Mortality
    pub shade_death_years: u16,
    pub dead_stub_persistence: u32,

    // Foliage
    pub leaf_area_per_meter: f32,
    pub needle_lifespan: u16,

    // Bark (passthrough to tube_mesh)
    pub flute_count: u32,
    pub flute_depth: f32,
}

impl Default for RedwoodSpecies {
    fn default() -> Self {
        Self {
            max_height: 65.0,
            base_diameter: 9.0,

            leader_extension_rate: 0.45,
            lateral_extension_rate: 0.7,

            phyllotactic_angle: 137.508_f32.to_radians(), // golden angle
            bud_activation_probability: 0.15,
            auxin_decay_per_meter: 0.4,
            auxin_suppression_threshold: 2.0,
            leader_auxin_strength: 5.0,
            min_bud_age: 3,
            insertion_angle: 25.0,

            water_stress_per_meter: 0.008,
            fog_drip_coefficient: 0.15,
            stomatal_closure_threshold: 0.7,

            photosynthetic_efficiency: 0.25,
            maintenance_respiration: 0.001,

            wood_density: 420.0,
            droop_rate: 0.35,
            reaction_wood_strength: 0.08,
            reaction_wood_decay: 0.02,
            sapwood_area_per_leaf: 0.08,

            reiteration_min_age: 50,
            reiteration_min_radius: 0.3,
            reiteration_probability: 0.02,

            shade_death_years: 8,
            dead_stub_persistence: 30,

            leaf_area_per_meter: 2.5,
            needle_lifespan: 5,

            flute_count: 8,
            flute_depth: 0.20,
        }
    }
}

// ──────────────────────── Growth Parameters ────────────────────────

/// Top-level parameters for the growth simulation.
#[derive(Clone, Debug)]
pub struct GrowthParams {
    pub seed: u64,
    pub species: RedwoodSpecies,
    pub target_age: u16,
    pub enable_light_field: bool,
}

impl Default for GrowthParams {
    fn default() -> Self {
        Self {
            seed: 42,
            species: RedwoodSpecies::default(),
            target_age: 500,
            enable_light_field: false,
        }
    }
}

// ──────────────────────── GPU Access ────────────────────────

/// Borrowed references to GPU device and queue for light field computation.
pub struct GpuAccess<'a> {
    pub device: &'a wgpu::Device,
    pub queue: &'a wgpu::Queue,
}

// ──────────────────────── Public API ────────────────────────

/// Grow a tree using the biological simulation with optional GPU light field.
pub fn grow_tree(params: &GrowthParams, gpu: Option<GpuAccess<'_>>) -> GrowthTree {
    if params.enable_light_field {
        if let Some(gpu) = gpu {
            return grow_tree_gpu(params, gpu.device, gpu.queue);
        }
    }
    grow_tree_cpu(params)
}

/// Grow a tree using CPU-only light heuristic (no GPU required).
pub fn grow_tree_cpu(params: &GrowthParams) -> GrowthTree {
    let mut rng = TreeRng::new(params.seed);
    let mut tree = initialize_seedling(&params.species);

    for year in 0..params.target_age {
        simulate_year_cpu(&mut tree, &params.species, &mut rng);

        // Log progress at milestones.
        if (year + 1) % 100 == 0 {
            tracing::debug!(
                "growth year {}: {} segments, {:.0} leaf area",
                year + 1,
                tree.segments.len(),
                tree.total_leaf_area,
            );
        }
    }

    tree
}

/// Grow a tree with GPU-accelerated light field queries.
fn grow_tree_gpu(params: &GrowthParams, device: &wgpu::Device, queue: &wgpu::Queue) -> GrowthTree {
    let mut rng = TreeRng::new(params.seed);
    let mut tree = initialize_seedling(&params.species);

    let light_field = light_field::LightFieldGpu::new(device);

    for year in 0..params.target_age {
        // Query GPU light field periodically (more often when young).
        let query_interval = (tree.age_years as u32 / 50).max(1);
        let use_gpu_light = (year as u32).is_multiple_of(query_interval);

        if use_gpu_light {
            let capsules: Vec<_> = tree
                .segments
                .iter()
                .filter(|s| s.is_alive && s.leaf_area > 0.0)
                .filter_map(|s| {
                    s.parent.map(|pidx| {
                        let parent = &tree.segments[pidx];
                        (
                            parent.position,
                            s.position,
                            parent.radius,
                            s.radius,
                            s.leaf_area,
                        )
                    })
                })
                .collect();

            let meristems = tree.meristems();
            let positions: Vec<Vec3> = meristems
                .iter()
                .map(|&idx| tree.segments[idx].position)
                .collect();

            if !positions.is_empty() && !capsules.is_empty() {
                let light_values =
                    light_field.query_meristem_light(device, queue, &capsules, &positions);
                // Map light values back to segment indices.
                let mut full_light = vec![0.5f32; tree.segments.len()];
                for (i, &midx) in meristems.iter().enumerate() {
                    if let Some(&val) = light_values.get(i) {
                        full_light[midx] = val;
                    }
                }
                biology::apply_light_sampling(&mut tree, &full_light);
            } else {
                biology::apply_light_heuristic(&mut tree, &params.species);
            }
        } else {
            biology::apply_light_heuristic(&mut tree, &params.species);
        }

        // Run remaining biological processes.
        simulate_year_after_light(&mut tree, &params.species, &mut rng);
    }

    tree
}

// ──────────────────────── Simulation Core ────────────────────────

fn initialize_seedling(species: &RedwoodSpecies) -> GrowthTree {
    let mut tree = GrowthTree::new();

    // Root segment at ground level.
    let mut root = GrowthSegment::new(Vec3::ZERO, None, 0);
    root.radius = species.base_diameter / 2.0 * 0.01; // Seedling starts tiny.
    root.is_meristem = false;
    root.leaf_area = species.leaf_area_per_meter * 0.5;
    tree.push(root);

    // Initial leader meristem.
    let mut leader = GrowthSegment::new(Vec3::Y * 0.3, Some(0), 0);
    leader.is_meristem = true;
    leader.vigor = 1.0;
    leader.leaf_area = species.leaf_area_per_meter * 0.3;
    tree.push(leader);

    tree
}

/// Full CPU year: light heuristic + all biological processes.
fn simulate_year_cpu(tree: &mut GrowthTree, species: &RedwoodSpecies, rng: &mut TreeRng) {
    biology::apply_light_heuristic(tree, species);
    simulate_year_after_light(tree, species, rng);
}

/// Biological processes that run after light values are set.
fn simulate_year_after_light(tree: &mut GrowthTree, species: &RedwoodSpecies, rng: &mut TreeRng) {
    biology::apply_water_stress(tree, species);
    let budget = biology::compute_resource_budget(tree, species);
    biology::propagate_auxin(tree, species);
    biology::extend_meristems(tree, species, budget, rng);
    biology::lateral_budding(tree, species, rng);
    biology::pipe_model_thicken(tree, species);
    biology::apply_gravitropism(tree, species);
    biology::check_reiteration(tree, species, rng);
    biology::self_pruning(tree, species);
    biology::age_segments(tree, species);
}

// ──────────────────────── Tests ────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_growth_produces_nontrivial_tree() {
        let params = GrowthParams {
            target_age: 50,
            ..GrowthParams::default()
        };
        let tree = grow_tree_cpu(&params);

        assert!(
            tree.segments.len() > 10,
            "50-year tree should have meaningful segment count, got {}",
            tree.segments.len()
        );
        assert_eq!(tree.age_years, 50);
    }

    #[test]
    fn growth_determinism() {
        let params = GrowthParams {
            target_age: 20,
            ..GrowthParams::default()
        };
        let a = grow_tree_cpu(&params);
        let b = grow_tree_cpu(&params);

        assert_eq!(a.segments.len(), b.segments.len());
        for (sa, sb) in a.segments.iter().zip(b.segments.iter()) {
            assert_eq!(sa.position, sb.position);
            assert_eq!(sa.radius, sb.radius);
        }
    }

    #[test]
    fn different_seeds_produce_variation() {
        let a = grow_tree_cpu(&GrowthParams {
            seed: 1,
            target_age: 30,
            ..GrowthParams::default()
        });
        let b = grow_tree_cpu(&GrowthParams {
            seed: 2,
            target_age: 30,
            ..GrowthParams::default()
        });

        let differs = a.segments.len() != b.segments.len()
            || a.segments
                .iter()
                .zip(b.segments.iter())
                .any(|(sa, sb)| sa.position != sb.position);
        assert!(differs, "different seeds should produce different trees");
    }

    #[test]
    fn trunk_height_increases_with_age() {
        let young = grow_tree_cpu(&GrowthParams {
            target_age: 20,
            ..GrowthParams::default()
        });
        let old = grow_tree_cpu(&GrowthParams {
            target_age: 100,
            ..GrowthParams::default()
        });

        let max_y = |tree: &GrowthTree| -> f32 {
            tree.segments
                .iter()
                .filter(|s| s.branch_order == 0)
                .map(|s| s.position.y)
                .fold(0.0f32, |a, b| a.max(b))
        };

        assert!(max_y(&old) > max_y(&young), "older tree should be taller");
    }

    #[test]
    fn grow_tree_with_none_gpu_uses_cpu() {
        let params = GrowthParams {
            target_age: 10,
            enable_light_field: true, // Requested but no GPU provided.
            ..GrowthParams::default()
        };
        let tree = grow_tree(&params, None);
        assert!(tree.segments.len() > 2);
    }
}
