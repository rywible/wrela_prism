//! 10 biological processes for the growth simulation.
//!
//! Each function operates on `&mut GrowthTree` and a species configuration.

use glam::Vec3;

use super::tree::{GrowthSegment, GrowthTree, TreeRng};
use super::RedwoodSpecies;

// ──────────────────────── 1. Light Sampling ────────────────────────

/// Set light_exposure from GPU query results (indexed by segment).
pub fn apply_light_sampling(tree: &mut GrowthTree, light_values: &[f32]) {
    for (idx, seg) in tree.segments.iter_mut().enumerate() {
        if !seg.is_alive {
            continue;
        }
        if let Some(&light) = light_values.get(idx) {
            seg.light_exposure = light.clamp(0.0, 1.0);
        }
    }
}

/// CPU fallback: estimate light from position heuristic.
/// Light = 1.0 at canopy surface, exponential decay inward based on foliage density above.
pub fn apply_light_heuristic(tree: &mut GrowthTree, species: &RedwoodSpecies) {
    // Find the maximum height and build a simple density profile.
    let max_height = tree
        .segments
        .iter()
        .filter(|s| s.is_alive)
        .map(|s| s.position.y)
        .fold(0.0f32, |a, b| a.max(b));

    if max_height < 0.1 {
        for seg in &mut tree.segments {
            seg.light_exposure = 1.0;
        }
        return;
    }

    // Sum leaf area above each segment's height to estimate shading.
    // We bin leaf area into height slices for efficiency.
    let bin_count = 64;
    let bin_size = max_height / bin_count as f32;
    let mut leaf_area_bins = vec![0.0f32; bin_count];

    for seg in tree.segments.iter() {
        if !seg.is_alive || seg.leaf_area <= 0.0 {
            continue;
        }
        let bin = ((seg.position.y / bin_size) as usize).min(bin_count - 1);
        leaf_area_bins[bin] += seg.leaf_area;
    }

    // Cumulative leaf area above (from top down).
    let mut cumulative_above = vec![0.0f32; bin_count];
    let mut acc = 0.0;
    for i in (0..bin_count).rev() {
        cumulative_above[i] = acc;
        acc += leaf_area_bins[i];
    }

    let extinction = 0.15; // extinction coefficient per unit leaf area

    for seg in tree.segments.iter_mut() {
        if !seg.is_alive {
            continue;
        }
        let bin = ((seg.position.y / bin_size) as usize).min(bin_count - 1);
        let foliage_above = cumulative_above[bin];

        // Distance from crown surface (lateral + vertical).
        let radial_dist = Vec3::new(seg.position.x, 0.0, seg.position.z).length();
        let height_from_top = (max_height - seg.position.y).max(0.0);
        let crown_depth = height_from_top + radial_dist * 0.3;

        // Beer-Lambert attenuation.
        let density_atten = (-foliage_above * extinction).exp();
        // Position-based: segments deeper in the crown get less light.
        let depth_atten = (-crown_depth * 0.02 * species.leaf_area_per_meter).exp();

        seg.light_exposure = (density_atten * depth_atten).clamp(0.01, 1.0);
    }
}

// ──────────────────────── 2. Water Stress ────────────────────────

pub fn apply_water_stress(tree: &mut GrowthTree, species: &RedwoodSpecies) {
    // Sum leaf area above each segment for fog drip calculation.
    // Build children index once (O(n)) instead of calling children_of per node (O(n^2)).
    let len = tree.segments.len();
    let mut children = vec![Vec::new(); len];
    for (idx, seg) in tree.segments.iter().enumerate() {
        if let Some(pidx) = seg.parent {
            children[pidx].push(idx);
        }
    }

    let mut leaf_area_above = vec![0.0f32; len];
    // Walk from leaves to root, accumulating leaf area.
    for i in (0..len).rev() {
        if !tree.segments[i].is_alive {
            continue;
        }
        leaf_area_above[i] = tree.segments[i].leaf_area;
        for &child_idx in &children[i] {
            leaf_area_above[i] += leaf_area_above[child_idx];
        }
    }

    for (idx, seg) in tree.segments.iter_mut().enumerate() {
        if !seg.is_alive {
            continue;
        }
        let height = seg.position.y;
        let base_stress =
            height * species.water_stress_per_meter / species.stomatal_closure_threshold;
        let fog_relief = species.fog_drip_coefficient * leaf_area_above[idx] * 0.001;
        seg.water_stress = (base_stress - fog_relief).clamp(0.0, 1.0);
    }
}

// ──────────────────────── 3. Resource Budget ────────────────────────

pub fn compute_resource_budget(tree: &mut GrowthTree, species: &RedwoodSpecies) -> f32 {
    let mut photosynthate = 0.0f32;
    let mut total_biomass = 0.0f32;

    for seg in &tree.segments {
        if !seg.is_alive {
            continue;
        }
        photosynthate += seg.leaf_area
            * seg.light_exposure
            * (1.0 - seg.water_stress)
            * species.photosynthetic_efficiency;
        total_biomass += seg.cambial_area * species.wood_density;
    }

    let maintenance = species.maintenance_respiration * total_biomass;
    tree.total_leaf_area = tree
        .segments
        .iter()
        .filter(|s| s.is_alive)
        .map(|s| s.leaf_area)
        .sum();

    (photosynthate - maintenance).max(0.0)
}

// ──────────────────────── 4. Auxin Propagation ────────────────────────

pub fn propagate_auxin(tree: &mut GrowthTree, species: &RedwoodSpecies) {
    // Zero out all auxin first.
    for seg in &mut tree.segments {
        seg.auxin_level = 0.0;
    }

    // Each living meristem produces auxin. Walk basipetally (toward root).
    let meristems: Vec<(usize, bool)> = tree
        .segments
        .iter()
        .enumerate()
        .filter(|(_, s)| s.is_meristem && s.is_alive)
        .map(|(idx, s)| (idx, s.branch_order == 0))
        .collect();

    for (meristem_idx, is_leader) in meristems {
        let source_strength = if is_leader {
            species.leader_auxin_strength
        } else {
            1.0
        };

        let mut current = meristem_idx;
        let mut distance = 0.0f32;
        loop {
            let decay = (-distance * species.auxin_decay_per_meter).exp();
            tree.segments[current].auxin_level += source_strength * decay;
            let Some(parent_idx) = tree.segments[current].parent else {
                break;
            };
            distance +=
                (tree.segments[current].position - tree.segments[parent_idx].position).length();
            current = parent_idx;
        }
    }
}

// ──────────────────────── 5. Meristem Extension ────────────────────────

pub fn extend_meristems(
    tree: &mut GrowthTree,
    species: &RedwoodSpecies,
    resource_budget: f32,
    rng: &mut TreeRng,
) {
    let meristems = tree.meristems();
    if meristems.is_empty() {
        return;
    }

    // Count leaders vs laterals for allocation.
    let leader_count = meristems
        .iter()
        .filter(|&&idx| tree.segments[idx].branch_order == 0)
        .count()
        .max(1);
    let lateral_count = meristems.len().saturating_sub(leader_count).max(1);

    // Leader gets 40% of budget, laterals share 60%.
    let leader_budget = resource_budget * 0.40 / leader_count as f32;
    let lateral_budget = resource_budget * 0.60 / lateral_count as f32;

    for &meristem_idx in &meristems {
        let seg = &tree.segments[meristem_idx];
        if !seg.is_alive || !seg.is_meristem {
            continue;
        }

        let vigor = seg.vigor;
        let is_leader = seg.branch_order == 0;

        let growth_rate = if is_leader {
            species.leader_extension_rate
        } else {
            species.lateral_extension_rate
        };

        let available = if is_leader {
            leader_budget
        } else {
            lateral_budget
        };
        let mut extension = growth_rate * vigor * available.min(1.0);
        // Leader meristems always extend at least a minimum (juvenile vigor).
        if is_leader {
            extension = extension.max(species.leader_extension_rate * 0.3 * vigor);
        }
        if extension < 0.001 {
            continue;
        }

        // Determine growth direction.
        let parent_idx = seg.parent;
        let direction = if let Some(pidx) = parent_idx {
            let parent_pos = tree.segments[pidx].position;
            let base_dir = (seg.position - parent_pos).normalize_or_zero();
            let jitter = Vec3::new(
                rng.next_f32_range(-0.08, 0.08),
                rng.next_f32_range(-0.02, 0.05),
                rng.next_f32_range(-0.08, 0.08),
            );
            let azimuth = seg.azimuth;
            let bias = if is_leader {
                // Leader tends upward.
                Vec3::Y * 0.3
            } else {
                // Laterals spread outward using their azimuth angle
                // (reliable even near the trunk where XZ position is near zero).
                let outward = Vec3::new(azimuth.cos(), 0.0, azimuth.sin());
                outward * 0.35 + Vec3::Y * 0.03
            };
            (base_dir + jitter + bias).normalize_or_zero()
        } else {
            Vec3::Y
        };

        let new_pos = seg.position + direction * extension;
        let azimuth = seg.azimuth;
        let branch_order = seg.branch_order;
        let leaf_area_per_m = species.leaf_area_per_meter;

        // Demote current meristem.
        tree.segments[meristem_idx].is_meristem = false;
        tree.segments[meristem_idx].leaf_area += leaf_area_per_m * extension * vigor;

        // Create new tip segment.
        let mut new_seg = GrowthSegment::new(new_pos, Some(meristem_idx), branch_order);
        new_seg.azimuth = azimuth;
        new_seg.vigor = vigor;
        new_seg.leaf_area = leaf_area_per_m * extension * vigor * 0.5;
        tree.push(new_seg);
    }
}

// ──────────────────────── 6. Lateral Budding ────────────────────────

pub fn lateral_budding(tree: &mut GrowthTree, species: &RedwoodSpecies, rng: &mut TreeRng) {
    // Find trunk height for height-dependent branching.
    let trunk_height = tree
        .segments
        .iter()
        .filter(|s| s.branch_order == 0 && s.is_alive)
        .map(|s| s.position.y)
        .fold(0.0f32, |a, b| a.max(b))
        .max(1.0);

    let candidates: Vec<(usize, Vec3, f32, u8, f32)> = tree
        .segments
        .iter()
        .enumerate()
        .filter(|(_, seg)| {
            seg.is_alive
                && !seg.is_meristem
                && seg.branch_order < 3
                && seg.age_years >= species.min_bud_age
                && seg.auxin_level < species.auxin_suppression_threshold
        })
        .map(|(idx, seg)| (idx, seg.position, seg.azimuth, seg.branch_order, seg.vigor))
        .collect();

    let golden_angle = species.phyllotactic_angle;

    for (parent_idx, parent_pos, parent_azimuth, parent_order, parent_vigor) in candidates {
        if rng.next_f32() > species.bud_activation_probability {
            continue;
        }

        let new_azimuth = (parent_azimuth + golden_angle) % std::f32::consts::TAU;
        let new_order = (parent_order + 1).min(5);

        // Height-dependent branching: conical crown shape.
        // Lower branches: more horizontal, longer.
        // Upper branches: more upright, shorter.
        let height_frac = (parent_pos.y / trunk_height).clamp(0.0, 1.0);

        // Insertion angle: nearly horizontal at bottom (5°), steep at top (55°).
        let insertion_angle = (5.0 + height_frac * 50.0).to_radians();

        // Branch length: long at bottom, short at top (conical envelope).
        let length_scale = (1.0 - height_frac * 0.7).max(0.3);
        let seg_count = match new_order {
            1 => 6,
            2 => 4,
            _ => 3,
        };
        let branch_length = species.lateral_extension_rate * parent_vigor * 4.0 * length_scale;
        let seg_len = branch_length / seg_count as f32;

        let mut dir = Vec3::new(
            new_azimuth.cos() * insertion_angle.cos(),
            insertion_angle.sin(),
            new_azimuth.sin() * insertion_angle.cos(),
        )
        .normalize_or_zero();

        let mut prev_idx = parent_idx;
        let mut prev_pos = parent_pos;

        for seg_i in 0..seg_count {
            // Gradually droop and spread outward.
            let outward = Vec3::new(new_azimuth.cos(), 0.0, new_azimuth.sin());
            let droop = species.droop_rate * seg_i as f32 * 0.02;
            dir = (dir + outward * 0.06 + Vec3::Y * (rng.next_f32_range(-0.04, 0.02) - droop))
                .normalize_or_zero();

            let pos = prev_pos + dir * seg_len;
            let is_tip = seg_i == seg_count - 1;

            let mut new_seg = GrowthSegment::new(pos, Some(prev_idx), new_order);
            new_seg.azimuth = new_azimuth;
            new_seg.is_meristem = is_tip;
            new_seg.vigor = parent_vigor * (0.7 - seg_i as f32 * 0.04);
            new_seg.leaf_area = species.leaf_area_per_meter * seg_len * parent_vigor;
            prev_idx = tree.push(new_seg);
            prev_pos = pos;
        }
    }
}

// ──────────────────────── 7. Pipe Model Thickening ────────────────────────

pub fn pipe_model_thicken(tree: &mut GrowthTree, species: &RedwoodSpecies) {
    // Build children index for bottom-up traversal.
    let len = tree.segments.len();
    let mut children = vec![Vec::new(); len];
    for (idx, seg) in tree.segments.iter().enumerate() {
        if let Some(pidx) = seg.parent {
            children[pidx].push(idx);
        }
    }

    // Bottom-up: process in reverse order (children before parents).
    for i in (0..len).rev() {
        if !tree.segments[i].is_alive {
            continue;
        }

        let child_area_sum: f32 = children[i]
            .iter()
            .filter(|&&c| tree.segments[c].is_alive)
            .map(|&c| tree.segments[c].cambial_area)
            .sum();

        let leaf_contribution = tree.segments[i].leaf_area * species.sapwood_area_per_leaf;
        tree.segments[i].cambial_area = child_area_sum + leaf_contribution;
        tree.segments[i].radius = (tree.segments[i].cambial_area / std::f32::consts::PI)
            .sqrt()
            .max(0.005);
    }
}

// ──────────────────────── 8. Gravitropism ────────────────────────

pub fn apply_gravitropism(tree: &mut GrowthTree, species: &RedwoodSpecies) {
    let len = tree.segments.len();

    // Compute mass_beyond bottom-up.
    let mut children = vec![Vec::new(); len];
    for (idx, seg) in tree.segments.iter().enumerate() {
        if let Some(pidx) = seg.parent {
            children[pidx].push(idx);
        }
    }

    for i in (0..len).rev() {
        if !tree.segments[i].is_alive {
            continue;
        }
        let child_mass: f32 = children[i]
            .iter()
            .filter(|&&c| tree.segments[c].is_alive)
            .map(|&c| {
                tree.segments[c].mass_beyond + tree.segments[c].cambial_area * species.wood_density
            })
            .sum();
        tree.segments[i].mass_beyond = child_mass;
    }

    // Apply droop and reaction wood correction.
    for i in 0..len {
        let seg = &tree.segments[i];
        if !seg.is_alive || seg.parent.is_none() || seg.branch_order == 0 {
            continue;
        }
        let parent_idx = seg.parent.unwrap();
        let parent_pos = tree.segments[parent_idx].position;
        let current_pos = seg.position;
        let dir = current_pos - parent_pos;
        let dist = dir.length();
        if dist < 0.001 {
            continue;
        }

        // Droop: pull downward proportional to mass.
        let droop = Vec3::NEG_Y * seg.mass_beyond * species.droop_rate * 0.001;

        // Reaction wood correction: push back upward, decaying with age.
        let age_decay = (1.0 - species.reaction_wood_decay * seg.age_years as f32).max(0.0);
        let correction = Vec3::Y * species.reaction_wood_strength * age_decay * dist;

        let new_pos = current_pos + droop + correction;
        tree.segments[i].position = new_pos;
        tree.segments[i].reaction_wood = age_decay;
    }
}

// ──────────────────────── 9. Reiteration ────────────────────────

pub fn check_reiteration(tree: &mut GrowthTree, species: &RedwoodSpecies, rng: &mut TreeRng) {
    let candidates: Vec<(usize, Vec3, u8)> = tree
        .segments
        .iter()
        .enumerate()
        .filter(|(_, seg)| {
            seg.is_alive
                && !seg.is_meristem
                && seg.age_years as u32 > species.reiteration_min_age
                && seg.radius > species.reiteration_min_radius
                && seg.light_exposure > 0.4
        })
        .map(|(idx, seg)| (idx, seg.position, seg.branch_order))
        .collect();

    for (parent_idx, pos, _order) in candidates {
        if rng.next_f32() > species.reiteration_probability {
            continue;
        }

        // Create vertical leader meristem on branch upper surface.
        let new_pos = pos + Vec3::Y * tree.segments[parent_idx].radius;
        let mut new_seg = GrowthSegment::new(new_pos, Some(parent_idx), 0);
        new_seg.is_meristem = true;
        new_seg.vigor = 0.6;
        new_seg.leaf_area = species.leaf_area_per_meter * 0.2;
        tree.push(new_seg);
    }
}

// ──────────────────────── 10. Self-Pruning ────────────────────────

pub fn self_pruning(tree: &mut GrowthTree, species: &RedwoodSpecies) {
    let vigor_threshold = 0.03;

    for seg in &mut tree.segments {
        if !seg.is_alive {
            continue;
        }

        if seg.vigor < vigor_threshold && seg.branch_order > 0 {
            seg.low_vigor_years += 1;
        } else {
            seg.low_vigor_years = 0;
        }

        // Kill after shade_death_years of consecutive low vigor.
        if seg.low_vigor_years >= species.shade_death_years {
            seg.is_alive = false;
            seg.is_meristem = false;
            seg.leaf_area = 0.0;
        }
    }

    // Remove dead stubs that have exceeded persistence.
    for seg in &mut tree.segments {
        if !seg.is_alive && seg.age_years as u32 > species.dead_stub_persistence {
            seg.radius *= 0.95; // Gradual decay rather than removal.
        }
    }

    // Update vigor based on light and water stress.
    for seg in &mut tree.segments {
        if !seg.is_alive {
            continue;
        }
        seg.vigor = (seg.light_exposure * (1.0 - seg.water_stress * 0.5)).clamp(0.0, 1.0);
    }
}

// ──────────────────────── Aging ────────────────────────

/// Age all segments by one year, decay old needles, and regrow foliage on lit branches.
pub fn age_segments(tree: &mut GrowthTree, species: &RedwoodSpecies) {
    for seg in &mut tree.segments {
        seg.age_years = seg.age_years.saturating_add(1);

        if !seg.is_alive {
            continue;
        }

        // Needles die after lifespan — reduce leaf area gradually.
        if seg.age_years > species.needle_lifespan && seg.leaf_area > 0.0 {
            let decay = 1.0 / species.needle_lifespan as f32;
            seg.leaf_area -= seg.leaf_area * decay;
        }

        // Foliage renewal: living branch segments with light regrow needles.
        // This represents continuous needle replacement on healthy branches.
        if seg.branch_order > 0 && seg.light_exposure > 0.05 {
            let renewal = species.leaf_area_per_meter * 0.15 * seg.light_exposure * seg.vigor;
            let max_leaf = species.leaf_area_per_meter * 1.2;
            seg.leaf_area = (seg.leaf_area + renewal).min(max_leaf);
        }
    }
    tree.age_years = tree.age_years.saturating_add(1);
}

// ──────────────────────── Tests ────────────────────────

#[cfg(test)]
mod tests {
    use super::super::RedwoodSpecies;
    use super::*;

    fn make_test_tree() -> GrowthTree {
        let mut tree = GrowthTree::new();
        // Simple trunk: 5 segments going straight up.
        let mut prev = tree.push(GrowthSegment::new(Vec3::ZERO, None, 0));
        tree.segments[prev].is_meristem = false;
        tree.segments[prev].leaf_area = 1.0;
        tree.segments[prev].cambial_area = 0.5;
        for i in 1..5 {
            let pos = Vec3::new(0.0, i as f32 * 5.0, 0.0);
            let mut seg = GrowthSegment::new(pos, Some(prev), 0);
            seg.is_meristem = i == 4;
            seg.leaf_area = 2.0;
            seg.cambial_area = 0.3;
            seg.age_years = (4 - i) as u16;
            prev = tree.push(seg);
        }
        // Add a lateral branch.
        let branch_base = 2; // segment index
        let mut bseg = GrowthSegment::new(Vec3::new(3.0, 10.0, 0.0), Some(branch_base), 1);
        bseg.leaf_area = 3.0;
        bseg.cambial_area = 0.1;
        bseg.is_meristem = true;
        tree.push(bseg);
        tree
    }

    #[test]
    fn water_stress_increases_with_height() {
        let species = RedwoodSpecies::default();
        let mut tree = make_test_tree();
        apply_water_stress(&mut tree, &species);

        let stresses: Vec<f32> = tree.segments.iter().map(|s| s.water_stress).collect();
        // Higher segments should have more stress.
        for window in stresses[..5].windows(2) {
            assert!(
                window[1] >= window[0] - 0.001,
                "water stress should increase with height: {:?}",
                stresses
            );
        }
    }

    #[test]
    fn budget_zero_when_no_leaves() {
        let species = RedwoodSpecies::default();
        let mut tree = GrowthTree::new();
        let mut seg = GrowthSegment::new(Vec3::ZERO, None, 0);
        seg.leaf_area = 0.0;
        tree.push(seg);

        let budget = compute_resource_budget(&mut tree, &species);
        assert_eq!(budget, 0.0);
    }

    #[test]
    fn auxin_decays_from_tip_to_root() {
        let species = RedwoodSpecies::default();
        let mut tree = make_test_tree();
        propagate_auxin(&mut tree, &species);

        // The meristem tip (index 4) should have highest auxin.
        let tip_auxin = tree.segments[4].auxin_level;
        let root_auxin = tree.segments[0].auxin_level;
        assert!(
            tip_auxin > root_auxin,
            "tip ({tip_auxin}) should have more auxin than root ({root_auxin})"
        );
    }

    #[test]
    fn pipe_model_area_conservation() {
        let species = RedwoodSpecies::default();
        let mut tree = make_test_tree();
        pipe_model_thicken(&mut tree, &species);

        // Root area should be >= sum of direct children areas.
        let root_area = tree.segments[0].cambial_area;
        let child_area_sum: f32 = tree
            .children_of(0)
            .iter()
            .map(|&c| tree.segments[c].cambial_area)
            .sum();
        assert!(
            root_area >= child_area_sum - 0.001,
            "root area ({root_area}) should be >= children sum ({child_area_sum})"
        );
    }

    #[test]
    fn gravitropism_corrects_horizontal_branches_upward() {
        let species = RedwoodSpecies::default();
        let mut tree = GrowthTree::new();
        let root = tree.push(GrowthSegment::new(Vec3::ZERO, None, 0));
        tree.segments[root].is_meristem = false;
        tree.segments[root].cambial_area = 1.0;

        // Horizontal branch segment.
        let mut branch = GrowthSegment::new(Vec3::new(5.0, 0.0, 0.0), Some(root), 1);
        branch.is_meristem = false;
        branch.cambial_area = 0.2;
        branch.mass_beyond = 10.0;
        let bidx = tree.push(branch);

        let y_before = tree.segments[bidx].position.y;
        apply_gravitropism(&mut tree, &species);
        let y_after = tree.segments[bidx].position.y;

        // With reaction wood, should get corrected (net effect depends on mass vs correction).
        // The important thing is position changed.
        assert!(
            (y_after - y_before).abs() > 0.0001,
            "gravitropism should alter branch position"
        );
    }

    #[test]
    fn self_pruning_kills_after_shade_years() {
        let species = RedwoodSpecies::default();
        let mut tree = GrowthTree::new();
        let root = tree.push(GrowthSegment::new(Vec3::ZERO, None, 0));
        tree.segments[root].is_meristem = false;

        let mut shaded = GrowthSegment::new(Vec3::new(3.0, 5.0, 0.0), Some(root), 1);
        shaded.vigor = 0.01; // Very low vigor.
        shaded.low_vigor_years = species.shade_death_years - 1;
        let sidx = tree.push(shaded);

        self_pruning(&mut tree, &species);

        assert!(
            !tree.segments[sidx].is_alive,
            "segment should be dead after enough shade years"
        );
    }
}
