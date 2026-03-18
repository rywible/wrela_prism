//! Growth tree data structures for biological simulation.

use glam::Vec3;

pub use crate::util::Rng as TreeRng;

// ──────────────────────── Growth Segment ────────────────────────

/// A single segment in the growth tree with biological state.
#[derive(Clone, Debug)]
pub struct GrowthSegment {
    pub position: Vec3,
    pub radius: f32,
    pub parent: Option<usize>,

    // Biological state
    pub leaf_area: f32,
    pub light_exposure: f32,
    pub water_stress: f32,
    pub auxin_level: f32,
    pub age_years: u16,
    pub is_alive: bool,
    pub is_meristem: bool,
    pub cambial_area: f32,
    pub branch_order: u8,
    pub azimuth: f32,
    pub mass_beyond: f32,
    pub reaction_wood: f32,
    pub vigor: f32,
    pub low_vigor_years: u16,
}

impl GrowthSegment {
    pub fn new(position: Vec3, parent: Option<usize>, branch_order: u8) -> Self {
        Self {
            position,
            radius: 0.01,
            parent,
            leaf_area: 0.0,
            light_exposure: 1.0,
            water_stress: 0.0,
            auxin_level: 0.0,
            age_years: 0,
            is_alive: true,
            is_meristem: true,
            cambial_area: 0.0,
            branch_order,
            azimuth: 0.0,
            mass_beyond: 0.0,
            reaction_wood: 0.0,
            vigor: 1.0,
            low_vigor_years: 0,
        }
    }
}

// ──────────────────────── Growth Tree ────────────────────────

/// A tree represented as a collection of growth segments forming a DAG.
#[derive(Clone, Debug)]
pub struct GrowthTree {
    pub segments: Vec<GrowthSegment>,
    pub age_years: u16,
    pub total_leaf_area: f32,
}

impl Default for GrowthTree {
    fn default() -> Self {
        Self {
            segments: Vec::new(),
            age_years: 0,
            total_leaf_area: 0.0,
        }
    }
}

impl GrowthTree {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, segment: GrowthSegment) -> usize {
        let idx = self.segments.len();
        self.segments.push(segment);
        idx
    }

    /// Returns indices of all children of the given segment.
    pub fn children_of(&self, parent_idx: usize) -> Vec<usize> {
        self.segments
            .iter()
            .enumerate()
            .filter(|(_, seg)| seg.parent == Some(parent_idx))
            .map(|(idx, _)| idx)
            .collect()
    }

    /// Returns indices of all living meristem tips.
    pub fn meristems(&self) -> Vec<usize> {
        self.segments
            .iter()
            .enumerate()
            .filter(|(_, seg)| seg.is_meristem && seg.is_alive)
            .map(|(idx, _)| idx)
            .collect()
    }

    /// Returns capsule data (a, b, radius_a, radius_b) for all parent-child pairs.
    pub fn capsules(&self) -> Vec<(Vec3, Vec3, f32, f32)> {
        let mut result = Vec::new();
        for seg in &self.segments {
            if let Some(parent_idx) = seg.parent {
                let parent = &self.segments[parent_idx];
                result.push((parent.position, seg.position, parent.radius, seg.radius));
            }
        }
        result
    }
}

// ──────────────────────── Tests ────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parent_invariants() {
        let mut tree = GrowthTree::new();
        let root = tree.push(GrowthSegment::new(Vec3::ZERO, None, 0));
        let child = tree.push(GrowthSegment::new(Vec3::Y, Some(root), 0));
        let grandchild = tree.push(GrowthSegment::new(Vec3::new(0.0, 2.0, 0.0), Some(child), 1));

        assert!(tree.segments[root].parent.is_none());
        assert_eq!(tree.segments[child].parent, Some(root));
        assert_eq!(tree.segments[grandchild].parent, Some(child));
    }

    #[test]
    fn children_of_correctness() {
        let mut tree = GrowthTree::new();
        let root = tree.push(GrowthSegment::new(Vec3::ZERO, None, 0));
        let c1 = tree.push(GrowthSegment::new(Vec3::Y, Some(root), 0));
        let c2 = tree.push(GrowthSegment::new(Vec3::X, Some(root), 1));
        let _gc = tree.push(GrowthSegment::new(Vec3::new(0.0, 2.0, 0.0), Some(c1), 0));

        let children = tree.children_of(root);
        assert_eq!(children.len(), 2);
        assert!(children.contains(&c1));
        assert!(children.contains(&c2));

        let c1_children = tree.children_of(c1);
        assert_eq!(c1_children.len(), 1);
    }

    #[test]
    fn meristem_filtering() {
        let mut tree = GrowthTree::new();
        let root = tree.push(GrowthSegment::new(Vec3::ZERO, None, 0));
        tree.segments[root].is_meristem = false;
        let tip = tree.push(GrowthSegment::new(Vec3::Y, Some(root), 0));
        let dead_tip = tree.push(GrowthSegment::new(Vec3::X, Some(root), 1));
        tree.segments[dead_tip].is_alive = false;

        let meristems = tree.meristems();
        assert_eq!(meristems.len(), 1);
        assert!(meristems.contains(&tip));
    }

    #[test]
    fn capsules_produces_parent_child_pairs() {
        let mut tree = GrowthTree::new();
        let root = tree.push(GrowthSegment::new(Vec3::ZERO, None, 0));
        tree.segments[root].radius = 1.0;
        let child = tree.push(GrowthSegment::new(Vec3::Y * 5.0, Some(root), 0));
        tree.segments[child].radius = 0.5;

        let caps = tree.capsules();
        assert_eq!(caps.len(), 1);
        assert_eq!(caps[0].0, Vec3::ZERO);
        assert_eq!(caps[0].1, Vec3::Y * 5.0);
        assert_eq!(caps[0].2, 1.0);
        assert_eq!(caps[0].3, 0.5);
    }

    #[test]
    fn rng_determinism() {
        let mut a = TreeRng::new(12345);
        let mut b = TreeRng::new(12345);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn rng_f32_in_range() {
        let mut rng = TreeRng::new(42);
        for _ in 0..1000 {
            let v = rng.next_f32();
            assert!((0.0..1.0).contains(&v), "next_f32 out of range: {v}");
        }
    }
}
