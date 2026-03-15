/// Integer hash → [0, 1).
pub fn hash01(n: u32) -> f32 {
    let n = n.wrapping_mul(747796405).wrapping_add(2891336453);
    let n = ((n >> ((n >> 28).wrapping_add(4))) ^ n).wrapping_mul(277803737);
    let n = (n >> 22) ^ n;
    (n as f32) / (u32::MAX as f32)
}

/// Hermite smoothstep.
pub fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}
