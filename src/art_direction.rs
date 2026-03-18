use bytemuck::{Pod, Zeroable};
use std::sync::Arc;
use std::time::Duration;

// ──────────────────────── Style Axes ────────────────────────

/// High-level, intuitive style knobs. All f32, all interpolatable.
/// Default: all zeros = photorealistic (except `palette_warmth: 0.5`).
#[derive(Clone, Copy, Debug)]
pub struct StyleAxes {
    pub softness: f32,
    pub exaggeration: f32,
    pub shadow_graphicness: f32,
    pub palette_warmth: f32,
    pub atmospheric_depth: f32,
    pub surface_detail: f32,
    pub outline_presence: f32,
    pub color_discipline: f32,
}

impl Default for StyleAxes {
    fn default() -> Self {
        Self {
            softness: 0.0,
            exaggeration: 0.0,
            shadow_graphicness: 0.0,
            palette_warmth: 0.5,
            atmospheric_depth: 0.0,
            surface_detail: 0.0,
            outline_presence: 0.0,
            color_discipline: 0.0,
        }
    }
}

// ──────────────────────── GPU Uniforms ────────────────────────

/// GPU-side art direction parameters. All vec3s padded to [f32; 4].
/// All-zero = identity (no style transforms).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct ArtDirectionUniforms {
    // Tier 1: Geometry Deformation
    pub silhouette_inflate: f32,
    pub normal_smoothing: f32,
    pub detail_suppression: f32,
    pub color_discipline: f32,

    // Tier 2: Shading Transforms
    pub shading_bands: f32,
    pub band_softness: f32,
    pub specular_flattening: f32,
    pub rim_light_strength: f32,

    pub shadow_color_shift: [f32; 4], // xyz = hue offset, w = pad
    pub rim_light_color: [f32; 4],    // xyz = tint, w = bark_detail_blend

    pub palette_saturation: f32,
    pub palette_hue_shift: f32,
    pub palette_value_contrast: f32,
    pub foliage_transmission_boost: f32,

    // Tier 3: Post-Processing
    pub outline_strength: f32,
    pub outline_thickness: f32,
    pub outline_depth_sensitivity: f32,
    pub outline_normal_sensitivity: f32,

    pub outline_color: [f32; 4], // xyz = color, w = pad

    pub color_grade_tint: [f32; 4], // xyz = tint, w = strength

    pub bloom_tint: [f32; 4], // xyz = bloom color, w = bloom_softness

    pub detail_fade_distance: f32,
    pub skip_bark_texture: f32,        // >0.5 = skip bark texture sampling
    pub skip_normal_perturbation: f32, // >0.5 = skip normal map perturbation
    pub foliage_shadow_band_softness: f32,

    pub wind_amplitude_scale: f32,
    pub wind_response_scale: f32,
    pub wind_gust_sharpness: f32,
    pub _pad4d: f32,
}

// ──────────────────────── Style Palette ────────────────────────

/// Discrete color remapping palette. 9 slots × 4 floats = 144 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct StylePalette {
    // Tonal slots
    pub shadow: [f32; 4],
    pub dark: [f32; 4],
    pub mid: [f32; 4],
    pub light: [f32; 4],
    pub accent: [f32; 4],

    // Material-hint slots
    pub bark: [f32; 4],
    pub foliage: [f32; 4],
    pub earth: [f32; 4],
    pub sky_tint: [f32; 4],
}

// ──────────────────────── Style Snapshot ────────────────────────

/// Bundles axes + palette for serialization/transitions.
#[derive(Clone, Debug)]
pub struct StyleSnapshot {
    pub axes: StyleAxes,
    pub palette: StylePalette,
}

impl Default for StyleSnapshot {
    fn default() -> Self {
        Self {
            axes: StyleAxes::default(),
            palette: StylePalette::zeroed(),
        }
    }
}

// ──────────────────────── Ease Curves ────────────────────────

#[derive(Clone, Copy, Debug)]
pub enum EaseCurve {
    Linear,
    EaseIn,
    EaseOut,
    EaseInOut,
}

impl EaseCurve {
    pub fn evaluate(&self, t: f32) -> f32 {
        let t = t.clamp(0.0, 1.0);
        match self {
            EaseCurve::Linear => t,
            EaseCurve::EaseIn => t * t,
            EaseCurve::EaseOut => 1.0 - (1.0 - t) * (1.0 - t),
            EaseCurve::EaseInOut => {
                if t < 0.5 {
                    2.0 * t * t
                } else {
                    1.0 - (-2.0 * t + 2.0).powi(2) / 2.0
                }
            }
        }
    }
}

// ──────────────────────── Quality Budget ────────────────────────

#[derive(Clone, Copy, Debug)]
pub struct QualityBudget {
    pub lod_bias: f32,
    pub shadow_resolution_scale: f32,
    pub ssao_sample_scale: f32,
    pub skip_bark_texture: bool,
    pub skip_normal_perturbation: bool,
    pub simplify_foliage_alpha: bool,
}

impl Default for QualityBudget {
    fn default() -> Self {
        Self {
            lod_bias: 0.0,
            shadow_resolution_scale: 1.0,
            ssao_sample_scale: 1.0,
            skip_bark_texture: false,
            skip_normal_perturbation: false,
            simplify_foliage_alpha: false,
        }
    }
}

// ──────────────────────── Semantic Channel Packing ────────────────────────

/// Pack four [0,1] channels into a single u32 (4×u8).
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

/// Unpack a u32 back to four [0,1] channels.
pub fn unpack_semantic_channels(packed: u32) -> (f32, f32, f32, f32) {
    let c = (packed & 0xFF) as f32 / 255.0;
    let e = ((packed >> 8) & 0xFF) as f32 / 255.0;
    let s = ((packed >> 16) & 0xFF) as f32 / 255.0;
    let i = ((packed >> 24) & 0xFF) as f32 / 255.0;
    (c, e, s, i)
}

// ──────────────────────── Axis→Uniform Mapping ────────────────────────

/// Hand-authored mapping from intuitive axes to GPU parameters.
pub fn map_axes_to_uniforms(axes: &StyleAxes) -> ArtDirectionUniforms {
    ArtDirectionUniforms {
        silhouette_inflate: axes.exaggeration * 0.15,
        normal_smoothing: axes.softness * 0.6,
        // Banding requires smooth surfaces — both softness and graphicness drive suppression
        detail_suppression: (axes.softness * 0.4 + axes.shadow_graphicness * 0.55).min(1.0),
        color_discipline: axes.color_discipline,

        shading_bands: axes.shadow_graphicness * 3.0,
        band_softness: 0.08 + (1.0 - axes.shadow_graphicness) * 0.15,
        specular_flattening: axes.softness * 0.75,
        rim_light_strength: axes.exaggeration * 3.0,

        shadow_color_shift: [0.0; 4],
        rim_light_color: [1.0, 1.0, 1.0, 0.0],

        palette_saturation: 1.0,
        palette_hue_shift: (axes.palette_warmth - 0.5) * 30.0,
        palette_value_contrast: 1.0,
        foliage_transmission_boost: 1.0,

        outline_strength: axes.outline_presence * 1.2,
        outline_thickness: 1.0 + axes.outline_presence * 1.5,
        outline_depth_sensitivity: 2.0 + axes.outline_presence,
        outline_normal_sensitivity: 0.3 + axes.outline_presence * 0.2,

        outline_color: [0.01, 0.005, 0.02, 0.0],

        color_grade_tint: [1.0, 1.0, 1.0, 0.0],

        bloom_tint: [1.0, 1.0, 1.0, 0.0],

        detail_fade_distance: 100.0 - axes.surface_detail * 60.0,
        skip_bark_texture: if axes.softness > 0.7 && axes.surface_detail < 0.3 {
            1.0
        } else {
            0.0
        },
        skip_normal_perturbation: if axes.softness > 0.6 { 1.0 } else { 0.0 },
        foliage_shadow_band_softness: 0.14 + axes.shadow_graphicness * 0.12,
        wind_amplitude_scale: 1.0 + axes.exaggeration * 0.35,
        wind_response_scale: 1.0 + axes.exaggeration * 0.25,
        wind_gust_sharpness: 1.0 + axes.shadow_graphicness * 1.5,
        _pad4d: 0.0,
    }
}

/// Compute quality budget from style axes.
pub fn compute_quality_budget(axes: &StyleAxes) -> QualityBudget {
    let stylization = axes
        .shadow_graphicness
        .max(axes.softness)
        .max(axes.exaggeration);
    QualityBudget {
        lod_bias: stylization * 1.5,
        shadow_resolution_scale: 1.0 - stylization * 0.5,
        ssao_sample_scale: 1.0 - axes.softness * 0.4,
        skip_bark_texture: axes.softness > 0.7 && axes.surface_detail < 0.3,
        skip_normal_perturbation: axes.softness > 0.6,
        simplify_foliage_alpha: axes.softness > 0.5,
    }
}

// ──────────────────────── Named Style Constants ────────────────────────

pub const IDENTITY: StyleSnapshot = StyleSnapshot {
    axes: StyleAxes {
        softness: 0.0,
        exaggeration: 0.0,
        shadow_graphicness: 0.0,
        palette_warmth: 0.5,
        atmospheric_depth: 0.0,
        surface_detail: 0.0,
        outline_presence: 0.0,
        color_discipline: 0.0,
    },
    palette: StylePalette {
        shadow: [0.0; 4],
        dark: [0.0; 4],
        mid: [0.0; 4],
        light: [0.0; 4],
        accent: [0.0; 4],
        bark: [0.0; 4],
        foliage: [0.0; 4],
        earth: [0.0; 4],
        sky_tint: [0.0; 4],
    },
};

pub const DEMON_SLAYER: StyleSnapshot = StyleSnapshot {
    axes: StyleAxes {
        softness: 0.45,
        exaggeration: 0.70,
        shadow_graphicness: 0.70,
        palette_warmth: 0.40,
        atmospheric_depth: 0.3,
        surface_detail: 0.4,
        outline_presence: 0.80,
        color_discipline: 0.30,
    },
    palette: StylePalette {
        // ufotable shadows: rich blue-violet, not crushed
        shadow: [0.16, 0.12, 0.26, 0.0],
        // dark midtone: warm indigo
        dark: [0.28, 0.22, 0.36, 0.0],
        // primary midtone: natural with violet shift
        mid: [0.52, 0.46, 0.56, 0.0],
        // highlights: warm cream
        light: [1.0, 0.92, 0.82, 0.0],
        // accent: ufotable warm red-pink for rim lights
        accent: [1.0, 0.40, 0.50, 0.0],
        // bark: warm reddish-brown
        bark: [0.38, 0.20, 0.14, 0.0],
        // foliage: vivid saturated green
        foliage: [0.30, 0.50, 0.22, 0.0],
        // earth: warm ochre-brown
        earth: [0.30, 0.22, 0.14, 0.0],
        // sky_tint: subtle cool atmospheric grade (values near 1.0 for multiplicative)
        sky_tint: [0.92, 0.88, 0.96, 0.0],
    },
};

// ──────────────────────── Post-Processing Params ────────────────────────

/// Parameters for bloom and tonemap passes, derived from art direction.
pub struct PostProcessingParams {
    pub bloom_tint: [f32; 3],
    pub bloom_softness: f32,
    pub color_grade: [f32; 4],
}

// ──────────────────────── Transition Engine ────────────────────────

struct Transition {
    from: StyleSnapshot,
    to: StyleSnapshot,
    elapsed: Duration,
    duration: Duration,
    curve: EaseCurve,
}

/// Runtime art direction controller.
pub struct ArtDirection {
    current_axes: StyleAxes,
    current_palette: StylePalette,
    active_transition: Option<Transition>,
    blend_sources: Vec<(Arc<StyleSnapshot>, f32)>,
    pub named_style_index: usize,
    pub quality_budget: QualityBudget,
}

impl Default for ArtDirection {
    fn default() -> Self {
        Self::new()
    }
}

impl ArtDirection {
    pub fn new() -> Self {
        Self {
            current_axes: StyleAxes::default(),
            current_palette: StylePalette::zeroed(),
            active_transition: None,
            blend_sources: Vec::new(),
            named_style_index: 0,
            quality_budget: QualityBudget::default(),
        }
    }

    pub fn from_snapshot(snapshot: &StyleSnapshot) -> Self {
        Self {
            current_axes: snapshot.axes,
            current_palette: snapshot.palette,
            active_transition: None,
            blend_sources: Vec::new(),
            named_style_index: 0,
            quality_budget: compute_quality_budget(&snapshot.axes),
        }
    }

    pub fn named_styles() -> &'static [(&'static str, StyleSnapshot)] {
        // Can't use const refs to const items in a static slice in const context,
        // so we clone them each time (they're Copy).
        &[("identity", IDENTITY), ("demon_slayer", DEMON_SLAYER)]
    }

    pub fn find_style(name: &str) -> Option<&'static StyleSnapshot> {
        Self::named_styles()
            .iter()
            .find(|(n, _)| *n == name)
            .map(|(_, s)| s)
    }

    pub fn transition_to(&mut self, target: &StyleSnapshot, duration: Duration, curve: EaseCurve) {
        self.active_transition = Some(Transition {
            from: StyleSnapshot {
                axes: self.current_axes,
                palette: self.current_palette,
            },
            to: target.clone(),
            elapsed: Duration::ZERO,
            duration,
            curve,
        });
    }

    pub fn current_axes(&self) -> &StyleAxes {
        &self.current_axes
    }

    /// Returns true if outline pass should be skipped (outline_presence near zero).
    pub fn should_skip_outline(&self) -> bool {
        self.current_axes.outline_presence < 0.01
    }

    /// Return the current post-processing parameters derived from axes.
    pub fn post_params(&self) -> PostProcessingParams {
        let uniforms = map_axes_to_uniforms(&self.current_axes);
        let mut color_grade = uniforms.color_grade_tint;
        // If we have palette-driven color grading, apply it
        if self.current_axes.color_discipline > 0.01 {
            let cd = self.current_axes.color_discipline;
            color_grade = [
                lerp(1.0, self.current_palette.sky_tint[0] + 0.88, cd * 0.3),
                lerp(1.0, self.current_palette.sky_tint[1] + 0.88, cd * 0.3),
                lerp(1.0, self.current_palette.sky_tint[2] + 0.88, cd * 0.3),
                cd * 0.3,
            ];
        }
        PostProcessingParams {
            bloom_tint: [
                uniforms.bloom_tint[0],
                uniforms.bloom_tint[1],
                uniforms.bloom_tint[2],
            ],
            bloom_softness: uniforms.bloom_tint[3],
            color_grade,
        }
    }

    /// Advance transitions, blend, compute uniforms, upload to GPU.
    pub fn update(
        &mut self,
        dt: Duration,
        queue: &wgpu::Queue,
        uniform_buf: &wgpu::Buffer,
        palette_buf: &wgpu::Buffer,
    ) {
        // Advance active transition
        if let Some(ref mut trans) = self.active_transition {
            trans.elapsed += dt;
            let t = (trans.elapsed.as_secs_f32() / trans.duration.as_secs_f32()).min(1.0);
            let eased = trans.curve.evaluate(t);

            self.current_axes = lerp_axes(&trans.from.axes, &trans.to.axes, eased);
            self.current_palette = lerp_palette(&trans.from.palette, &trans.to.palette, eased);

            if t >= 1.0 {
                self.active_transition = None;
            }
        }

        // Apply blend sources (if any)
        if !self.blend_sources.is_empty() {
            let total_weight: f32 = self.blend_sources.iter().map(|(_, w)| w).sum();
            if total_weight > 0.001 {
                let mut blended_axes = StyleAxes::default();
                let mut blended_palette = StylePalette::zeroed();
                for (snap, weight) in &self.blend_sources {
                    let w = weight / total_weight;
                    blended_axes = lerp_axes(&blended_axes, &snap.axes, w);
                    blended_palette = lerp_palette(&blended_palette, &snap.palette, w);
                }
                self.current_axes = blended_axes;
                self.current_palette = blended_palette;
            }
        }

        // Map axes to GPU uniforms
        let mut uniforms = map_axes_to_uniforms(&self.current_axes);

        // Apply palette-driven colors when color_discipline > 0
        if self.current_axes.color_discipline > 0.01 {
            let cd = self.current_axes.color_discipline;
            // Shadow color shift: target hue for shadow regions (blue-violet)
            uniforms.shadow_color_shift = [
                lerp(0.0, self.current_palette.shadow[0] + 0.08, cd),
                lerp(0.0, self.current_palette.shadow[1] + 0.04, cd),
                lerp(0.0, self.current_palette.shadow[2] + 0.18, cd),
                0.0,
            ];
            // Rim light: warm accent tint
            uniforms.rim_light_color = [
                lerp(1.0, self.current_palette.accent[0], cd * 0.8),
                lerp(1.0, self.current_palette.accent[1], cd * 0.8),
                lerp(1.0, self.current_palette.accent[2], cd * 0.8),
                0.0,
            ];
            // Overall color grade toward palette atmosphere
            uniforms.color_grade_tint = [
                lerp(1.0, self.current_palette.sky_tint[0] + 0.90, cd * 0.35),
                lerp(1.0, self.current_palette.sky_tint[1] + 0.90, cd * 0.35),
                lerp(1.0, self.current_palette.sky_tint[2] + 0.90, cd * 0.35),
                cd * 0.3,
            ];
        }

        // Compute quality budget
        self.quality_budget = compute_quality_budget(&self.current_axes);

        // Upload to GPU
        queue.write_buffer(uniform_buf, 0, bytemuck::bytes_of(&uniforms));
        queue.write_buffer(palette_buf, 0, bytemuck::bytes_of(&self.current_palette));
    }
}

// ──────────────────────── Interpolation Helpers ────────────────────────

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Linearly interpolate between two style snapshots.
pub fn blend_snapshots(a: &StyleSnapshot, b: &StyleSnapshot, t: f32) -> StyleSnapshot {
    StyleSnapshot {
        axes: lerp_axes(&a.axes, &b.axes, t),
        palette: lerp_palette(&a.palette, &b.palette, t),
    }
}

fn lerp_axes(a: &StyleAxes, b: &StyleAxes, t: f32) -> StyleAxes {
    StyleAxes {
        softness: lerp(a.softness, b.softness, t),
        exaggeration: lerp(a.exaggeration, b.exaggeration, t),
        shadow_graphicness: lerp(a.shadow_graphicness, b.shadow_graphicness, t),
        palette_warmth: lerp(a.palette_warmth, b.palette_warmth, t),
        atmospheric_depth: lerp(a.atmospheric_depth, b.atmospheric_depth, t),
        surface_detail: lerp(a.surface_detail, b.surface_detail, t),
        outline_presence: lerp(a.outline_presence, b.outline_presence, t),
        color_discipline: lerp(a.color_discipline, b.color_discipline, t),
    }
}

fn lerp_palette(a: &StylePalette, b: &StylePalette, t: f32) -> StylePalette {
    fn lerp4(a: [f32; 4], b: [f32; 4], t: f32) -> [f32; 4] {
        [
            lerp(a[0], b[0], t),
            lerp(a[1], b[1], t),
            lerp(a[2], b[2], t),
            lerp(a[3], b[3], t),
        ]
    }
    StylePalette {
        shadow: lerp4(a.shadow, b.shadow, t),
        dark: lerp4(a.dark, b.dark, t),
        mid: lerp4(a.mid, b.mid, t),
        light: lerp4(a.light, b.light, t),
        accent: lerp4(a.accent, b.accent, t),
        bark: lerp4(a.bark, b.bark, t),
        foliage: lerp4(a.foliage, b.foliage, t),
        earth: lerp4(a.earth, b.earth, t),
        sky_tint: lerp4(a.sky_tint, b.sky_tint, t),
    }
}

// ──────────────────────── Tests ────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_axes_produce_identity_uniforms() {
        let axes = StyleAxes::default();
        let u = map_axes_to_uniforms(&axes);
        assert_eq!(u.shading_bands, 0.0);
        assert_eq!(u.outline_strength, 0.0);
        assert_eq!(u.rim_light_strength, 0.0);
        assert_eq!(u.specular_flattening, 0.0);
        assert_eq!(u.detail_suppression, 0.0);
        // No NaN
        assert!(!u.shading_bands.is_nan());
        assert!(!u.band_softness.is_nan());
    }

    #[test]
    fn semantic_channel_roundtrip() {
        let vals = [0.0_f32, 0.25, 0.5, 0.75, 1.0];
        for &c in &vals {
            for &e in &vals {
                let packed = pack_semantic_channels(c, e, 0.5, 0.5);
                let (uc, ue, us, ui) = unpack_semantic_channels(packed);
                assert!(
                    (uc - c).abs() < 1.0 / 255.0 + 0.001,
                    "curvature roundtrip failed"
                );
                assert!(
                    (ue - e).abs() < 1.0 / 255.0 + 0.001,
                    "edge roundtrip failed"
                );
                assert!((us - 0.5).abs() < 1.0 / 255.0 + 0.001);
                assert!((ui - 0.5).abs() < 1.0 / 255.0 + 0.001);
            }
        }
    }

    #[test]
    fn ease_curve_endpoints() {
        for curve in [
            EaseCurve::Linear,
            EaseCurve::EaseIn,
            EaseCurve::EaseOut,
            EaseCurve::EaseInOut,
        ] {
            assert!(
                (curve.evaluate(0.0)).abs() < 1e-6,
                "{curve:?} evaluate(0.0) should be 0.0"
            );
            assert!(
                (curve.evaluate(1.0) - 1.0).abs() < 1e-6,
                "{curve:?} evaluate(1.0) should be 1.0"
            );
        }
    }

    #[test]
    fn uniforms_expected_size() {
        // 44 floats = 176 bytes
        let expected = 176;
        assert_eq!(
            std::mem::size_of::<ArtDirectionUniforms>(),
            expected,
            "ArtDirectionUniforms size mismatch"
        );
    }

    #[test]
    fn palette_expected_size() {
        // 9 slots × 4 floats × 4 bytes = 144 bytes
        assert_eq!(
            std::mem::size_of::<StylePalette>(),
            144,
            "StylePalette size mismatch"
        );
    }
}
