pub mod bounds;
pub mod sky_probe;

use bytemuck::{Pod, Zeroable};
use glam::{Affine3A, Mat4, Vec3};

use crate::camera::CameraState;
use crate::material::MaterialId;
use crate::scene::bounds::Aabb;

// ──────────────────────── Scene Graph ────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SceneHandle(pub u32);

pub enum GeometryDef {
    Sdf(SdfTree),
    Parametric(ParametricDef),
    Instanced {
        base: Box<GeometryDef>,
        transforms: Vec<Affine3A>,
    },
}

#[derive(Clone, Debug)]
pub struct ParametricDef {
    pub kind: ParametricKind,
}

#[derive(Clone, Debug)]
pub enum ParametricKind {
    LeafCard,
}

/// SDF tree — recursive boolean CSG of SDF primitives.
#[derive(Clone, Debug)]
pub enum SdfTree {
    Primitive(SdfPrimitive),
    Union(Box<SdfTree>, Box<SdfTree>),
    SmoothUnion {
        a: Box<SdfTree>,
        b: Box<SdfTree>,
        radius: f32,
    },
    Intersection(Box<SdfTree>, Box<SdfTree>),
    Difference(Box<SdfTree>, Box<SdfTree>),
}

#[derive(Clone, Debug)]
pub enum SdfPrimitive {
    Sphere {
        center: Vec3,
        radius: f32,
    },
    TaperedCapsule {
        a: Vec3,
        b: Vec3,
        radius_a: f32,
        radius_b: f32,
    },
    Box {
        center: Vec3,
        half_extents: Vec3,
    },
    Plane {
        normal: Vec3,
        offset: f32,
    },
}

pub struct SceneNode {
    pub handle: SceneHandle,
    pub geometry: GeometryDef,
    pub material: MaterialId,
    pub world_bounds: Aabb,
    pub lipschitz_constant: f32,
}

pub struct SceneGraph {
    nodes: Vec<SceneNode>,
    next_handle: u32,
}

impl SceneGraph {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            next_handle: 0,
        }
    }

    pub fn add(&mut self, geometry: GeometryDef, material: MaterialId) -> SceneHandle {
        let handle = SceneHandle(self.next_handle);
        self.next_handle += 1;

        let world_bounds = compute_geometry_bounds(&geometry);
        let lipschitz_constant = compute_lipschitz(&geometry);

        self.nodes.push(SceneNode {
            handle,
            geometry,
            material,
            world_bounds,
            lipschitz_constant,
        });
        handle
    }

    pub fn nodes(&self) -> &[SceneNode] {
        &self.nodes
    }

    pub fn node(&self, handle: SceneHandle) -> Option<&SceneNode> {
        self.nodes.iter().find(|n| n.handle == handle)
    }
}

impl Default for SceneGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl SdfTree {
    pub fn bounds(&self) -> bounds::Aabb {
        compute_sdf_bounds(self)
    }
}

// ──────────────────────── Bounds Computation ────────────────────────

fn compute_geometry_bounds(geom: &GeometryDef) -> Aabb {
    match geom {
        GeometryDef::Sdf(tree) => compute_sdf_bounds(tree),
        GeometryDef::Parametric(_) => Aabb::from_center_half_extents(Vec3::ZERO, Vec3::splat(1.0)),
        GeometryDef::Instanced { base, transforms } => {
            let base_bounds = compute_geometry_bounds(base);
            let mut result = Aabb::empty();
            for xform in transforms {
                result = result.union(&crate::runtime_scene::transform_aabb(&base_bounds, *xform));
            }
            result
        }
    }
}

fn compute_sdf_bounds(tree: &SdfTree) -> Aabb {
    match tree {
        SdfTree::Primitive(prim) => compute_primitive_bounds(prim),
        SdfTree::Union(a, b) | SdfTree::SmoothUnion { a, b, .. } => {
            let ba = compute_sdf_bounds(a);
            let bb = compute_sdf_bounds(b);
            ba.union(&bb)
        }
        SdfTree::Intersection(a, _b) => {
            // Conservative: use bounds of a (intersection can only shrink)
            compute_sdf_bounds(a)
        }
        SdfTree::Difference(a, _b) => {
            // Conservative: use bounds of a (difference can only shrink)
            compute_sdf_bounds(a)
        }
    }
}

fn compute_primitive_bounds(prim: &SdfPrimitive) -> Aabb {
    match prim {
        SdfPrimitive::Sphere { center, radius } => {
            Aabb::from_center_half_extents(*center, Vec3::splat(*radius))
        }
        SdfPrimitive::TaperedCapsule {
            a,
            b,
            radius_a,
            radius_b,
        } => {
            let max_r = radius_a.max(*radius_b);
            let mut aabb = Aabb::empty();
            aabb.expand_sphere(*a, max_r);
            aabb.expand_sphere(*b, max_r);
            aabb
        }
        SdfPrimitive::Box {
            center,
            half_extents,
        } => Aabb::from_center_half_extents(*center, *half_extents),
        SdfPrimitive::Plane { .. } => {
            // Planes are infinite — use a large bounding box
            Aabb::from_center_half_extents(Vec3::ZERO, Vec3::splat(1000.0))
        }
    }
}

/// Lipschitz constant: 1.0 for standard SDF primitives, composed analytically.
fn compute_lipschitz(geom: &GeometryDef) -> f32 {
    match geom {
        GeometryDef::Sdf(tree) => compute_sdf_lipschitz(tree),
        GeometryDef::Parametric(_) => 1.0,
        GeometryDef::Instanced { base, .. } => compute_lipschitz(base),
    }
}

fn compute_sdf_lipschitz(tree: &SdfTree) -> f32 {
    match tree {
        SdfTree::Primitive(_) => 1.0,
        SdfTree::Union(a, b) | SdfTree::Intersection(a, b) | SdfTree::Difference(a, b) => {
            compute_sdf_lipschitz(a).max(compute_sdf_lipschitz(b))
        }
        SdfTree::SmoothUnion { a, b, .. } => compute_sdf_lipschitz(a).max(compute_sdf_lipschitz(b)),
    }
}

// ──────────────────────── Lighting / Scene Uniforms ────────────────────────

/// Cloud rendering resolution tier.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CloudResolution {
    /// 1/4 screen resolution (default for gameplay presets)
    Quarter,
    /// 1/2 screen resolution (beauty presets)
    Half,
}

/// Cloud volume profile — one density function, different parameters per preset.
#[derive(Clone, Debug)]
pub struct CloudProfile {
    pub coverage: f32,
    pub density_scale: f32,
    pub cloud_base_km: f32,
    pub cloud_top_km: f32,
    pub detail_erosion: f32,
    pub wind_speed: f32,
    pub march_steps: u32,
    pub light_steps: u32,
    pub temporal_blend: f32,
    pub resolution: CloudResolution,
}

impl Default for CloudProfile {
    fn default() -> Self {
        Self {
            coverage: 0.35,
            density_scale: 16.0,
            cloud_base_km: 1.8,
            cloud_top_km: 3.2,
            detail_erosion: 0.6,
            wind_speed: 1.0,
            march_steps: 48,
            light_steps: 6,
            temporal_blend: 0.88,
            resolution: CloudResolution::Quarter,
        }
    }
}

/// Area light types for LTC-based evaluation.
#[derive(Clone, Debug)]
pub enum AreaLight {
    /// Spherical area light.
    Sphere {
        position: Vec3,
        radius: f32,
        color: Vec3,
        intensity: f32,
    },
    /// Tube (capsule) area light.
    Tube {
        start: Vec3,
        end: Vec3,
        radius: f32,
        color: Vec3,
        intensity: f32,
    },
}

/// Scene-wide lighting and environment settings.
pub struct SceneSettings {
    pub sun_direction: Vec3,
    pub sun_color: Vec3,
    pub sun_strength: f32,
    pub sun_angular_radius: f32,
    pub fog_density: f32,
    pub fog_height_falloff: f32,
    pub fog_start: f32,
    pub fog_end: f32,
    pub fog_color: Vec3,
    pub fog_sky_mix: f32,  // Legacy: not read by shaders (sky uses LUT)
    pub sky_zenith: Vec3,  // Legacy: only used by CPU sky probe + cloud ambient
    pub sky_horizon: Vec3, // Legacy: only used by CPU cloud ambient
    pub sky_strength: f32, // Legacy: only used by CPU sky probe
    pub rayleigh_strength: f32,
    pub mie_strength: f32,
    pub mie_anisotropy: f32,
    pub horizon_haze: f32,
    pub shaft_intensity: f32,
    pub shaft_decay: f32,
    pub ambient_up: Vec3,
    pub ambient_down: Vec3,
    pub ambient_right: Vec3,
    pub ambient_left: Vec3,
    pub ambient_front: Vec3,
    pub ambient_back: Vec3,
    pub exposure: f32,
    pub tonemap_strength: f32, // Legacy: not read by shaders (ACES is fixed)
    pub contact_shadow_strength: f32,
    pub ambient_intensity: f32,
    pub gi_intensity: f32,
    pub cloud_coverage: f32,
    pub cloud_profile: CloudProfile,
    pub wind: WindSettings,
    pub fog_volume_density: f32,
    pub fog_volume_albedo: Vec3,
    pub fog_volume_anisotropy: f32,
    pub focus_distance: f32,
    pub aperture: f32,
    pub dof_enabled: bool,
    pub motion_blur_enabled: bool,
    pub fxaa_enabled: bool,
    pub area_lights: Vec<AreaLight>,
    pub ca_strength: f32,
    pub film_grain_strength: f32,
}

/// Wind settings for foliage animation.
#[derive(Clone, Copy, Debug)]
pub struct WindSettings {
    pub direction: glam::Vec2,
    pub mean_speed: f32,
    pub gust_strength: f32,
    pub gust_frequency: f32,
    pub turbulence: f32,
    pub frozen: bool,
}

impl Default for WindSettings {
    fn default() -> Self {
        Self {
            direction: glam::Vec2::new(1.0, 0.3),
            mean_speed: 1.0,
            gust_strength: 0.35,
            gust_frequency: 0.28,
            turbulence: 0.65,
            frozen: false,
        }
    }
}

/// Number of shadow cascades.
pub const NUM_CASCADES: usize = 4;

/// Uniform buffer layout for the lighting pass.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct LightingUniforms {
    pub view_proj: [[f32; 4]; 4],
    pub inv_view_proj: [[f32; 4]; 4],
    pub camera_position: [f32; 4],
    pub sun_direction: [f32; 4],
    pub sun_color: [f32; 4],
    pub fog_color: [f32; 4],
    pub sky_zenith: [f32; 4],
    pub sky_horizon: [f32; 4],
    pub fog_params: [f32; 4],
    pub lighting_params: [f32; 4],
    pub light_vp: [[f32; 4]; 4],
    pub ambient_up: [f32; 4],
    pub ambient_down: [f32; 4],
    pub ambient_right: [f32; 4],
    pub ambient_left: [f32; 4],
    pub ambient_front: [f32; 4],
    pub ambient_back: [f32; 4],
    pub atmosphere_params: [f32; 4],
    pub shaft_params: [f32; 4],
    pub time_params: [f32; 4],
    pub screen_size: [f32; 4],
    pub wind_params: [f32; 4],
    // CSM: cascade 1–3 light view-projection matrices (cascade 0 = light_vp above)
    pub light_vp_1: [[f32; 4]; 4],
    pub light_vp_2: [[f32; 4]; 4],
    pub light_vp_3: [[f32; 4]; 4],
    pub cascade_splits: [f32; 4],
}

impl LightingUniforms {
    pub fn from_camera(
        camera: &CameraState,
        settings: &SceneSettings,
        cascade_light_vps: &[Mat4; NUM_CASCADES],
        cascade_splits: &[f32; NUM_CASCADES],
        elapsed_secs: f32,
        screen_width: u32,
        screen_height: u32,
        sky_probe_cache: &mut sky_probe::SkyProbeCache,
    ) -> Self {
        let vp = camera.view_projection_matrix();
        let inv_vp = vp.inverse();
        let cam_pos = camera.position();

        let has_manual_ambient = [
            settings.ambient_up,
            settings.ambient_down,
            settings.ambient_right,
            settings.ambient_left,
            settings.ambient_front,
            settings.ambient_back,
        ]
        .into_iter()
        .any(|ambient| ambient.length_squared() > 1e-4);

        let mut probe = *sky_probe_cache.get_or_compute(settings);
        if has_manual_ambient {
            let m = 0.22;
            probe.up = probe.up.lerp(settings.ambient_up, m);
            probe.down = probe.down.lerp(settings.ambient_down, m);
            probe.right = probe.right.lerp(settings.ambient_right, m);
            probe.left = probe.left.lerp(settings.ambient_left, m);
            probe.front = probe.front.lerp(settings.ambient_front, m);
            probe.back = probe.back.lerp(settings.ambient_back, m);
        }

        Self {
            view_proj: vp.to_cols_array_2d(),
            inv_view_proj: inv_vp.to_cols_array_2d(),
            camera_position: [cam_pos.x, cam_pos.y, cam_pos.z, 1.0],
            sun_direction: [
                settings.sun_direction.x,
                settings.sun_direction.y,
                settings.sun_direction.z,
                0.0,
            ],
            sun_color: [
                settings.sun_color.x,
                settings.sun_color.y,
                settings.sun_color.z,
                settings.sun_strength,
            ],
            fog_color: [
                settings.fog_color.x,
                settings.fog_color.y,
                settings.fog_color.z,
                settings.fog_sky_mix,
            ],
            sky_zenith: [
                settings.sky_zenith.x,
                settings.sky_zenith.y,
                settings.sky_zenith.z,
                settings.sky_strength,
            ],
            sky_horizon: [
                settings.sky_horizon.x,
                settings.sky_horizon.y,
                settings.sky_horizon.z,
                1.0,
            ],
            fog_params: [
                settings.fog_density,
                settings.fog_height_falloff,
                settings.fog_start,
                settings.fog_end,
            ],
            lighting_params: [
                settings.exposure,
                settings.tonemap_strength,
                settings.contact_shadow_strength,
                settings.gi_intensity,
            ],
            light_vp: cascade_light_vps[0].to_cols_array_2d(),
            ambient_up: [probe.up.x, probe.up.y, probe.up.z, 0.0],
            ambient_down: [probe.down.x, probe.down.y, probe.down.z, 0.0],
            ambient_right: [probe.right.x, probe.right.y, probe.right.z, 0.0],
            ambient_left: [probe.left.x, probe.left.y, probe.left.z, 0.0],
            ambient_front: [probe.front.x, probe.front.y, probe.front.z, 0.0],
            ambient_back: [probe.back.x, probe.back.y, probe.back.z, 0.0],
            atmosphere_params: [
                settings.sun_angular_radius,
                settings.rayleigh_strength,
                settings.mie_strength,
                settings.mie_anisotropy,
            ],
            shaft_params: [
                settings.horizon_haze,
                settings.shaft_intensity,
                settings.shaft_decay,
                settings.cloud_coverage,
            ],
            time_params: [
                elapsed_secs,
                if settings.wind.frozen {
                    0.0
                } else {
                    elapsed_secs
                },
                0.0,
                0.0,
            ],
            screen_size: [screen_width as f32, screen_height as f32, 0.0, 0.0],
            wind_params: [
                settings.wind.direction.x,
                settings.wind.direction.y,
                settings.wind.mean_speed,
                settings.wind.gust_strength,
            ],
            light_vp_1: cascade_light_vps[1].to_cols_array_2d(),
            light_vp_2: cascade_light_vps[2].to_cols_array_2d(),
            light_vp_3: cascade_light_vps[3].to_cols_array_2d(),
            cascade_splits: *cascade_splits,
        }
    }
}

// ──────────────────────── Vertex / Mesh ────────────────────────

/// Vertex format for rasterized geometry.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub material: u32,
    pub feature_id: u32,
    pub uv: [f32; 2],
    pub ao: f32,
    pub semantic_channels: u32,
}

impl Vertex {
    pub fn layout() -> wgpu::VertexBufferLayout<'static> {
        const ATTRIBUTES: [wgpu::VertexAttribute; 7] = wgpu::vertex_attr_array![
            0 => Float32x3,
            1 => Float32x3,
            2 => Uint32,
            3 => Uint32,
            4 => Float32x2,
            5 => Float32,
            6 => Uint32
        ];
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &ATTRIBUTES,
        }
    }
}

/// Transform a vertex's position and normal by an affine transform.
pub fn transform_vertex(vertex: &Vertex, transform: glam::Affine3A) -> Vertex {
    let normal_transform = glam::Mat3::from_cols(
        transform.matrix3.x_axis.into(),
        transform.matrix3.y_axis.into(),
        transform.matrix3.z_axis.into(),
    );
    let mut transformed = *vertex;
    transformed.position = transform
        .transform_point3(Vec3::from_array(vertex.position))
        .to_array();
    transformed.normal = normal_transform
        .mul_vec3(Vec3::from_array(vertex.normal))
        .normalize_or_zero()
        .to_array();
    transformed
}

pub const MATERIAL_TRUNK: u32 = 0;
pub const MATERIAL_FOLIAGE: u32 = 1;
pub const MATERIAL_GROUND: u32 = 2;
pub const MATERIAL_SKIN: u32 = 3;
pub const MATERIAL_EMISSIVE: u32 = 4;

/// Pack alpha and emissive into `semantic_channels` for foliage/emissive materials.
///
/// Bit layout (foliage + emissive materials):
/// - Bits 0–7: alpha (0–255 maps to 0.0–1.0)
/// - Bits 8–15: emissive intensity (0–255 maps to 0.0–1.0)
/// - Bits 16–31: reserved
///
/// NOTE: Trunk geometry uses a *different* packing scheme via
/// `art_direction::pack_semantic_channels` (curvature, edge_sharpness, etc.).
/// The shader dispatches on `material` to decide which interpretation to use.
pub fn pack_foliage_channels(alpha: f32, emissive: f32) -> u32 {
    let a = (alpha.clamp(0.0, 1.0) * 255.0).round() as u32;
    let e = (emissive.clamp(0.0, 1.0) * 255.0).round() as u32;
    a | (e << 8)
}

/// Unpack alpha from a `semantic_channels` u32 (foliage packing). Returns 0.0–1.0.
pub fn unpack_alpha(semantic_channels: u32) -> f32 {
    (semantic_channels & 0xFF) as f32 / 255.0
}

/// Unpack emissive intensity from a `semantic_channels` u32 (foliage packing). Returns 0.0–1.0.
pub fn unpack_emissive(semantic_channels: u32) -> f32 {
    ((semantic_channels >> 8) & 0xFF) as f32 / 255.0
}

/// Shadow map helpers.
pub mod shadow {
    use super::NUM_CASCADES;
    use glam::{Mat4, Vec3, Vec4};

    pub const SHADOW_MAP_SIZE: u32 = 2048;

    pub struct ShadowMap {
        pub texture: wgpu::Texture,
        /// Combined view of all 4 cascade layers (for shader sampling as texture_depth_2d_array).
        pub view: wgpu::TextureView,
        /// Per-cascade views for render attachments.
        pub cascade_views: [wgpu::TextureView; NUM_CASCADES],
        pub sampler: wgpu::Sampler,
    }

    impl ShadowMap {
        pub fn new(device: &wgpu::Device) -> Self {
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("prism-shadow-map"),
                size: wgpu::Extent3d {
                    width: SHADOW_MAP_SIZE,
                    height: SHADOW_MAP_SIZE,
                    depth_or_array_layers: NUM_CASCADES as u32,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth32Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            // Combined view for sampling all layers
            let view = texture.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                ..Default::default()
            });
            // Per-cascade views for render attachments
            let cascade_views = std::array::from_fn(|i| {
                texture.create_view(&wgpu::TextureViewDescriptor {
                    label: Some(&format!("prism-shadow-cascade-{i}")),
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    base_array_layer: i as u32,
                    array_layer_count: Some(1),
                    ..Default::default()
                })
            });
            let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("prism-shadow-sampler"),
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                compare: Some(wgpu::CompareFunction::LessEqual),
                ..Default::default()
            });
            Self {
                texture,
                view,
                cascade_views,
                sampler,
            }
        }
    }

    /// Compute practical-logarithmic cascade split distances.
    ///
    /// Returns `NUM_CASCADES` far distances for each cascade (view-space Z).
    pub fn compute_cascade_splits(near: f32, far: f32, lambda: f32) -> [f32; NUM_CASCADES] {
        let n = NUM_CASCADES as f32;
        std::array::from_fn(|i| {
            let p = (i + 1) as f32 / n;
            let log_split = near * (far / near).powf(p);
            let lin_split = near + (far - near) * p;
            lambda * log_split + (1.0 - lambda) * lin_split
        })
    }

    /// Compute an orthographic light VP matrix for one cascade, fitted to
    /// the camera sub-frustum corners projected into light space.
    /// Includes texel snapping to prevent shimmer on camera rotation.
    pub fn compute_cascade_light_vp(
        sun_dir: Vec3,
        frustum_corners: &[Vec3; 8],
        shadow_depth: f32,
    ) -> Mat4 {
        let light_dir = sun_dir.normalize();
        let up = if light_dir.y.abs() > 0.95 {
            Vec3::Z
        } else {
            Vec3::Y
        };

        // Use bounding sphere of frustum corners — rotation-invariant, prevents shimmer.
        let center = frustum_corners.iter().copied().sum::<Vec3>() / 8.0;
        let radius = frustum_corners
            .iter()
            .map(|c| (*c - center).length())
            .fold(0.0_f32, f32::max);

        let light_view = Mat4::look_at_rh(center + light_dir * shadow_depth * 0.5, center, up);

        // Use sphere radius for XY extents — doesn't change on rotation
        let half = radius;

        // Find Z range in light space for proper near/far
        let mut min_z = f32::MAX;
        let mut max_z = f32::MIN;
        for &corner in frustum_corners {
            let lv = light_view * Vec4::from((corner, 1.0));
            min_z = min_z.min(lv.z);
            max_z = max_z.max(lv.z);
        }
        min_z -= shadow_depth * 0.5;

        // Snap center to texel grid in light space to prevent sub-texel drift
        let texel_size = (2.0 * half) / SHADOW_MAP_SIZE as f32;
        let lv_center = light_view * Vec4::from((center, 1.0));
        let snapped_x = (lv_center.x / texel_size).floor() * texel_size;
        let snapped_y = (lv_center.y / texel_size).floor() * texel_size;
        let offset_x = snapped_x - lv_center.x;
        let offset_y = snapped_y - lv_center.y;

        let proj = Mat4::orthographic_rh(
            -half + offset_x,
            half + offset_x,
            -half + offset_y,
            half + offset_y,
            -max_z,
            -min_z,
        );
        proj * light_view
    }

    /// Extract the 8 corners of a view sub-frustum between near_split and far_split.
    ///
    /// Computes corners directly from camera geometry to avoid issues with
    /// infinite far plane in the reversed-Z projection matrix.
    pub fn frustum_corners(
        camera: &crate::camera::CameraState,
        near_split: f32,
        far_split: f32,
    ) -> [Vec3; 8] {
        let forward = camera.forward();
        let right = camera.right();
        let up = camera.up();
        let pos = camera.position();
        let half_fov_tan = (camera.fov_y_radians * 0.5).tan();

        let near_h = near_split * half_fov_tan;
        let near_w = near_h * camera.aspect;
        let far_h = far_split * half_fov_tan;
        let far_w = far_h * camera.aspect;

        let near_center = pos + forward * near_split;
        let far_center = pos + forward * far_split;

        [
            // Near plane corners
            near_center - right * near_w - up * near_h,
            near_center + right * near_w - up * near_h,
            near_center - right * near_w + up * near_h,
            near_center + right * near_w + up * near_h,
            // Far plane corners
            far_center - right * far_w - up * far_h,
            far_center + right * far_w - up * far_h,
            far_center - right * far_w + up * far_h,
            far_center + right * far_w + up * far_h,
        ]
    }

    /// Compute all cascade light VP matrices for the current frame.
    pub fn compute_all_cascade_vps(
        camera: &crate::camera::CameraState,
        sun_dir: Vec3,
        shadow_depth: f32,
    ) -> ([Mat4; NUM_CASCADES], [f32; NUM_CASCADES]) {
        let near = camera.near_plane;
        let far = camera.far_plane;
        let splits = compute_cascade_splits(near, far, 0.75);

        let cascade_vps = std::array::from_fn(|i| {
            let split_near = if i == 0 { near } else { splits[i - 1] };
            let split_far = splits[i];
            let corners = frustum_corners(camera, split_near, split_far);
            compute_cascade_light_vp(sun_dir, &corners, shadow_depth)
        });

        // View-space split distances for fragment shader cascade selection.
        // In RH view space, Z is negative for objects in front of camera.
        // A point at distance d along forward has view_z = -d, so |view_z| = d.
        let view_splits = splits;

        (cascade_vps, view_splits)
    }
}

// ──────────────────────── Tests ────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::material::MaterialId;

    #[test]
    fn scene_graph_add_and_retrieve() {
        let mut graph = SceneGraph::new();
        let h = graph.add(
            GeometryDef::Sdf(SdfTree::Primitive(SdfPrimitive::Sphere {
                center: Vec3::ZERO,
                radius: 5.0,
            })),
            MaterialId(0),
        );
        assert_eq!(h, SceneHandle(0));
        assert!(graph.node(h).is_some());
        assert_eq!(graph.nodes().len(), 1);
    }

    #[test]
    fn bounds_computation_sphere() {
        let geom = GeometryDef::Sdf(SdfTree::Primitive(SdfPrimitive::Sphere {
            center: Vec3::new(1.0, 2.0, 3.0),
            radius: 5.0,
        }));
        let bounds = compute_geometry_bounds(&geom);
        assert!((bounds.min.x - (-4.0)).abs() < 1e-5);
        assert!((bounds.max.x - 6.0).abs() < 1e-5);
    }

    #[test]
    fn bounds_computation_smooth_union() {
        let tree = SdfTree::SmoothUnion {
            a: Box::new(SdfTree::Primitive(SdfPrimitive::Sphere {
                center: Vec3::ZERO,
                radius: 1.0,
            })),
            b: Box::new(SdfTree::Primitive(SdfPrimitive::Sphere {
                center: Vec3::new(5.0, 0.0, 0.0),
                radius: 2.0,
            })),
            radius: 0.5,
        };
        let bounds = compute_geometry_bounds(&GeometryDef::Sdf(tree));
        assert!(bounds.min.x <= -1.0);
        assert!(bounds.max.x >= 7.0);
    }

    #[test]
    fn pack_unpack_foliage_channels_roundtrip() {
        let packed = super::pack_foliage_channels(0.75, 0.5);
        let alpha = super::unpack_alpha(packed);
        let emissive = super::unpack_emissive(packed);
        assert!((alpha - 0.75).abs() < 0.005);
        assert!((emissive - 0.5).abs() < 0.005);
    }

    #[test]
    fn pack_foliage_channels_extremes() {
        let zero = super::pack_foliage_channels(0.0, 0.0);
        assert_eq!(super::unpack_alpha(zero), 0.0);
        assert_eq!(super::unpack_emissive(zero), 0.0);

        let full = super::pack_foliage_channels(1.0, 1.0);
        assert!((super::unpack_alpha(full) - 1.0).abs() < 0.005);
        assert!((super::unpack_emissive(full) - 1.0).abs() < 0.005);
    }

    #[test]
    fn lipschitz_is_one_for_standard_primitives() {
        let geom = GeometryDef::Sdf(SdfTree::SmoothUnion {
            a: Box::new(SdfTree::Primitive(SdfPrimitive::Sphere {
                center: Vec3::ZERO,
                radius: 1.0,
            })),
            b: Box::new(SdfTree::Primitive(SdfPrimitive::TaperedCapsule {
                a: Vec3::ZERO,
                b: Vec3::Y * 10.0,
                radius_a: 1.0,
                radius_b: 0.5,
            })),
            radius: 0.5,
        });
        assert!((compute_lipschitz(&geom) - 1.0).abs() < 1e-6);
    }
}
