pub mod bounds;

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
    pub fog_sky_mix: f32,
    pub sky_zenith: Vec3,
    pub sky_horizon: Vec3,
    pub sky_strength: f32,
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
    pub tonemap_strength: f32,
    pub contact_shadow_strength: f32,
    pub wind: WindSettings,
}

/// Wind settings for foliage animation.
#[derive(Clone, Copy, Debug)]
pub struct WindSettings {
    pub direction: glam::Vec2,
    pub speed: f32,
    pub frozen: bool,
}

impl Default for WindSettings {
    fn default() -> Self {
        Self {
            direction: glam::Vec2::new(1.0, 0.3),
            speed: 1.0,
            frozen: false,
        }
    }
}

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
}

impl LightingUniforms {
    pub fn from_camera(
        camera: &CameraState,
        settings: &SceneSettings,
        light_vp: Mat4,
        elapsed_secs: f32,
        screen_width: u32,
        screen_height: u32,
    ) -> Self {
        let vp = camera.view_projection_matrix();
        let inv_vp = vp.inverse();
        let cam_pos = camera.position();

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
                0.0,
            ],
            light_vp: light_vp.to_cols_array_2d(),
            ambient_up: [
                settings.ambient_up.x,
                settings.ambient_up.y,
                settings.ambient_up.z,
                0.0,
            ],
            ambient_down: [
                settings.ambient_down.x,
                settings.ambient_down.y,
                settings.ambient_down.z,
                0.0,
            ],
            ambient_right: [
                settings.ambient_right.x,
                settings.ambient_right.y,
                settings.ambient_right.z,
                0.0,
            ],
            ambient_left: [
                settings.ambient_left.x,
                settings.ambient_left.y,
                settings.ambient_left.z,
                0.0,
            ],
            ambient_front: [
                settings.ambient_front.x,
                settings.ambient_front.y,
                settings.ambient_front.z,
                0.0,
            ],
            ambient_back: [
                settings.ambient_back.x,
                settings.ambient_back.y,
                settings.ambient_back.z,
                0.0,
            ],
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
                0.0,
            ],
            time_params: [
                elapsed_secs,
                if settings.wind.frozen { 0.0 } else { elapsed_secs },
                0.0,
                0.0,
            ],
            screen_size: [screen_width as f32, screen_height as f32, 0.0, 0.0],
            wind_params: [
                settings.wind.direction.x,
                settings.wind.direction.y,
                settings.wind.speed,
                if settings.wind.frozen { 1.0 } else { 0.0 },
            ],
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
    pub uv: [f32; 2],
    pub ao: f32,
}

impl Vertex {
    pub fn layout() -> wgpu::VertexBufferLayout<'static> {
        const ATTRIBUTES: [wgpu::VertexAttribute; 5] = wgpu::vertex_attr_array![
            0 => Float32x3,
            1 => Float32x3,
            2 => Uint32,
            3 => Float32x2,
            4 => Float32
        ];
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &ATTRIBUTES,
        }
    }
}

pub const MATERIAL_TRUNK: u32 = 0;
pub const MATERIAL_FOLIAGE: u32 = 1;
pub const MATERIAL_GROUND: u32 = 2;

/// Shadow map helpers.
pub mod shadow {
    use glam::{Mat4, Vec3};

    pub const SHADOW_MAP_SIZE: u32 = 2048;

    pub struct ShadowMap {
        pub texture: wgpu::Texture,
        pub view: wgpu::TextureView,
        pub sampler: wgpu::Sampler,
    }

    impl ShadowMap {
        pub fn new(device: &wgpu::Device) -> Self {
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("prism-shadow-map"),
                size: wgpu::Extent3d {
                    width: SHADOW_MAP_SIZE,
                    height: SHADOW_MAP_SIZE,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth32Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
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
                sampler,
            }
        }
    }

    pub fn shadow_half_extent(focus_radius: f32) -> f32 {
        focus_radius * 1.05
    }

    pub fn shadow_world_width(focus_radius: f32) -> f32 {
        shadow_half_extent(focus_radius) * 2.0
    }

    pub fn compute_light_vp(
        sun_dir: Vec3,
        shadow_center: Vec3,
        focus_radius: f32,
        shadow_depth: f32,
    ) -> Mat4 {
        let light_dir = sun_dir.normalize();
        let light_distance = focus_radius + shadow_depth * 0.35;
        let light_pos = shadow_center + light_dir * light_distance;
        let up = if light_dir.y.abs() > 0.95 {
            Vec3::Z
        } else {
            Vec3::Y
        };
        let view = Mat4::look_at_rh(light_pos, shadow_center, up);
        let half = shadow_half_extent(focus_radius);
        let proj = Mat4::orthographic_rh(-half, half, -half, half, 0.1, shadow_depth);
        proj * view
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
