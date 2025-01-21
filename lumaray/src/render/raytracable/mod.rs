use ultraviolet::DVec3;

pub mod bvh;
pub mod mesh;
pub mod sphere;

#[allow(unused_imports)]
pub(crate) use sphere::Sphere;

use super::{math::EPS, Ray, AABB};

#[derive(Debug, Copy, Clone)]
pub struct Intersection {
    /// The ray that was intersected.
    pub ray: Ray,

    /// Distance to the intersection point.
    pub t: f64,

    /// The normal at the intersection point.
    pub normal: DVec3,

    /// The UVW texture coordinates at the intersection point.
    pub uvw: DVec3,

    /// The 2nd intersection point, if applicable.
    ///
    /// For objects such as spheres, you can immediately calculate the 2nd intersection point,
    /// and then you don't waste time calculating everything again.
    pub thick_intersection: Option<(f64, DVec3, DVec3)>,

    /// Number of nodes intersected, if applicable.
    pub nodes_intersected: usize,

    /// Number of primitives intersected, if applicable.
    pub primitives_intersected: usize,
}

impl Intersection {
    pub fn intersection_point(&self) -> DVec3 {
        self.ray.at(self.t)
    }

    pub fn is_thick(&self) -> bool {
        self.thick_intersection.is_some()
    }

    pub fn thick_intersection(&self) -> Self {
        if self.thick_intersection.is_none() {
            panic!("Intersection is not thick");
        }

        let (t, normal, uvw) = self.thick_intersection.unwrap();
        Intersection {
            ray: self.ray,
            t,
            normal,
            uvw,
            thick_intersection: None,
            nodes_intersected: self.nodes_intersected,
            primitives_intersected: self.primitives_intersected,
        }
    }
}

pub trait RaytracableGeometry {
    /// Checks for a thin intersection with the object.
    fn thin_intersection(&self, ray: &Ray, max_t: f64, compute_uvw: bool) -> Option<Intersection>;

    /// Checks for a "thick" intersection with the object.
    ///
    /// By default this is implemented by nudging the ray slightly and calling `thin_intersection`,
    /// however, if first_isect already contains the 2nd intersection point, it is just copied and
    /// returned.
    fn thicc_intersection(
        &self,
        first_isect: &Intersection,
        max_t: f64,
        compute_uvw: bool,
    ) -> Option<Intersection> {
        if let Some((t, normal, uvw)) = first_isect.thick_intersection {
            return Some(Intersection {
                ray: first_isect.ray,
                t,
                normal,
                uvw: if compute_uvw {
                    uvw
                } else {
                    DVec3::new(0.0, 0.0, 0.0)
                },
                thick_intersection: None,
                nodes_intersected: first_isect.nodes_intersected,
                primitives_intersected: first_isect.primitives_intersected,
            });
        }

        self.thin_intersection(
            &Ray::new(first_isect.ray.at(first_isect.t), first_isect.ray.direction).eps_adjust(),
            max_t - first_isect.t - EPS,
            compute_uvw,
        )
    }

    /// Computes the UVW coordinates at a point on the object.
    ///
    /// Returns (0, 0, 0) by default.
    fn uvw(&self, _point: DVec3) -> DVec3 {
        DVec3::zero()
    }

    /// Translates a point to the objects local space.
    ///
    /// By default this simply returns the point.
    fn point_to_local_space(&self, point: DVec3) -> DVec3 {
        point
    }

    /// Translates a point to the objects world space.
    ///
    /// By default this simply returns the point.
    fn point_to_world_space(&self, point: DVec3) -> DVec3 {
        point
    }

    /// Translates a direction to the objects local space.
    ///
    /// By default this simply returns the direction.
    fn direction_to_local_space(&self, direction: DVec3) -> DVec3 {
        direction
    }

    /// Translates a direction to the objects world space.
    ///
    /// By default this simply returns the direction.
    fn direction_to_world_space(&self, direction: DVec3) -> DVec3 {
        direction
    }
}

pub trait BoundedGeometry {
    /// Returns the bounding box of the object in local space.
    fn local_bounding_box(&self) -> AABB;

    /// Returns the bounding box of the object in world space.
    ///
    /// By default this simply returns the same as `local_bounding_box`.
    fn world_bounding_box(&self) -> AABB {
        self.local_bounding_box()
    }

    /// Returns the center of the object. This does not have to be the true geometric center
    /// and defaults to the center of the bounding box.
    fn center_point(&self) -> DVec3 {
        self.local_bounding_box().center()
    }
}
