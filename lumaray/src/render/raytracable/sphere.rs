use ultraviolet::DVec3;

use crate::render::math::{PI, PI_2};

use super::{BoundedGeometry, Intersection, Ray, RaytracableGeometry, AABB};

pub struct Sphere {
    pub center: DVec3,
    pub radius: f64,
}

impl Sphere {
    pub fn new(center: DVec3, radius: f64) -> Self {
        Self { center, radius }
    }
}

impl RaytracableGeometry for Sphere {
    fn thin_intersection(&self, ray: &Ray, compute_uvw: bool) -> Option<Intersection> {
        let oc = ray.origin - self.center;
        let a = ray.direction.mag_sq();
        let b = 2.0 * ray.direction.dot(oc);
        let c = oc.mag_sq() - self.radius * self.radius;

        let discriminant = b * b - 4.0 * a * c;

        if discriminant < 0.0 {
            return None;
        }

        let t1 = (-b - discriminant.sqrt()) / (2.0 * a);
        let t2 = (-b + discriminant.sqrt()) / (2.0 * a);

        // far point is behind us, no intersection
        if t2 < 0.0 {
            return None;
        }

        // near point is behind us, but far point is in front of us
        if t1 < 0.0 {
            let intersection_point = ray.at(t2);
            let normal = (intersection_point - self.center).normalized();
            let uvw = DVec3::new(0.0, 0.0, 0.0);
            return Some(Intersection {
                ray: *ray,
                t: t2,
                normal,
                uvw,
                thick_intersection: None,
            });
        }

        // both points are in front of us
        let ip1 = ray.at(t1);
        let ip2 = ray.at(t2);

        let normal1 = (ip1 - self.center).normalized();
        let normal2 = (ip2 - self.center).normalized();

        let uvw1 = if compute_uvw {
            self.uvw(ip1)
        } else {
            DVec3::new(0.0, 0.0, 0.0)
        };

        let uvw2 = if compute_uvw {
            self.uvw(ip2)
        } else {
            DVec3::new(0.0, 0.0, 0.0)
        };

        Some(Intersection {
            ray: *ray,
            t: t1,
            normal: normal1,
            uvw: uvw1,
            thick_intersection: Some((t2, normal2, uvw2)),
        })
    }

    /// Computes the UVW coordinates at a point on the sphere,
    /// where U is the azimuthal angle, V is the polar angle in range [0, 1].
    /// W is the distance from the surface, where 0 is on the surface and 1 is at the center.
    fn uvw(&self, point: DVec3) -> DVec3 {
        let p = point - self.center;
        let mag = p.mag();

        let w = 1.0 - mag / self.radius;

        let p = p / mag;
        let phi = p.z.asin();
        let theta = p.y.atan2(p.x);

        let u = 1.0 - (theta + PI) / (2.0 * PI);
        let v = (phi + PI_2) / PI;

        DVec3::new(u, v, w)
    }
}

impl BoundedGeometry for Sphere {
    fn local_bounding_box(&self) -> AABB {
        *AABB::null()
            .contain_point(self.center - DVec3::broadcast(self.radius))
            .contain_point(self.center + DVec3::broadcast(self.radius))
    }

    fn center_point(&self) -> DVec3 {
        self.center
    }
}
