use std::sync::Arc;

use ultraviolet::{DMat4, DVec3};

use super::{mesh::Mesh, BoundedGeometry, Intersection, Ray, RaytracableGeometry, AABB};

pub struct MeshInstance {
    pub mesh: Arc<Mesh>,
    pub transform: DMat4,
    pub inverse_transform: DMat4,
}

impl MeshInstance {
    pub fn new(mesh: Arc<Mesh>, transform: DMat4) -> Self {
        Self {
            mesh,
            transform,
            inverse_transform: transform.inversed(),
        }
    }

    pub fn set_transform(&mut self, transform: &DMat4) {
        self.transform = transform.clone();
        self.inverse_transform = transform.inversed();
    }
}

impl RaytracableGeometry for MeshInstance {
    fn thin_intersection(&self, ray: &Ray, max_t: f64, compute_uvw: bool) -> Option<Intersection> {
        let local_ray = self.ray_to_local_space(ray);

        // TODO: need to preserve the value of max_t after the transform

        let isect = self.mesh.thin_intersection(&local_ray, max_t, compute_uvw);

        if let Some(mut isect) = isect {
            isect.normal = self.direction_to_world_space(isect.normal);
            isect.ray = ray.clone();
            // TODO: need to recompute t if the transform has a scale
            Some(isect)
        } else {
            None
        }
    }

    fn point_to_local_space(&self, point: DVec3) -> DVec3 {
        self.inverse_transform.transform_point3(point)
    }

    fn point_to_world_space(&self, point: DVec3) -> DVec3 {
        self.transform.transform_point3(point)
    }

    fn direction_to_local_space(&self, direction: DVec3) -> DVec3 {
        self.inverse_transform.transform_vec3(direction)
    }

    fn direction_to_world_space(&self, direction: DVec3) -> DVec3 {
        self.transform.transform_vec3(direction)
    }
}

impl BoundedGeometry for MeshInstance {
    fn local_bounding_box(&self) -> AABB {
        self.mesh.local_bounding_box()
    }

    fn world_bounding_box(&self) -> AABB {
        self.local_bounding_box().transformed(&self.transform)
    }

    fn local_center_point(&self) -> DVec3 {
        self.local_bounding_box().center()
    }

    fn world_center_point(&self) -> DVec3 {
        self.transform.transform_point3(self.local_center_point())
    }
}
