use ultraviolet::{DRotor3, DVec3};

use super::{math::degrees_to_radians, Ray};

pub trait Camera {
    fn position(&self) -> DVec3;
    fn forward(&self) -> DVec3;
    fn right(&self) -> DVec3;
    fn up(&self) -> DVec3;
    fn fov_y(&self) -> f64;
    fn fov_x(&self) -> f64;

    fn get_primary_ray(&self, x: f64, y: f64) -> Ray;

    fn set_position(&mut self, position: DVec3);
    fn set_direction(&mut self, direction: DVec3);
    fn set_look_at(&mut self, look_at: DVec3);
    fn set_fov(&mut self, fov_y: f64);
}

#[derive(Debug)]
pub struct PerspectiveCamera {
    position: DVec3,
    forward: DVec3,
    right: DVec3,
    up: DVec3,

    fov_x: f64, // tan(fov_x in radians / 2)

    film_width: f64,
    film_height: f64,
}

impl PerspectiveCamera {
    pub fn new(film_width: f64, film_height: f64) -> Self {
        let position = DVec3::zero();
        let forward = DVec3::unit_x();
        let right = DVec3::unit_y();
        let up = DVec3::unit_z();

        let fov_x = (degrees_to_radians(90.0) / 2.0).tan();

        Self {
            position,
            forward,
            right,
            up,
            fov_x,
            film_width,
            film_height,
        }
    }
}

impl Camera for PerspectiveCamera {
    fn position(&self) -> DVec3 {
        self.position
    }

    fn forward(&self) -> DVec3 {
        self.forward
    }

    fn right(&self) -> DVec3 {
        self.right
    }

    fn up(&self) -> DVec3 {
        self.up
    }

    fn fov_y(&self) -> f64 {
        self.fov_x * self.film_height / self.film_width
    }

    fn fov_x(&self) -> f64 {
        self.fov_x
    }

    fn get_primary_ray(&self, x: f64, y: f64) -> Ray {
        let xi = 2.0 * x / self.film_width - 1.0;
        let yi = 2.0 * y / self.film_height - 1.0;
        let xi = xi * self.fov_x();
        let yi = yi * self.fov_y();
        Ray::new(
            self.position,
            (self.forward - self.right * xi - self.up * yi).normalized(),
        )
    }

    fn set_position(&mut self, position: DVec3) {
        self.position = position;
    }

    fn set_direction(&mut self, direction: DVec3) {
        self.forward = direction.normalized();
        self.right = DVec3::unit_z().cross(self.forward).normalized();
        self.up = self.forward.cross(self.right);
    }

    fn set_look_at(&mut self, look_at: DVec3) {
        // instead of just stupidly setting the forward vector and recalculating
        // right and up, rotate the camera to look at the new point, so that any
        // prior rotations are preserved

        let new_forward = (look_at - self.position).normalized();
        println!("{:?}", new_forward);

        let rotation = DRotor3::from_rotation_between(self.forward, new_forward).normalized();

        println!("{:?}", rotation);

        rotation.rotate_vec(&mut self.forward);
        rotation.rotate_vec(&mut self.right);
        rotation.rotate_vec(&mut self.up);
    }

    fn set_fov(&mut self, fov_x: f64) {
        self.fov_x = (degrees_to_radians(fov_x) / 2.0).tan();
    }
}
