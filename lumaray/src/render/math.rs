use ultraviolet::{DMat3, DMat4, DVec3};

pub const EPS: f64 = 1e-6;

pub const PI: f64 = std::f64::consts::PI;
pub const PI_2: f64 = std::f64::consts::PI * 2.0;
pub const PI_4: f64 = std::f64::consts::PI * 4.0;

pub const IOR_VACUUM: f64 = 1.0;
pub const IOR_HELIUM: f64 = 1.000036;
pub const IOR_AIR: f64 = 1.00029;
pub const IOR_CARBON_DIOXIDE: f64 = 1.00045;

pub const IOR_WATER: f64 = 1.333;
pub const IOR_OIL: f64 = 1.47;

pub const IOR_WATER_ICE: f64 = 1.31;
pub const IOR_PLASTIC: f64 = 1.46;
pub const IOR_PLEXIGLASS: f64 = 1.49;
pub const IOR_GLASS: f64 = 1.5;
pub const IOR_QUARTZ: f64 = 1.544;
pub const IOR_EMERALD: f64 = 1.57;
pub const IOR_RUBY: f64 = 1.77;
pub const IOR_SAPPHIRE: f64 = 1.77;
pub const IOR_DIAMOND: f64 = 2.42;

pub fn degrees_to_radians(degrees: f64) -> f64 {
    degrees * PI / 180.0
}

pub fn radians_to_degrees(radians: f64) -> f64 {
    radians * 180.0 / PI
}

#[derive(Debug, Copy, Clone)]
pub struct Ray {
    pub origin: DVec3,
    pub direction: DVec3,
}

unsafe impl Send for Ray {}
unsafe impl Sync for Ray {}

impl Ray {
    pub fn new(origin: DVec3, direction: DVec3) -> Self {
        Self { origin, direction }
    }

    pub fn at(&self, t: f64) -> DVec3 {
        self.origin + self.direction * t
    }

    pub fn from_two_points(a: DVec3, b: DVec3) -> (Self, f64) {
        let direction = b - a;
        let length = direction.mag();
        let direction = direction / length;
        (
            Self {
                origin: a,
                direction,
            },
            length,
        )
    }

    /// Adjusts the origin of the ray by a small amount in the direction of the ray.
    ///
    /// This is useful after an intersection to prevent intersecting with the same thing again.
    pub fn eps_adjust(&self) -> Self {
        Self {
            origin: self.origin + self.direction * EPS,
            direction: self.direction,
        }
    }

    /// Reflects the ray around a normal, given a point of intersection.
    pub fn reflect(&self, point: DVec3, normal: DVec3) -> Self {
        Self {
            origin: point,
            direction: self.direction - 2.0 * self.direction.dot(normal) * normal,
        }
    }

    /// Refracts the ray through a medium, given a point of intersection.
    ///
    /// `medium_a` is the IOR of the medium the ray is currently in.
    ///
    /// `medium_b` is the IOR of the medium the ray is entering.
    pub fn refract(&self, point: DVec3, normal: DVec3, medium_a: f64, medium_b: f64) -> Self {
        let eta = medium_a / medium_b;
        let cos_theta = (-self.direction).dot(normal);
        let r_out_perp = eta * (self.direction + cos_theta * normal);
        let r_out_parallel = -((1.0 - r_out_perp.mag_sq()).abs().sqrt()) * normal;
        Self {
            origin: point,
            direction: r_out_perp + r_out_parallel,
        }
    }

    /// Refracts the ray through a medium, given a point of intersection.
    ///
    /// This function uses Schlick's approximation to determine whether to reflect or refract.
    ///
    /// Specify a random number between 0.0 and 1.0 as `random_arg`.
    ///
    /// `medium_a` is the IOR of the medium the ray is currently in.
    ///
    /// `medium_b` is the IOR of the medium the ray is entering.
    pub fn refract_with_schlick(
        &self,
        point: DVec3,
        normal: DVec3,
        medium_a: f64,
        medium_b: f64,
        random_arg: f64,
    ) -> Self {
        let eta = medium_a / medium_b;
        let cos_theta = (-self.direction).dot(normal);
        let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();
        let schlick = {
            let r0 = ((medium_a - medium_b) / (medium_a + medium_b)).powi(2);
            r0 + (1.0 - r0) * (1.0 - cos_theta).powi(5)
        };
        if eta * sin_theta > 1.0 || random_arg < schlick {
            self.reflect(point, normal)
        } else {
            self.refract(point, normal, medium_a, medium_b)
        }
    }

    pub fn intersect_triangle(&self, v0: DVec3, v1: DVec3, v2: DVec3) -> Option<f64> {
        let e1 = v1 - v0;
        let e2 = v2 - v0;
        let h = self.direction.cross(e2);
        let a = e1.dot(h);

        if a > -EPS && a < EPS {
            return None;
        }

        let f = 1.0 / a;
        let s = self.origin - v0;
        let u = f * s.dot(h);

        if u < 0.0 || u > 1.0 {
            return None;
        }

        let q = s.cross(e1);
        let v = f * self.direction.dot(q);

        if v < 0.0 || u + v > 1.0 {
            return None;
        }

        let t = f * e2.dot(q);

        if t > EPS {
            Some(t)
        } else {
            None
        }
    }

    pub fn intersect_aabb(&self, aabb: &AABB) -> Option<(f64, f64)> {
        let rd = DVec3::one() / self.direction;

        let (t1, t3, t5) = ((aabb.min - self.origin) * rd).into();
        let (t2, t4, t6) = ((aabb.max - self.origin) * rd).into();

        let t7 = t1.min(t2).max(t3.min(t4)).max(t5.min(t6));
        let t8 = t1.max(t2).min(t3.max(t4)).min(t5.max(t6));

        if t8 < 0.0 || t7 > t8 {
            None
        } else {
            Some((t7, t8))
        }
    }
}

pub fn decompose_mat4(mat: DMat4) -> (DVec3, DMat3, DVec3) {
    let translation = DVec3::new(mat.cols[3].x, mat.cols[3].y, mat.cols[3].z);
    let rotation = mat.extract_rotation().normalized().into_matrix();
    let scale = DVec3::new(
        DVec3::from(mat.cols[0]).mag(),
        DVec3::from(mat.cols[1]).mag(),
        DVec3::from(mat.cols[2]).mag(),
    );

    (translation, rotation, scale)
}

#[derive(Debug, Clone, Copy)]
pub struct AABB {
    pub min: DVec3,
    pub max: DVec3,
}

impl AABB {
    pub fn new(min: DVec3, max: DVec3) -> Self {
        Self { min, max }
    }

    /// Creates a new AABB with all zeros.
    pub fn zero() -> Self {
        Self {
            min: DVec3::zero(),
            max: DVec3::zero(),
        }
    }

    /// Creates a zero volume AABB at a point.
    pub fn at_point(point: DVec3) -> Self {
        Self {
            min: point,
            max: point,
        }
    }

    /// Creates a new AABB at a point with a half-size.
    ///
    /// All components of `half_size` should be positive.
    pub fn at_point_with_half_size(point: DVec3, half_size: DVec3) -> Self {
        Self {
            min: point - half_size,
            max: point + half_size,
        }
    }

    /// Creates a new "null" AABB, where the minimum is positive infinity and the maximum is negative infinity.
    ///
    /// This is useful for initializing an AABB that will only be expanded by other
    /// AABBs or points, since min or max with any point will be that point.
    pub fn null() -> Self {
        Self {
            min: DVec3::broadcast(f64::INFINITY),
            max: DVec3::broadcast(f64::NEG_INFINITY),
        }
    }

    /// Transforms the AABB by a matrix, returning a new AABB.
    pub fn transformed(&self, mat: &DMat4) -> Self {
        let translation = DVec3::new(mat.cols[3].x, mat.cols[3].y, mat.cols[3].z);
        let mut bbox = Self::at_point(translation);

        for i in 0..3 {
            for j in 0..3 {
                let a = mat.cols[i][j] * self.min[j];
                let b = mat.cols[i][j] * self.max[j];
                bbox.min[j] += a.min(b);
                bbox.max[j] += a.max(b);
            }
        }

        bbox
    }

    /// Transforms itself by a matrix.
    pub fn transform(&mut self, mat: &DMat4) -> &mut Self {
        *self = self.transformed(mat);
        self
    }

    /// Returns the center of the AABB.
    pub fn center(&self) -> DVec3 {
        (self.min + self.max) / 2.0
    }

    /// Returns a new AABB containing a point.
    pub fn containing_point(&self, point: DVec3) -> Self {
        Self {
            min: self.min.min_by_component(point),
            max: self.max.max_by_component(point),
        }
    }

    /// Returns a new AABB containing another AABB.
    pub fn containing_aabb(&self, other: &AABB) -> Self {
        self.containing_point(other.min).containing_point(other.max)
    }

    /// Contains the point in the AABB.
    pub fn contain_point(&mut self, point: DVec3) -> &mut Self {
        self.min = self.min.min_by_component(point);
        self.max = self.max.max_by_component(point);
        self
    }

    /// Contains the AABB in the AABB.
    pub fn contain_aabb(&mut self, other: &AABB) -> &mut Self {
        self.contain_point(other.min).contain_point(other.max)
    }
}
