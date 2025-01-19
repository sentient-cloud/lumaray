use crate::render::{Ray, AABB};

use super::TwoVolumeTest;

pub fn hi() {
    println!("no-simd");
}

pub struct TwoRay(pub Ray);

impl TwoRay {
    pub fn new(ray: Ray) -> Self {
        Self(ray)
    }
}

#[derive(Debug, Clone)]
pub struct TwoVolume {
    pub left: AABB,
    pub right: AABB,
}

impl TwoVolume {
    pub fn zero() -> Self {
        Self::new(AABB::zero(), AABB::zero())
    }

    pub fn new(left: AABB, right: AABB) -> Self {
        Self { left, right }
    }

    pub unsafe fn test(&self, ray: &TwoRay, max_t: f64) -> TwoVolumeTest {
        let ta = ray.0.intersect_aabb(&self.left);
        let tb = ray.0.intersect_aabb(&self.right);

        let mut state = 0;
        let mut left_t1 = 0.0;
        let mut left_t2 = 0.0;
        let mut right_t1 = 0.0;
        let mut right_t2 = 0.0;

        if let Some((t1, t2)) = ta {
            if t1 < max_t {
                state |= 0b01;
                left_t1 = t1;
                left_t2 = t2;
            }
        }

        if let Some((t1, t2)) = tb {
            if t1 < max_t {
                state |= 0b10;
                right_t1 = t1;
                right_t2 = t2;
            }
        }

        TwoVolumeTest {
            state,
            left_t1,
            left_t2,
            right_t1,
            right_t2,
        }
    }
}
