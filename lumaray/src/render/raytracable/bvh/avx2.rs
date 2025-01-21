pub fn hi() {
    println!("using avx2");
}

use core::f64;
use std::arch::x86_64::*;

use crate::render::{Ray, AABB};

use super::TwoVolumeTest;

#[repr(align(16))]
struct AlignedF64x2([f64; 2]);

#[repr(align(16))]
struct AlignedF64x4([f64; 4]);

pub struct TwoRay {
    pub origin: [__m128d; 3],
    pub inv_dir: [__m128d; 3],
}

impl TwoRay {
    pub fn new(ray: Ray) -> Self {
        unsafe {
            Self {
                origin: [
                    _mm_set1_pd(ray.origin.x),
                    _mm_set1_pd(ray.origin.y),
                    _mm_set1_pd(ray.origin.z),
                ],
                inv_dir: [
                    _mm_set1_pd(1.0 / ray.direction.x),
                    _mm_set1_pd(1.0 / ray.direction.y),
                    _mm_set1_pd(1.0 / ray.direction.z),
                ],
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct TwoVolume {
    pub min: [__m128d; 3], // a min x, b min x, a min y, b min y, a min z, b min z
    pub max: [__m128d; 3], // a max x, b max x, a max y, b max y, a max z, b max z
}

impl TwoVolume {
    pub fn zero() -> Self {
        Self::new(AABB::zero(), AABB::zero())
    }

    pub fn new(left: AABB, right: AABB) -> Self {
        unsafe {
            Self {
                min: [
                    _mm_set_pd(right.min.x, left.min.x),
                    _mm_set_pd(right.min.y, left.min.y),
                    _mm_set_pd(right.min.z, left.min.z),
                ],
                max: [
                    _mm_set_pd(right.max.x, left.max.x),
                    _mm_set_pd(right.max.y, left.max.y),
                    _mm_set_pd(right.max.z, left.max.z),
                ],
            }
        }
    }

    pub unsafe fn test(&self, ray: &TwoRay, max_t: f64) -> TwoVolumeTest {
        // yoUr cOdE iS uNsAfE
        let t0 = _mm_mul_pd(_mm_sub_pd(self.min[0], ray.origin[0]), ray.inv_dir[0]);
        let t1 = _mm_mul_pd(_mm_sub_pd(self.max[0], ray.origin[0]), ray.inv_dir[0]);
        let t2 = _mm_mul_pd(_mm_sub_pd(self.min[1], ray.origin[1]), ray.inv_dir[1]);
        let t3 = _mm_mul_pd(_mm_sub_pd(self.max[1], ray.origin[1]), ray.inv_dir[1]);
        let t4 = _mm_mul_pd(_mm_sub_pd(self.min[2], ray.origin[2]), ray.inv_dir[2]);
        let t5 = _mm_mul_pd(_mm_sub_pd(self.max[2], ray.origin[2]), ray.inv_dir[2]);

        let tnear = _mm_max_pd(
            _mm_min_pd(t0, t1),
            _mm_max_pd(_mm_min_pd(t2, t3), _mm_min_pd(t4, t5)),
        );
        let tfar = _mm_min_pd(
            _mm_max_pd(t0, t1),
            _mm_min_pd(_mm_max_pd(t2, t3), _mm_max_pd(t4, t5)),
        );

        // tnear < max_t
        let tnear_lt_t = _mm_movemask_pd(_mm_cmp_pd::<_CMP_LT_OQ>(tnear, _mm_set1_pd(max_t))) as u8;

        // tfar > 0.0
        // movemask gets the msb (sign bit) of each float, which is 0 if the
        // float is positive, so get the movemask, and then negate it
        let negmask = !_mm_movemask_pd(tfar) as u8;

        // tnear < tfar
        let tnear_lt_tfar = _mm_movemask_pd(_mm_cmp_pd::<_CMP_LT_OQ>(tnear, tfar)) as u8;

        let isect_mask = tnear_lt_t & negmask & tnear_lt_tfar;

        // copy out the tnear and tfar values
        let mut tnearfars: AlignedF64x4 = std::mem::zeroed();
        let tnears = tnearfars.0.as_mut_ptr();
        let tfars = tnears.add(2);

        _mm_store_pd(tnears, tnear);
        _mm_store_pd(tfars, tfar);

        // no branching to return valid floats or whatever, just let
        // the values be bullshit if theyre supposed to be unused
        TwoVolumeTest {
            state: isect_mask,
            left_t1: *tnears,
            left_t2: *tfars,
            right_t1: *tnears.add(1),
            right_t2: *tfars.add(1),
        }
    }
    /// Extracts the AABB at index `I` from the TwoVolume.
    ///
    /// I = 0 extracts the left AABB, I = 1 extracts the right AABB.
    pub fn extract_aabb<const I: usize>(&self) -> AABB
    where
        [(); constconstrain::is_zero_or_one(I) - 1]:,
    {
        if I == 0 {
            let mut mins = [AlignedF64x2([0.0; 2]); 3];
            let mut maxs = [AlignedF64x2([0.0; 2]); 3];

            unsafe {
                _mm_store_pd(mins[0].0.as_mut_ptr(), self.min[0]);
                _mm_store_pd(mins[1].0.as_mut_ptr(), self.min[1]);
                _mm_store_pd(mins[2].0.as_mut_ptr(), self.min[2]);
                _mm_store_pd(maxs[0].0.as_mut_ptr(), self.max[0]);
                _mm_store_pd(maxs[1].0.as_mut_ptr(), self.max[1]);
                _mm_store_pd(maxs[2].0.as_mut_ptr(), self.max[2]);

                AABB {
                    min: DVec3::new(mins[0].0[1], mins[1].0[1], mins[2].0[1]),
                    max: DVec3::new(maxs[0].0[1], maxs[1].0[1], maxs[2].0[1]),
                }
            }
        } else if I == 1 {
            let mut mins = [AlignedF64x2([0.0; 2]); 3];
            let mut maxs = [AlignedF64x2([0.0; 2]); 3];

            unsafe {
                _mm_store_pd(mins[0].0.as_mut_ptr(), self.min[0]);
                _mm_store_pd(mins[1].0.as_mut_ptr(), self.min[1]);
                _mm_store_pd(mins[2].0.as_mut_ptr(), self.min[2]);
                _mm_store_pd(maxs[0].0.as_mut_ptr(), self.max[0]);
                _mm_store_pd(maxs[1].0.as_mut_ptr(), self.max[1]);
                _mm_store_pd(maxs[2].0.as_mut_ptr(), self.max[2]);

                AABB {
                    min: DVec3::new(mins[0].0[0], mins[1].0[0], mins[2].0[0]),
                    max: DVec3::new(maxs[0].0[0], maxs[1].0[0], maxs[2].0[0]),
                }
            }
        } else {
            unreachable!()
        }
    }
}

impl Into<(AABB, AABB)> for TwoVolume {
    fn into(self) -> (AABB, AABB) {
        (self.extract_aabb::<0>(), self.extract_aabb::<1>())
    }
}
