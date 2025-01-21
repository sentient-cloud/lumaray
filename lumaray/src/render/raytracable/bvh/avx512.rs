pub fn hi() {
    println!("using avx512");
}

use std::arch::x86_64::*;

use ultraviolet::DVec3;

use crate::{
    render::{Ray, AABB},
    utils::constconstrain,
};

use super::TwoVolumeTest;

#[repr(align(16))]
struct AlignedF64x4([f64; 4]);

#[repr(align(64))]
struct AlignedF64x16([f64; 16]);

pub struct TwoRay {
    pub origin: __m512d,
    pub inv_dir: __m512d,
}

impl TwoRay {
    pub fn new(ray: Ray) -> Self {
        unsafe {
            Self {
                origin: _mm512_set_pd(
                    ray.origin.x,
                    ray.origin.x,
                    ray.origin.y,
                    ray.origin.y,
                    ray.origin.z,
                    ray.origin.z,
                    0.0,
                    0.0,
                ),
                inv_dir: _mm512_set_pd(
                    1.0 / ray.direction.x,
                    1.0 / ray.direction.x,
                    1.0 / ray.direction.y,
                    1.0 / ray.direction.y,
                    1.0 / ray.direction.z,
                    1.0 / ray.direction.z,
                    1.0,
                    1.0,
                ),
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct TwoVolume {
    pub min: __m512d,
    pub max: __m512d,
}

impl TwoVolume {
    pub fn zero() -> Self {
        Self::new(AABB::zero(), AABB::zero())
    }

    pub fn new(left: AABB, right: AABB) -> Self {
        unsafe {
            Self {
                min: _mm512_set_pd(
                    right.min.x,
                    left.min.x,
                    right.min.y,
                    left.min.y,
                    right.min.z,
                    left.min.z,
                    f64::INFINITY,
                    f64::INFINITY,
                ),
                max: _mm512_set_pd(
                    right.max.x,
                    left.max.x,
                    right.max.y,
                    left.max.y,
                    right.max.z,
                    left.max.z,
                    -f64::INFINITY,
                    -f64::INFINITY,
                ),
            }
        }
    }

    #[target_feature(enable = "avx512f,avx512vl")]
    pub unsafe fn test(&self, ray: &TwoRay, max_t: f64) -> TwoVolumeTest {
        let mins = _mm512_mul_pd(_mm512_sub_pd(self.min, ray.origin), ray.inv_dir);
        let maxs = _mm512_mul_pd(_mm512_sub_pd(self.max, ray.origin), ray.inv_dir);

        let mins_12_34 = _mm512_extractf64x4_pd::<0>(mins);
        let mins_56_78 = _mm512_extractf64x4_pd::<1>(mins);

        let t0 = _mm256_extractf128_pd::<1>(mins_12_34);
        let t2 = _mm256_extractf128_pd::<0>(mins_56_78);
        let t4 = _mm256_extractf128_pd::<1>(mins_56_78);

        let maxs_12_34 = _mm512_extractf64x4_pd::<0>(maxs);
        let maxs_56_78 = _mm512_extractf64x4_pd::<1>(maxs);

        let t1 = _mm256_extractf128_pd::<1>(maxs_12_34);
        let t3 = _mm256_extractf128_pd::<0>(maxs_56_78);
        let t5 = _mm256_extractf128_pd::<1>(maxs_56_78);

        let tnear = _mm_max_pd(
            _mm_min_pd(t0, t1),
            _mm_max_pd(_mm_min_pd(t2, t3), _mm_min_pd(t4, t5)),
        );
        let tfar = _mm_min_pd(
            _mm_max_pd(t0, t1),
            _mm_min_pd(_mm_max_pd(t2, t3), _mm_max_pd(t4, t5)),
        );

        let tnear_lt_t = _mm_movemask_pd(_mm_cmp_pd::<_CMP_LT_OQ>(tnear, _mm_set1_pd(max_t))) as u8;

        let negmask = !_mm_movemask_pd(tfar) as u8;

        let tnear_lt_tfar = _mm_movemask_pd(_mm_cmp_pd::<_CMP_LT_OQ>(tnear, tfar)) as u8;

        let isect_mask = tnear_lt_t & negmask & tnear_lt_tfar;

        let mut tnearfars: AlignedF64x4 = std::mem::zeroed();
        let tnears = tnearfars.0.as_mut_ptr();
        let tfars = tnears.add(2);

        _mm_store_pd(tnears, tnear);
        _mm_store_pd(tfars, tfar);

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
            let mut mins = AlignedF64x16([0.0; 16]);
            let mut maxs = AlignedF64x16([0.0; 16]);

            unsafe {
                _mm512_store_pd(mins.0.as_mut_ptr(), self.min);
                _mm512_store_pd(maxs.0.as_mut_ptr(), self.max);

                AABB {
                    min: DVec3::new(mins.0[3], mins.0[5], mins.0[7]),
                    max: DVec3::new(maxs.0[3], maxs.0[5], maxs.0[7]),
                }
            }
        } else if I == 1 {
            let mut mins = AlignedF64x16([0.0; 16]);
            let mut maxs = AlignedF64x16([0.0; 16]);

            unsafe {
                _mm512_store_pd(mins.0.as_mut_ptr(), self.min);
                _mm512_store_pd(maxs.0.as_mut_ptr(), self.max);

                AABB {
                    min: DVec3::new(mins.0[2], mins.0[4], mins.0[6]),
                    max: DVec3::new(maxs.0[2], maxs.0[4], maxs.0[6]),
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
