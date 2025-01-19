pub fn hi() {
    println!("using avx512");
}

use std::arch::x86_64::*;

use crate::render::{Ray, AABB};

use super::TwoVolumeTest;

#[repr(align(16))]
struct AlignedF64x4([f64; 4]);

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
}
