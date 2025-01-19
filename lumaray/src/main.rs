#![allow(dead_code)]
#![allow(incomplete_features)]
#![feature(stdarch_x86_avx512)]
#![feature(generic_const_exprs)]
#![feature(avx512_target_feature)]

use ultraviolet::DVec3;

#[cfg(all(feature = "no-simd", feature = "avx512"))]
compile_error!("Cannot enable no-simd and avx512 features at the same time");

mod compositor;
mod parsers;
mod render;
mod utils;

fn main() {
    println!("meow :3");

    #[cfg(feature = "avx512")]
    use std::arch::is_x86_feature_detected;

    #[cfg(all(feature = "avx2", not(feature = "no-simd"), not(feature = "avx512")))]
    if !is_x86_feature_detected!("avx2") {
        panic!("Wrong build used. AVX2 is not supported by the CPU. Please recompile with the no-simd feature.");
    }

    #[cfg(feature = "avx512")]
    if !is_x86_feature_detected!("avx512f") {
        panic!("Wrong build used. AVX512 is not supported by the CPU. Please recompile with the no-simd feature.");
    }

    render::raytracable::bvh::hi();

    let timer = utils::Timer::new();

    let iters = 1_000_000_000i64;
    let num_threads = 32;

    println!(
        "running {} iterations each on {} threads",
        iters, num_threads
    );

    let threads = (0..num_threads)
        .map(|_| {
            std::thread::spawn({
                let iters = iters;

                let left = render::math::AABB::at_point_with_half_size(
                    DVec3::new(0.0, 0.0, 0.0),
                    DVec3::new(1.0, 1.0, 1.0),
                );

                let right = render::math::AABB::at_point_with_half_size(
                    DVec3::new(0.0, 0.5, 0.0),
                    DVec3::new(1.0, 1.0, 1.0),
                );

                let two_volume = render::raytracable::bvh::TwoVolume::new(left, right);

                let ray = render::Ray::new(
                    ultraviolet::DVec3::new(0.0, 5.0, 0.0),
                    ultraviolet::DVec3::new(0.0, -1.0, 0.0).normalized(),
                );

                let two_ray = render::raytracable::bvh::TwoRay::new(ray);

                move || {
                    for _ in 0..iters {
                        unsafe {
                            let t = two_volume.test(&two_ray, f64::INFINITY);
                            // println!("{:?}", t);
                            core::hint::black_box(t);
                        }
                    }
                }
            })
        })
        .collect::<Vec<_>>();

    for t in threads {
        t.join().unwrap();
    }

    println!("{} seconds", timer.elapsed());
    println!(
        "{:.2} Mrays/s",
        (num_threads * iters) as f64 / timer.elapsed() / 1_000_000.0
    );

    let freq = 5.8e9 * timer.elapsed();
    println!("{} cycles per iteration", freq / iters as f64);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_implicit_texture() {
        let texture =
            render::Texture::new_implicit(|point| compositor::RGBA::from((point.x, point.y, 0.0)));

        let mut image = compositor::Image::new(512, 512);

        image.for_each_pixel({
            let w = image.width;
            let h = image.height;

            move |x, y, _| {
                let u = x as f64 / w as f64;
                let v = y as f64 / h as f64;
                texture.sample(&ultraviolet::DVec3::new(u, v, 0.0))
            }
        });

        image
            .output_as_png("trash/test_implicit_texture.png")
            .unwrap();
    }
}
