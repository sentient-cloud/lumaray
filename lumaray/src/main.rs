#![allow(dead_code)]
#![allow(incomplete_features)]
#![feature(stdarch_x86_avx512)]
#![feature(generic_const_exprs)]
#![feature(avx512_target_feature)]

use core::f64;
use std::{fs::File, io::BufReader, time::Instant};

use parsers::stl::STL;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use render::raytracable::{mesh::Mesh, RaytracableGeometry};
use ultraviolet::{DVec3, Vec3};

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

    use ultraviolet::DVec3;

    use crate::render::{raytracable::mesh::SimpleTriangle, Ray};

    let model = "../data/models/Asian_Dragon.stl";
    let mut bufreader = BufReader::new(File::open(model).unwrap());
    let stl = STL::new_from_bufreader(&mut bufreader).unwrap();

    let num_triangles = stl.triangles.len();

    let mesh = Mesh::new(
        stl.triangles
            .iter()
            .map(|tri| tri.vertices)
            .map(|tri| SimpleTriangle { vertices: tri })
            .collect::<Vec<_>>(),
        stl.triangles
            .iter()
            .map(|tri| tri.normal)
            .map(|n| [n; 3])
            .collect::<Vec<_>>(),
        stl.triangles
            .iter()
            .map(|_| [Vec3::zero(); 3])
            .collect::<Vec<_>>(),
    );

    let ray = Ray::new(DVec3::new(300.0, 0.0, 0.0), DVec3::new(-1.0, 0.0, 0.0));

    let iters = 50_000_000usize;
    let threads = 32;

    let start = Instant::now();

    let (nodes_isect, primitives_isect) = (0..(iters * threads))
        .into_par_iter()
        .map(|_| {
            let isect = mesh.thin_intersection(&ray, f64::INFINITY, false);
            // assert!(isect.is_some());
            isect
                .map(|i| (i.nodes_intersected, i.primitives_intersected))
                .unwrap_or((0, 0))
        })
        .reduce(|| (0, 0), |a, b| (a.0 + b.0, a.1 + b.1));

    let elapsed = start.elapsed();

    println!("Intersected {} nodes", nodes_isect);
    println!("Intersected {} primitives", primitives_isect);

    println!("Took: {}us", elapsed.as_micros());

    println!("Model: {}", model);
    println!("Real triangles: {}", num_triangles);

    let total_virtual_isects = iters * threads * num_triangles;
    println!("Total virtual intersections: {}", total_virtual_isects);

    println!(
        "Trillion triangles/s: {}",
        total_virtual_isects as f64 / elapsed.as_secs_f64() / 1_000_000_000_000.0
    );
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
