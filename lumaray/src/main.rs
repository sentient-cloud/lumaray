#![allow(dead_code)]
#![allow(incomplete_features)]
#![feature(stdarch_x86_avx512)]
#![feature(generic_const_exprs)]
#![feature(avx512_target_feature)]

use core::f64;
use std::{fs::File, io::BufReader, ops::Shl, time::Instant};

use parsers::stl::STL;
use rand::{thread_rng, Rng};
use rayon::iter::{IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator};
use render::{
    camera::Camera,
    raytracable::{self, mesh::Mesh, RaytracableGeometry},
};
use ultraviolet::Vec3;

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

    let model = "../data/models/Wikipedia_Globe.stl";
    let mut bufreader = BufReader::new(File::open(model).unwrap());
    let stl = STL::new_from_bufreader(&mut bufreader).unwrap();

    println!("{:#?}", stl.triangles.len());

    let mesh = Mesh::new(
        stl.triangles
            .iter()
            .map(|tri| tri.vertices)
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

    // println!("{:#?}", mesh);

    let mut film = compositor::Film::new(2048, 1536);
    let mut chunks = film.subdivide(16);

    let mut camera = render::PerspectiveCamera::new(film.width() as f64, film.height() as f64);

    camera.set_position(DVec3::new(-1.0, 0.2, 0.2));
    camera.set_look_at(DVec3::zero());

    println!("{:#?}", camera);

    // let sphere = raytracable::Sphere::new(DVec3::zero(), 3.0);

    let now = Instant::now();

    chunks.par_iter_mut().for_each(|chunk| {
        // chunk.fill(compositor::RGBA::new(
        //     thread_rng().gen_range(0.0..1.0),
        //     thread_rng().gen_range(0.0..1.0),
        //     thread_rng().gen_range(0.0..1.0),
        //     1.0,
        // ));

        chunk.iter().for_each(|(x, y)| {
            // println!("({}, {})", x, y);
            let ray = camera.get_primary_ray(x as f64, y as f64);
            let hit = mesh.thin_intersection(&ray, f64::INFINITY, false);

            if let Some(hit) = hit {
                chunk.splat(x, y, compositor::RGBA::new_normal(hit.normal));
            }
        })
    });

    println!("rendered in {:?}", now.elapsed());

    for chunk in chunks {
        film.splat_chunk(chunk);
    }

    let image = compositor::Image::from(film);
    println!("image {}x{}", image.width, image.height);
    image.output_as_png("../trash/chunk_test.png").unwrap();

    // let film = compositor::Film::new(4, 3);
    // let chunks = film.subdivide(1);

    // chunks[0].iter().for_each(|(x, y)| {
    //     println!("({}, {})", x, y);
    // });
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
