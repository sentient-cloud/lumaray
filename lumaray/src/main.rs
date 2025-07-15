#![allow(dead_code)]
#![allow(incomplete_features)]
#![feature(stdarch_x86_avx512)]
#![feature(generic_const_exprs)]
#![feature(avx512_target_feature)]
#![feature(specialization)]
#![feature(generic_const_items)]

use core::f64;
use std::{fs::File, io::BufReader, sync::Arc, time::Instant};

use parsers::stl::STL;
use rand::Rng;
use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};
use render::{
    camera::Camera,
    raytracable::{mesh::Mesh, MeshInstance, RaytracableGeometry},
};
use ultraviolet::{DMat4, Vec3};

#[cfg(all(feature = "no-simd", feature = "avx512"))]
compile_error!("Cannot enable no-simd and avx512 features at the same time");

mod compositor;
mod io;
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

    let model = "../data/models/Asian_Dragon.stl";
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

    let mesh = Arc::new(mesh);

    let mesh_instance1 = MeshInstance::new(
        mesh.clone(),
        DMat4::from_rotation_z(0.3) * DMat4::from_translation(DVec3::new(0.0, -80.0, 0.0)),
    );

    let mesh_instance2 = MeshInstance::new(
        mesh.clone(),
        DMat4::from_rotation_z(-0.3) * DMat4::from_translation(DVec3::new(0.0, 80.0, 0.0)),
    );

    let mut film = compositor::Film::new(2048, 1536);
    let mut chunks = film.subdivide(32);

    let mut camera = render::PerspectiveCamera::new(film.width() as f64, film.height() as f64);

    camera.set_position(DVec3::new(250.0, 50.0, 120.0));
    camera.set_look_at(DVec3::new(0.0, -10.0, 0.0));
    camera.set_direction(camera.forward());
    camera.set_fov(65.0);

    let now = Instant::now();
    let mut sample_time = Instant::now();

    for i in 0..256 {
        chunks.par_iter_mut().for_each(|chunk| {
            chunk.iter().for_each(|(x, y)| {
                let ray = camera.get_primary_ray(
                    x as f64 + rand::thread_rng().gen_range(-0.5..0.5),
                    y as f64 + rand::thread_rng().gen_range(-0.5..0.5),
                );

                let mut hit = mesh_instance1.thin_intersection(&ray, f64::INFINITY, false);
                let hit2 = mesh_instance2.thin_intersection(&ray, f64::INFINITY, false);

                if let Some(hit2) = hit2 {
                    if let Some(hit1) = hit {
                        if hit2.t < hit1.t {
                            hit = Some(hit2);
                        }
                    } else {
                        hit = Some(hit2);
                    }
                }

                if let Some(hit) = hit {
                    chunk.splat(x, y, compositor::RGBA::new_normal(hit.normal));
                } else {
                    chunk.splat(x, y, compositor::RGBA::new(0.1, 0.2, 0.3, 1.0));
                }
            });
        });
        println!("sample {}, took {:?}", i, sample_time.elapsed());
        sample_time = Instant::now();
    }

    // println!("max_x: {}, max_y: {}", max_x, max_y);

    println!("rendered in {:?}", now.elapsed());

    for chunk in chunks {
        film.splat_chunk(chunk);
    }

    let image = compositor::Image::from(film);
    image
        .output_as_png("../images/asian_dragon_normals.png")
        .unwrap();
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
