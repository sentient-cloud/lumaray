use rayon::prelude::*;
use rayon::slice::ParallelSlice;

use ultraviolet::{DVec3, Vec3};

#[cfg(any(feature = "avx2", feature = "avx512"))]
use std::arch::x86_64::*;

use crate::render::AABB;

#[derive(Debug, Clone)]
pub struct Triangle {
    pub vertices: [Vec3; 3],
}

#[cfg(feature = "no-simd")]
#[derive(Debug, Clone)]
pub struct TriangleChunk {
    pub vertices: [Vec3; 3],
}

#[cfg(feature = "no-simd")]
pub const MESH_CHUNK_SIZE: usize = 8;

#[cfg(all(feature = "avx2", not(feature = "no-simd"), not(feature = "avx512")))]
#[derive(Debug, Clone)]
pub struct TriangleChunk {
    pub edge1: [__m256; 3],
    pub edge2: [__m256; 3],
    pub vert0: [__m256; 3],
    pub inactive: __m256,
}

#[cfg(all(feature = "avx2", not(feature = "no-simd"), not(feature = "avx512")))]
pub const MESH_CHUNK_SIZE: usize = 8;

#[cfg(feature = "avx512")]
#[derive(Debug, Clone)]
pub struct MeshChunk {
    pub edge1: [__m512; 3],
    pub edge2: [__m512; 3],
    pub vert0: [__m512; 3],
    pub inactive: __m512,
}

impl MeshChunk {
    #[cfg(feature = "avx512")]
    pub fn new(triangles: &[Triangle]) -> Self {
        debug_assert!(triangles.len() > MESH_CHUNK_SIZE);

        #[repr(align(64))]
        #[derive(Clone, Copy)]
        struct AlignedF32x16([f32; MESH_CHUNK_SIZE]);

        let mut edge1 = [AlignedF32x16([0.0; MESH_CHUNK_SIZE]); 3];
        let mut edge2 = [AlignedF32x16([0.0; MESH_CHUNK_SIZE]); 3];
        let mut vert0 = [AlignedF32x16([0.0; MESH_CHUNK_SIZE]); 3];
        let mut inactive = [1.0; MESH_CHUNK_SIZE];

        for i in 0..triangles.len().min(MESH_CHUNK_SIZE) {
            let e1 = triangles[i].vertices[1] - triangles[i].vertices[0];
            let e2 = triangles[i].vertices[2] - triangles[i].vertices[0];
            let v0 = triangles[i].vertices[0];

            edge1[0].0[i] = e1.x;
            edge1[1].0[i] = e1.y;
            edge1[2].0[i] = e1.z;
            edge2[0].0[i] = e2.x;
            edge2[1].0[i] = e2.y;
            edge2[2].0[i] = e2.z;
            vert0[0].0[i] = v0.x;
            vert0[1].0[i] = v0.y;
            vert0[2].0[i] = v0.z;
            inactive[i] = 0.0;
        }

        unsafe {
            Self {
                edge1: [
                    _mm512_load_ps(edge1[0].0.as_ptr()),
                    _mm512_load_ps(edge1[1].0.as_ptr()),
                    _mm512_load_ps(edge1[2].0.as_ptr()),
                ],
                edge2: [
                    _mm512_load_ps(edge2[0].0.as_ptr()),
                    _mm512_load_ps(edge2[1].0.as_ptr()),
                    _mm512_load_ps(edge2[2].0.as_ptr()),
                ],
                vert0: [
                    _mm512_load_ps(vert0[0].0.as_ptr()),
                    _mm512_load_ps(vert0[1].0.as_ptr()),
                    _mm512_load_ps(vert0[2].0.as_ptr()),
                ],
                inactive: _mm512_load_ps(inactive.as_ptr()),
            }
        }
    }
}

#[cfg(feature = "avx512")]
pub const MESH_CHUNK_SIZE: usize = 16;

#[derive(Debug, Clone)]
struct OrderedTriangle {
    /// Original vertices of the triangle
    pub vertices: [Vec3; 3],

    /// Index of the triangle in the input sequence
    pub index: i32,

    /// Morton code of the triangle
    pub code: u32,
}

#[derive(Debug)]
struct SpatialClustering {
    pub chunks: Vec<Vec<OrderedTriangle>>,
}

impl SpatialClustering {
    /// Creates a new morton clustering from an iterator of triangles.
    ///
    /// this is dogshit
    pub fn new_morton(triangles: impl Iterator<Item = [Vec3; 3]>) -> Self {
        let triangles = triangles.enumerate().collect::<Vec<_>>();

        debug_assert!(triangles.len() > 0 && triangles.len() < i32::MAX as usize);

        // midpoint and what to scale by, to normalize the bounding box into the unit cube
        // applied as: (point - midpoint) * scale
        let (midpoint, scale) = {
            let (min, max) = triangles
                .par_chunks(8192)
                .map(|chunk| {
                    let mut min = Vec3::broadcast(f32::INFINITY);
                    let mut max = Vec3::broadcast(f32::NEG_INFINITY);
                    for (_, tri) in chunk {
                        let point = (tri[0] + tri[1] + tri[2]) / 3.0;
                        min = min.min_by_component(point);
                        max = max.max_by_component(point);
                    }
                    (min, max)
                })
                .reduce(
                    || {
                        (
                            Vec3::broadcast(f32::INFINITY),
                            Vec3::broadcast(f32::NEG_INFINITY),
                        )
                    },
                    |(a_min, a_max), (b_min, b_max)| {
                        (a_min.min_by_component(b_min), a_max.max_by_component(b_max))
                    },
                );

            let size = max - min;
            let scale = 1.0 / size.component_max();
            let midpoint = min + size * 0.5;

            (midpoint, scale)
        };

        // normalize triangles, and assign morton codes
        let mut triangles = triangles
            .par_iter()
            .map(|(i, tri)| {
                let point = (tri[0] + tri[1] + tri[2]) / 3.0;
                let point = (point - midpoint) * scale + Vec3::broadcast(0.5);

                debug_assert!(point.x >= 0.0 && point.x <= 1.0);
                debug_assert!(point.y >= 0.0 && point.y <= 1.0);
                debug_assert!(point.z >= 0.0 && point.z <= 1.0);

                let code = Self::morton30(point.x, point.y, point.z);

                OrderedTriangle {
                    vertices: *tri,
                    index: *i as i32,
                    code,
                }
            })
            .collect::<Vec<_>>();

        // sort by morton code
        triangles.par_sort_unstable_by(|a, b| a.code.cmp(&b.code));

        const BASE_MORTON_MASK: u32 = !0 >> 2;

        macro_rules! morton_masks {
            ($($x:expr),* $(,)?) => {
                {
                    const MASKS: &[u32] = &[$($x),*];
                    MASKS
                }
            };
        }

        const MORTON_MASKS: &[u32] = morton_masks![
            // (BASE_MORTON_MASK << 3) & BASE_MORTON_MASK,
            // (BASE_MORTON_MASK << 6) & BASE_MORTON_MASK,
            // (BASE_MORTON_MASK << 9) & BASE_MORTON_MASK,
            // (BASE_MORTON_MASK << 12) & BASE_MORTON_MASK,
            // (BASE_MORTON_MASK << 15) & BASE_MORTON_MASK,
            // (BASE_MORTON_MASK << 18) & BASE_MORTON_MASK,
            (BASE_MORTON_MASK << 21) & BASE_MORTON_MASK,
            (BASE_MORTON_MASK << 24) & BASE_MORTON_MASK,
            (BASE_MORTON_MASK << 27) & BASE_MORTON_MASK,
        ];

        struct Cluster {
            pub triangles: Vec<OrderedTriangle>,
            pub mask: usize,
            pub cluster_code: u32,
        }

        // make an initial cluster containing all the triangles
        // (effectively chunking by the fully masked out morton code, at division 0)
        let mut clusters = vec![Cluster {
            triangles,
            mask: MORTON_MASKS.len() - 1,
            cluster_code: 0,
        }];

        // splits a cluster into smaller clusters
        let split_cluster = |cluster: Cluster| -> Vec<Cluster> {
            let mask_index = cluster.mask - 1;
            debug_assert!(mask_index < MORTON_MASKS.len());
            debug_assert!(cluster.triangles.len() > MESH_CHUNK_SIZE);

            // println!("splitting cluster with mask {}", mask_index);

            let mask = MORTON_MASKS[mask_index];

            let mut new_clusters = vec![];

            let mut current_end = cluster.triangles.len();
            let mut current_start = current_end - 1;
            let mut current_code = cluster.triangles[current_end - 1].code & mask;

            while current_start != 0 {
                let code = cluster.triangles[current_start].code & mask;

                if code != current_code {
                    new_clusters.push(Cluster {
                        triangles: cluster.triangles[current_start..current_end].to_vec(),
                        mask: mask_index,
                        cluster_code: current_code,
                    });

                    current_end = current_start;
                    current_code = code;
                }

                current_start -= 1;
            }

            if current_end != current_start {
                new_clusters.push(Cluster {
                    triangles: cluster.triangles[0..current_end].to_vec(),
                    mask: mask_index,
                    cluster_code: current_code,
                });
            }

            // println!(
            //     "split cluster of {} triangles into {} clusters",
            //     cluster.triangles.len(),
            //     new_clusters.len()
            // );

            new_clusters
        };

        // split clusters until all are small enough
        // (the general uglyness of the code below is due to the borrow checker)
        let mut done = false;

        while !done {
            done = true;

            let mut cluster_to_split = 0;
            let mut simple_split = false;

            for i in (0..clusters.len()).rev() {
                if clusters[i].triangles.len() > MESH_CHUNK_SIZE {
                    cluster_to_split = i;

                    if clusters[i].mask == 0 {
                        simple_split = true;
                    }

                    done = false;
                    break;
                }
            }

            // println!("cluster to split: {}", cluster_to_split);

            if !done {
                // move the cluster out of the clusters vector by exchanging their
                // internal triangle vector pointers, as to avoid copying it
                let mut cluster = Cluster {
                    triangles: vec![],
                    mask: clusters[cluster_to_split].mask,
                    cluster_code: clusters[cluster_to_split].cluster_code,
                };

                std::mem::swap(
                    &mut cluster.triangles,
                    &mut clusters[cluster_to_split].triangles,
                );

                let old_length = cluster.triangles.len();

                let new_clusters = if simple_split {
                    // split the cluster into MESH_CHUNK_SIZE chunks

                    let mut new_clusters = vec![];

                    let mut i = cluster.triangles.len();
                    while i > 0 {
                        new_clusters.push(Cluster {
                            triangles: cluster
                                .triangles
                                .split_off(i.max(MESH_CHUNK_SIZE) - MESH_CHUNK_SIZE),
                            mask: cluster.mask,
                            cluster_code: cluster.cluster_code,
                        });

                        i = i.max(MESH_CHUNK_SIZE) - MESH_CHUNK_SIZE;
                    }

                    new_clusters
                } else {
                    // println!("clustering morton split");
                    split_cluster(cluster)
                };

                #[cfg(debug_assertions)]
                {
                    let new_length = new_clusters
                        .iter()
                        .map(|c| c.triangles.len())
                        .sum::<usize>();
                    debug_assert_eq!(new_length, old_length);
                }

                clusters.splice(cluster_to_split..=cluster_to_split, new_clusters);
            }
        }

        let mut chunked_mesh = Self { chunks: vec![] };

        for cluster in clusters {
            chunked_mesh.chunks.push(cluster.triangles);
        }

        chunked_mesh
    }

    pub fn morton30(x: f32, y: f32, z: f32) -> u32 {
        debug_assert!(!(x < 0.0 || x > 1.0 || y < 0.0 || y > 1.0 || z < 0.0 || z > 1.0));

        let shift = |x: u32| -> u32 {
            // (literal) edge case of x == 1.0 (1024)
            let x = if x == 1024 { x - 1 } else { x };

            let x = (x | (x << 16)) & 0x300000ff;
            let x = (x | (x << 8)) & 0x300f00f;
            let x = (x | (x << 4)) & 0x30c30c3;
            let x = (x | (x << 2)) & 0x9249249;

            x
        };

        let x = (x * 1024.0) as u32;
        let y = (y * 1024.0) as u32;
        let z = (z * 1024.0) as u32;

        shift(x) | (shift(y) << 1) | (shift(z) << 2)
    }

    pub fn new_bvh(triangles: impl Iterator<Item = [Vec3; 3]>) -> Self {
        type Partition = (i32, i32); // (left, right)

        let triangles = triangles
            .enumerate()
            .map(|(i, tri)| OrderedTriangle {
                vertices: tri,
                index: i as i32,
                code: 0,
            })
            .collect::<Vec<_>>();

        let midpoints = triangles
            .par_iter()
            .map(|tri| (tri.vertices[0] + tri.vertices[1] + tri.vertices[2]) / 3.0)
            .collect::<Vec<_>>();

        let bboxes = triangles
            .par_iter()
            .map(|tri| {
                let mut aabb = AABB::null();
                for vertex in tri.vertices.iter() {
                    aabb.contain_point(DVec3::new(
                        vertex.x as f64,
                        vertex.y as f64,
                        vertex.z as f64,
                    ));
                }
                aabb
            })
            .collect::<Vec<_>>();

        let mut queue = Vec::<Partition>::with_capacity(triangles.len());
        let mut refs = (0..triangles.len() as i32).collect::<Vec<_>>();

        queue.push((0, refs.len() as i32));

        let mut clusters = vec![];

        while let Some((left, right)) = queue.pop() {
            let mut bounding_box = AABB::null();

            if right - left >= 8192 * 3 / 2 {
                bounding_box = refs[left as usize..right as usize]
                    .par_chunks(8192)
                    .map(|chunk| {
                        let mut bbox = AABB::null();
                        for i in chunk {
                            bbox.contain_aabb(&bboxes[*i as usize]);
                        }
                        bbox
                    })
                    .reduce(
                        || AABB::null(),
                        |a, b| {
                            let mut bbox = a;
                            bbox.contain_aabb(&b);
                            bbox
                        },
                    );
            } else {
                unsafe {
                    (left..right)
                        .map(|i| *refs.get_unchecked(i as usize))
                        .for_each(|i| {
                            bounding_box.contain_aabb(&bboxes[i as usize]);
                        })
                }
            }

            if right - left <= MESH_CHUNK_SIZE as i32 {
                clusters.push(
                    refs[left as usize..right as usize]
                        .iter()
                        .map(|i| triangles[*i as usize].clone())
                        .collect::<Vec<_>>(),
                );
            } else {
                let longest_axis: usize = {
                    let size = bounding_box.max - bounding_box.min;
                    if size.x > size.y && size.x > size.z {
                        0
                    } else if size.y > size.x && size.y > size.z {
                        1
                    } else {
                        2
                    }
                };

                // sort refs[left..right] by midpoint[longest_axis]
                unsafe {
                    refs[left as usize..right as usize].par_sort_unstable_by(|a, b| {
                        if midpoints.get_unchecked(*a as usize)[longest_axis]
                            < midpoints.get_unchecked(*b as usize)[longest_axis]
                        {
                            std::cmp::Ordering::Less
                        } else {
                            std::cmp::Ordering::Greater
                        }
                    })
                }

                let mid = (left + right) / 2;
                queue.push((left, mid));
                queue.push((mid, right));
            }
        }

        SpatialClustering { chunks: clusters }
    }
}

#[derive(Debug)]
pub struct Mesh {
    triangles: Vec<MeshChunk>,
    normals: Vec<[Vec3; 3]>,
    uvws: Vec<[Vec3; 3]>,
}

#[cfg(test)]
mod tests {
    use std::{
        fs::File,
        io::{BufReader, BufWriter},
    };

    use rand::Rng;
    use rayon::{iter::ParallelIterator, slice::ParallelSliceMut};

    use crate::{
        parsers::stl::{STLTriangle, STL},
        render::raytracable::mesh::MESH_CHUNK_SIZE,
    };

    use super::SpatialClustering;

    fn random_rgb1() -> [f32; 4] {
        let mut rng = rand::thread_rng();
        [
            rng.gen_range(0.0..1.0),
            rng.gen_range(0.0..1.0),
            rng.gen_range(0.0..1.0),
            1.0,
        ]
    }

    fn rgb1_to_u16(color: [f32; 4]) -> u16 {
        let r = (color[0] * 31.0) as u16;
        let g = (color[1] * 63.0) as u16;
        let b = (color[2] * 31.0) as u16;
        (r << 11) | (g << 5) | b
    }

    #[test]
    fn test_spatial_chunk_dumb() {
        let mut bufreader = BufReader::new(File::open("../data/models/3DBenchy.stl").unwrap());
        let mut triangles = STL::new_from_bufreader(&mut bufreader).unwrap();

        triangles.triangles.par_chunks_mut(8192).for_each(|chunk| {
            let color = rgb1_to_u16(random_rgb1());
            for tri in chunk {
                tri.attribute = color;
            }
        });

        let mut bufwriter =
            BufWriter::new(File::create("../data/models/3DBenchy_color.stl").unwrap());

        triangles.write(&mut bufwriter).unwrap();
    }

    #[test]
    fn test_spatial_chunk_morton() {
        let mut bufreader = BufReader::new(File::open("../data/models/Asian_Dragon.stl").unwrap());
        let stl = STL::new_from_bufreader(&mut bufreader).unwrap();

        let now = std::time::Instant::now();

        println!("stl has {} triangles", stl.triangles.len());

        let clustering =
            SpatialClustering::new_morton(stl.triangles.iter().map(|tri| tri.vertices));

        println!(
            "clustering took {:?} microseconds",
            now.elapsed().as_micros()
        );

        let mut new_stl = STL { triangles: vec![] };

        println!("clustering has {} chunks", clustering.chunks.len());
        let mut chunk_sizes = [0; MESH_CHUNK_SIZE];
        for chunk in &clustering.chunks {
            chunk_sizes[chunk.len().min(MESH_CHUNK_SIZE) - 1] += 1;
        }

        for (i, size) in chunk_sizes.iter().enumerate() {
            println!("chunk size {}: {}", i + 1, size);
        }

        for (_i, chunk) in clustering.chunks.iter().enumerate() {
            let color = rgb1_to_u16(random_rgb1());

            for tri in chunk {
                new_stl.triangles.push(STLTriangle {
                    vertices: tri.vertices,
                    normal: stl.triangles[tri.index as usize].normal,
                    attribute: color, //(tri.code >> 16) as u16,
                });
            }
        }

        println!("{:?}", new_stl.triangles.len());

        let mut bufwriter =
            BufWriter::new(File::create("../data/models/Asian_Dragon_chunked.stl").unwrap());

        new_stl.write(&mut bufwriter).unwrap();
    }

    #[test]
    fn test_spatial_chunk_bvh() {
        let mut bufreader = BufReader::new(File::open("../data/models/Asian_Dragon.stl").unwrap());
        let stl = STL::new_from_bufreader(&mut bufreader).unwrap();

        let now = std::time::Instant::now();

        println!("stl has {} triangles", stl.triangles.len());

        let clustering = SpatialClustering::new_bvh(stl.triangles.iter().map(|tri| tri.vertices));

        println!(
            "clustering took {:?} microseconds",
            now.elapsed().as_micros()
        );

        let mut new_stl = STL { triangles: vec![] };

        println!("clustering has {} chunks", clustering.chunks.len());
        let mut chunk_sizes = [0; MESH_CHUNK_SIZE];
        for chunk in &clustering.chunks {
            chunk_sizes[chunk.len().min(MESH_CHUNK_SIZE) - 1] += 1;
        }

        for (i, size) in chunk_sizes.iter().enumerate() {
            println!("chunk size {}: {}", i + 1, size);
        }

        for (_i, chunk) in clustering.chunks.iter().enumerate() {
            let color = rgb1_to_u16(random_rgb1());

            for tri in chunk {
                new_stl.triangles.push(STLTriangle {
                    vertices: tri.vertices,
                    normal: stl.triangles[tri.index as usize].normal,
                    attribute: color, //(tri.code >> 16) as u16,
                });
            }
        }

        println!("{:?}", new_stl.triangles.len());

        let mut bufwriter =
            BufWriter::new(File::create("../data/models/Asian_Dragon_chunked.stl").unwrap());

        new_stl.write(&mut bufwriter).unwrap();
    }
}
