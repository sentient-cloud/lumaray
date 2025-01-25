//! A mesh implementation.
//!
//! Meshes are basically triangle soup, but they also have normals and uvws.
//!
//! The mesh contains its own BVH based off of chunks of the mesh, which are
//! sized to fit into AVX2 or AVX512 registers. Optionally you can disable
//! simd entirely, in which case each chunk will contain only one triangle.
//!
//! Furthermore the BVH is specialized for meshes, because the underlying
//! `MeshChunk` type does not implement `RaytracableGeometry`.

use rayon::prelude::*;
use rayon::slice::ParallelSlice;

use ultraviolet::{DVec3, Vec3};

#[cfg(any(feature = "avx2", feature = "avx512"))]
use std::arch::x86_64::*;

use crate::{render::AABB, utils::alignedmem::AlignedF32x16};

use super::{
    bvh::{TwoRay, BVH},
    BoundedGeometry, Intersection, Ray, RaytracableGeometry,
};

#[derive(Debug, Clone, Copy)]
pub struct ChunkIntersection {
    t: f32,     // distance to intersection, or undefined if no hit
    index: i32, // index of the triangle in the chunk, or -1 if no hit
}

#[derive(Debug, Clone)]
pub struct OrderedTriangle {
    /// Original vertices of the triangle
    pub vertices: [Vec3; 3],

    /// Index of the triangle in the input sequence
    pub index: i32,

    /// Morton code of the triangle
    pub code: u32,
}

#[cfg(feature = "no-simd")]
pub struct ChunkRay {
    origin: Vec3,
    direction: Vec3,
}

#[cfg(feature = "no-simd")]
#[derive(Debug, Clone)]
pub struct MeshChunk {
    pub edge1: Vec3,
    pub edge2: Vec3,
    pub vert0: Vec3,
}

#[cfg(feature = "no-simd")]
pub const MESH_CHUNK_SIZE: usize = 8;

#[cfg(all(feature = "avx2", not(feature = "no-simd"), not(feature = "avx512")))]
pub const MESH_CHUNK_SIZE: usize = 8;

#[cfg(feature = "avx512")]
pub const MESH_CHUNK_SIZE: usize = 16;

#[cfg(all(feature = "avx2", not(feature = "no-simd"), not(feature = "avx512")))]
pub struct ChunkRay {
    origin: [__m256; 3],
    direction: [__m256; 3],
}

#[cfg(all(feature = "avx2", not(feature = "no-simd"), not(feature = "avx512")))]
#[derive(Debug, Clone)]
pub struct MeshChunk {
    pub edge1: [__m256; 3],
    pub edge2: [__m256; 3],
    pub vert0: [__m256; 3],
    pub active: u8,
    pub index: [i32; MESH_CHUNK_SIZE],
}

#[cfg(feature = "avx512")]
pub struct ChunkRay {
    origin: [__m512; 3],
    direction: [__m512; 3],
}

#[cfg(feature = "avx512")]
#[derive(Debug, Clone)]
pub struct MeshChunk {
    pub edge1: [__m512; 3],
    pub edge2: [__m512; 3],
    pub vert0: [__m512; 3],
    pub active: u16,
    pub index: [i32; MESH_CHUNK_SIZE],
}

// TODO:
// keep the bvh from new_bvh
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
                    split_cluster(cluster)
                };

                debug_assert_eq!(
                    new_clusters
                        .iter()
                        .map(|c| c.triangles.len())
                        .sum::<usize>(),
                    old_length
                );

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
                    .reduce(|| AABB::null(), |a, b| a.containing_aabb(&b));
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
    chunks: Vec<MeshChunk>,
    normals: Vec<[Vec3; 3]>,
    uvws: Vec<[Vec3; 3]>,
    bvh: BVH<MeshChunk>,
}

impl Mesh {
    pub fn new(triangles: Vec<[Vec3; 3]>, normals: Vec<[Vec3; 3]>, uvws: Vec<[Vec3; 3]>) -> Self {
        let mut mesh_chunks = Vec::with_capacity(triangles.len() / MESH_CHUNK_SIZE);
        let mut mesh_normals = Vec::with_capacity(triangles.len());
        let mut mesh_uvws = Vec::with_capacity(triangles.len());

        if MESH_CHUNK_SIZE == 1 {
            for (i, ((tri_vertices, tri_normals), tri_uvws)) in triangles
                .iter()
                .zip(normals.iter())
                .zip(uvws.iter())
                .enumerate()
            {
                #[allow(unused_mut)]
                let mut chunk = MeshChunk::new(&vec![OrderedTriangle {
                    vertices: *tri_vertices,
                    index: i as i32,
                    code: 0,
                }]);

                mesh_chunks.push(chunk);
                mesh_normals.push(*tri_normals);
                mesh_uvws.push(*tri_uvws);
            }
        } else if MESH_CHUNK_SIZE == 8 || MESH_CHUNK_SIZE == 16 {
            let clustering = SpatialClustering::new_bvh(triangles.iter().cloned());

            let mut num = 0;

            for chunk in clustering.chunks.iter() {
                debug_assert!(chunk.len() <= MESH_CHUNK_SIZE);

                for _ in chunk {
                    mesh_normals.push(normals[num]);
                    mesh_uvws.push(uvws[num]);
                    num += 1;
                }

                mesh_chunks.push(MeshChunk::new(chunk));
            }
        } else {
            unreachable!();
        }

        let bvh = BVH::build(&mesh_chunks);

        Self {
            chunks: mesh_chunks,
            normals: mesh_normals,
            uvws: mesh_uvws,
            bvh,
        }
    }

    pub fn normal(&self, index: i32) -> [Vec3; 3] {
        self.normals[index as usize]
    }

    pub fn uvw(&self, index: i32) -> [Vec3; 3] {
        self.uvws[index as usize]
    }

    pub fn chunks(&self) -> &[MeshChunk] {
        &self.chunks
    }
}

impl BoundedGeometry for Mesh {
    fn local_bounding_box(&self) -> AABB {
        if self.bvh.nodes().len() == 0 {
            return AABB::null();
        }

        // root node has same aabb in both lanes
        self.bvh.nodes()[0].child_volumes().extract_aabb::<0>()
    }
}

impl RaytracableGeometry for Mesh {
    fn thin_intersection(&self, ray: &Ray, max_t: f64, _compute_uvw: bool) -> Option<Intersection> {
        let two_ray = unsafe { TwoRay::new(*ray) }; // ray used to traverse the bvh
        let chunk_ray = ChunkRay::new(*ray); // ray used to intersect the chunks

        let mut stack: [i32; 128] = [0; 128];
        let mut stack_ptr = 1;

        let mut max_t = max_t;

        let mut chunk_id = -1i32; // id of mesh chunk intersected
        let mut triangle_ref_id = -1i32;

        let mut nodes_intersected = 0;
        let mut primitives_intersected = 0;

        while stack_ptr > 0 {
            debug_assert!(stack_ptr < stack.len());

            stack_ptr -= 1;
            let node_id = unsafe { stack.get_unchecked(stack_ptr) };

            let node = unsafe { self.bvh.nodes().get_unchecked(*node_id as usize) };

            if !node.is_leaf() {
                nodes_intersected += 1;
                let isect = unsafe { node.child_volumes().test(&two_ray, max_t) };

                match isect.state {
                    0 => {
                        // no intersection
                        continue;
                    }
                    1 => {
                        // left hit
                        stack[stack_ptr] = node.left_child();
                        stack_ptr += 1;
                    }
                    2 => {
                        // right hit
                        stack[stack_ptr] = node.right_child();
                        stack_ptr += 1;
                    }
                    3 => {
                        // both hit, push the furthest one first,
                        // so that the closest one is popped first
                        if isect.left_t1 < isect.right_t1 {
                            stack[stack_ptr] = node.right_child();
                            stack_ptr += 1;

                            stack[stack_ptr] = node.left_child();
                            stack_ptr += 1;
                        } else {
                            stack[stack_ptr] = node.left_child();
                            stack_ptr += 1;

                            stack[stack_ptr] = node.right_child();
                            stack_ptr += 1;
                        }
                    }
                    _ => unreachable!(),
                }
            } else {
                debug_assert!(node.right >= 0 && node.right < self.chunks.len() as i32);

                primitives_intersected += 1;

                let chunk = unsafe { self.chunks.get_unchecked(node.right as usize) };
                let isect = unsafe { chunk_ray.intersect_chunk(chunk, max_t as f32) };

                debug_assert!(isect.index >= -1 && isect.index < MESH_CHUNK_SIZE as i32);

                if chunk.is_active(isect.index) && isect.t < max_t as f32 {
                    max_t = isect.t as f64;
                    chunk_id = node.right;
                    triangle_ref_id = chunk.extract_ref_index(isect.index);
                }
            }
        }

        if chunk_id == -1 {
            debug_assert!(triangle_ref_id == -1);
            None
        } else {
            debug_assert!(triangle_ref_id != -1);

            // let triangle_index = chunk_id * MESH_CHUNK_SIZE as i32 + chunk_inner_id;
            // let ref_id = unsafe { *self.refs.get_unchecked(triangle_index as usize) as usize };

            let normals = self.normals[triangle_ref_id as usize];

            // TODO: compute uvw, smooth normals
            // let chunk = unsafe { self.chunks.get_unchecked(chunk_id as usize) };
            // let vertices = chunk.extract_triangle(chunk_inner_id as usize);
            // let uvws = self.uvws[ref_id];

            Some(Intersection {
                ray: *ray,
                t: max_t,
                normal: DVec3::new(
                    normals[0].x as f64,
                    normals[0].y as f64,
                    normals[0].z as f64,
                ),
                uvw: DVec3::zero(),
                thick_intersection: None,
                nodes_intersected,
                primitives_intersected,
            })
        }
    }
}

#[cfg(feature = "avx512")]
impl MeshChunk {
    pub fn new(triangles: &Vec<OrderedTriangle>) -> Self {
        use crate::utils::alignedmem::AlignedF32x16;

        debug_assert!(triangles.len() <= MESH_CHUNK_SIZE);

        let mut edge1 = [AlignedF32x16([f32::NAN; MESH_CHUNK_SIZE]); 3];
        let mut edge2 = [AlignedF32x16([f32::NAN; MESH_CHUNK_SIZE]); 3];
        let mut vert0 = [AlignedF32x16([f32::NAN; MESH_CHUNK_SIZE]); 3];
        let mut active = 0;
        let mut index = [-1; MESH_CHUNK_SIZE];

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

            active |= 1 << i;

            index[i] = triangles[i].index;
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
                active,
                index,
            }
        }
    }

    pub fn is_active(&self, index: i32) -> bool {
        if index < 0 || index >= MESH_CHUNK_SIZE as i32 {
            false
        } else {
            self.active & (1 << index) > 0
        }
    }

    pub fn extract_ref_index(&self, index: i32) -> i32 {
        self.index[index as usize]
    }

    pub fn extract_triangle(&self, index: usize) -> [Vec3; 3] {
        let mut edge1 = [AlignedF32x16([0.0; MESH_CHUNK_SIZE]); 3];
        let mut edge2 = [AlignedF32x16([0.0; MESH_CHUNK_SIZE]); 3];
        let mut vert0 = [AlignedF32x16([0.0; MESH_CHUNK_SIZE]); 3];

        unsafe {
            _mm512_store_ps(edge1[0].0.as_mut_ptr(), self.edge1[0]);
            _mm512_store_ps(edge1[1].0.as_mut_ptr(), self.edge1[1]);
            _mm512_store_ps(edge1[2].0.as_mut_ptr(), self.edge1[2]);
            _mm512_store_ps(edge2[0].0.as_mut_ptr(), self.edge2[0]);
            _mm512_store_ps(edge2[1].0.as_mut_ptr(), self.edge2[1]);
            _mm512_store_ps(edge2[2].0.as_mut_ptr(), self.edge2[2]);
            _mm512_store_ps(vert0[0].0.as_mut_ptr(), self.vert0[0]);
            _mm512_store_ps(vert0[1].0.as_mut_ptr(), self.vert0[1]);
            _mm512_store_ps(vert0[2].0.as_mut_ptr(), self.vert0[2]);
        }

        [
            Vec3::new(vert0[0].0[index], vert0[1].0[index], vert0[2].0[index]),
            Vec3::new(
                vert0[0].0[index] + edge1[0].0[index],
                vert0[1].0[index] + edge1[1].0[index],
                vert0[2].0[index] + edge1[2].0[index],
            ),
            Vec3::new(
                vert0[0].0[index] + edge2[0].0[index],
                vert0[1].0[index] + edge2[1].0[index],
                vert0[2].0[index] + edge2[2].0[index],
            ),
        ]
    }
}

#[cfg(feature = "avx512")]
impl BoundedGeometry for MeshChunk {
    fn local_bounding_box(&self) -> AABB {
        let mut edge1 = [AlignedF32x16([0.0; MESH_CHUNK_SIZE]); 3];
        let mut edge2 = [AlignedF32x16([0.0; MESH_CHUNK_SIZE]); 3];
        let mut vert0 = [AlignedF32x16([0.0; MESH_CHUNK_SIZE]); 3];

        unsafe {
            _mm512_store_ps(edge1[0].0.as_mut_ptr(), self.edge1[0]);
            _mm512_store_ps(edge1[1].0.as_mut_ptr(), self.edge1[1]);
            _mm512_store_ps(edge1[2].0.as_mut_ptr(), self.edge1[2]);
            _mm512_store_ps(edge2[0].0.as_mut_ptr(), self.edge2[0]);
            _mm512_store_ps(edge2[1].0.as_mut_ptr(), self.edge2[1]);
            _mm512_store_ps(edge2[2].0.as_mut_ptr(), self.edge2[2]);
            _mm512_store_ps(vert0[0].0.as_mut_ptr(), self.vert0[0]);
            _mm512_store_ps(vert0[1].0.as_mut_ptr(), self.vert0[1]);
            _mm512_store_ps(vert0[2].0.as_mut_ptr(), self.vert0[2]);
        }
        let mut aabb = AABB::null();

        for i in 0..MESH_CHUNK_SIZE {
            if self.active & (1 << i) == 0 {
                break;
            }

            let e1 = DVec3::new(
                edge1[0].0[i] as f64,
                edge1[1].0[i] as f64,
                edge1[2].0[i] as f64,
            );
            let e2 = DVec3::new(
                edge2[0].0[i] as f64,
                edge2[1].0[i] as f64,
                edge2[2].0[i] as f64,
            );
            let v0 = DVec3::new(
                vert0[0].0[i] as f64,
                vert0[1].0[i] as f64,
                vert0[2].0[i] as f64,
            );

            let v1 = v0 + e1;
            let v2 = v0 + e2;

            aabb.contain_point(v0);
            aabb.contain_point(v1);
            aabb.contain_point(v2);
        }

        aabb
    }

    fn local_center_point(&self) -> DVec3 {
        let mut edge1 = [AlignedF32x16([0.0; MESH_CHUNK_SIZE]); 3];
        let mut edge2 = [AlignedF32x16([0.0; MESH_CHUNK_SIZE]); 3];
        let mut vert0 = [AlignedF32x16([0.0; MESH_CHUNK_SIZE]); 3];

        unsafe {
            _mm512_store_ps(edge1[0].0.as_mut_ptr(), self.edge1[0]);
            _mm512_store_ps(edge1[1].0.as_mut_ptr(), self.edge1[1]);
            _mm512_store_ps(edge1[2].0.as_mut_ptr(), self.edge1[2]);
            _mm512_store_ps(edge2[0].0.as_mut_ptr(), self.edge2[0]);
            _mm512_store_ps(edge2[1].0.as_mut_ptr(), self.edge2[1]);
            _mm512_store_ps(edge2[2].0.as_mut_ptr(), self.edge2[2]);
            _mm512_store_ps(vert0[0].0.as_mut_ptr(), self.vert0[0]);
            _mm512_store_ps(vert0[1].0.as_mut_ptr(), self.vert0[1]);
            _mm512_store_ps(vert0[2].0.as_mut_ptr(), self.vert0[2]);
        }

        let mut point = DVec3::zero();
        let mut count = 0;

        for i in 0..MESH_CHUNK_SIZE {
            if self.active & (1 << i) == 0 {
                break;
            }

            let e1 = DVec3::new(
                edge1[0].0[i] as f64,
                edge1[1].0[i] as f64,
                edge1[2].0[i] as f64,
            );
            let e2 = DVec3::new(
                edge2[0].0[i] as f64,
                edge2[1].0[i] as f64,
                edge2[2].0[i] as f64,
            );
            let v0 = DVec3::new(
                vert0[0].0[i] as f64,
                vert0[1].0[i] as f64,
                vert0[2].0[i] as f64,
            );

            let v1 = v0 + e1;
            let v2 = v0 + e2;

            point += v0 + v1 + v2;
            count += 3;
        }

        point / count as f64
    }
}

#[cfg(feature = "avx512")]
impl ChunkRay {
    pub fn new(ray: Ray) -> Self {
        unsafe {
            let origin = [
                _mm512_set1_ps(ray.origin.x as f32),
                _mm512_set1_ps(ray.origin.y as f32),
                _mm512_set1_ps(ray.origin.z as f32),
            ];
            let direction = [
                _mm512_set1_ps(ray.direction.x as f32),
                _mm512_set1_ps(ray.direction.y as f32),
                _mm512_set1_ps(ray.direction.z as f32),
            ];

            Self { origin, direction }
        }
    }

    #[target_feature(enable = "avx512f,avx512vl,bmi1")]
    pub unsafe fn intersect_chunk(&self, chunk: &MeshChunk, max_t: f32) -> ChunkIntersection {
        #[inline(always)]
        unsafe fn avx512_cross(a: &[__m512; 3], b: &[__m512; 3]) -> [__m512; 3] {
            [
                _mm512_fmsub_ps(a[1], b[2], _mm512_mul_ps(b[1], a[2])),
                _mm512_fmsub_ps(a[2], b[0], _mm512_mul_ps(b[2], a[0])),
                _mm512_fmsub_ps(a[0], b[1], _mm512_mul_ps(b[0], a[1])),
            ]
        }

        #[inline(always)]
        unsafe fn avx512_dot(a: &[__m512; 3], b: &[__m512; 3]) -> __m512 {
            _mm512_fmadd_ps(
                a[2],
                b[2],
                _mm512_fmadd_ps(a[1], b[1], _mm512_mul_ps(a[0], b[0])),
            )
        }

        #[inline(always)]
        unsafe fn avx512_sub(a: &[__m512; 3], b: &[__m512; 3]) -> [__m512; 3] {
            [
                _mm512_sub_ps(a[0], b[0]),
                _mm512_sub_ps(a[1], b[1]),
                _mm512_sub_ps(a[2], b[2]),
            ]
        }

        const EPS: f32 = 1e-6;

        let q = avx512_cross(&self.direction, &chunk.edge2);
        let a = avx512_dot(&chunk.edge1, &q);

        #[cfg(not(feature = "fast-reciprocal"))]
        let f = _mm512_div_ps(_mm512_set1_ps(1.0), a);

        #[cfg(feature = "fast-reciprocal")]
        let f = _mm512_rcp14_ps(a);

        let s = avx512_sub(&self.origin, &chunk.vert0);
        let r = avx512_cross(&s, &chunk.edge1);

        let u = _mm512_mul_ps(avx512_dot(&s, &q), f);
        let v = _mm512_mul_ps(avx512_dot(&self.direction, &r), f);
        let t = _mm512_mul_ps(avx512_dot(&chunk.edge2, &r), f);

        // t > 0
        let mask = _mm512_mask_cmp_ps_mask::<_CMP_NLE_UQ>(chunk.active, t, _mm512_set1_ps(EPS));

        // t < max_t
        let mask = _mm512_mask_cmp_ps_mask::<_CMP_NGE_UQ>(mask, t, _mm512_set1_ps(max_t));

        // u >= 0
        let mask = _mm512_mask_cmp_ps_mask::<_CMP_NLT_UQ>(mask, u, _mm512_set1_ps(0.0));

        // u <= 1
        let mask = _mm512_mask_cmp_ps_mask::<_CMP_NGT_UQ>(mask, u, _mm512_set1_ps(1.0));

        // v >= 0
        let mask = _mm512_mask_cmp_ps_mask::<_CMP_NLT_UQ>(mask, v, _mm512_set1_ps(0.0));

        // u + v <= 1
        let mask =
            _mm512_mask_cmp_ps_mask::<_CMP_NGT_UQ>(mask, _mm512_add_ps(u, v), _mm512_set1_ps(1.0));

        // llvm doing a funny
        // `mask` is *not* a u16, its an __mmask16 and intellisense
        // has just been gaslighting me this whole time
        // every time i access the mask, it generates a vextractf instruction

        // so instead, mask out the invalid t values (set them to inf),
        // then use reduce_min to find the smallest t value, compare that
        // against the original t values to get a mask of which t value it is,
        // then ACTUALLY convert it to a u16 with mask2int, and count the
        // trailing bits to get the index of the triangle that was hit
        let t = _mm512_mask_blend_ps(mask, _mm512_set1_ps(f32::INFINITY), t);
        let tmin = _mm512_reduce_min_ps(t);
        let tmask = _mm512_cmpeq_ps_mask(t, _mm512_set1_ps(tmin));
        let index = _mm512_mask2int(tmask);

        // oh and theres this, fucker produces a branch to
        // "fix" the case where the input value is 0.
        // let index = index.trailing_zeros();

        let index = _mm_tzcnt_32(index as u32);

        ChunkIntersection {
            t: tmin,
            index: index as i32,
        }
    }
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

        let mut new_stl = STL {
            header: [0; 80],
            triangles: vec![],
        };

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

        let mut new_stl = STL {
            header: [0; 80],
            triangles: vec![],
        };

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
    #[cfg(feature = "avx512")]
    fn test_avx512_chunk_intersection() {
        use ultraviolet::DVec3;

        use crate::render::{
            raytracable::mesh::{ChunkRay, MeshChunk, OrderedTriangle},
            Ray,
        };

        let mut bufreader =
            BufReader::new(File::open("../data/models/Stanford_Bunny.stl").unwrap());
        let stl = STL::new_from_bufreader(&mut bufreader).unwrap();

        let chunk = MeshChunk::new(
            &stl.triangles
                .iter()
                .map(|tri| OrderedTriangle {
                    vertices: tri.vertices,
                    index: 0,
                    code: 0,
                })
                .collect::<Vec<_>>(),
        );

        let ray = Ray::new(DVec3::new(3.0, 0.0, 0.0), DVec3::new(-1.0, 0.0, 0.0));
        let chunk_ray = ChunkRay::new(ray);

        let intersection = unsafe { chunk_ray.intersect_chunk(&chunk, f32::INFINITY) };

        println!("{:?}", intersection);
    }

    #[test]
    #[cfg(feature = "avx512")]
    fn test_avx512_mesh_intersection() {
        use ultraviolet::{DVec3, Vec3};

        use crate::render::{
            raytracable::{mesh::Mesh, RaytracableGeometry},
            Ray,
        };

        let mut bufreader =
            BufReader::new(File::open("../data/models/Stanford_Bunny.stl").unwrap());
        let stl = STL::new_from_bufreader(&mut bufreader).unwrap();

        println!("stl has {} triangles", stl.triangles.len());

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

        let ray = Ray::new(DVec3::new(300.0, 0.01, 0.01), DVec3::new(-1.0, 0.0, 0.0));

        let intersection = mesh.thin_intersection(&ray, f64::INFINITY, false);

        println!("{:#?}", intersection);
    }
}
