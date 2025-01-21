//! Bounding Volume Hierarchy (BVH) implementation.
//!
//! Anything that implements the `BoundedGeometry` trait can be used to build a BVH.
//!
//! The implementation for the `RaytracableGeometry` trait is optional, and is split
//! in order to specialize the BVH for meshes.

#[cfg(not(feature = "no-simd"))]
use std::arch::x86_64::*;

use rayon::prelude::*;
use rayon::slice::ParallelSlice;
use std::collections::{HashSet, VecDeque};
use ultraviolet::DVec3;

use super::{BoundedGeometry, Ray, RaytracableGeometry};
use crate::{
    render::AABB,
    utils::{
        alignedmem::{AlignedF64x16, AlignedF64x4},
        constconstrain,
    },
};

#[derive(Debug, Clone, Copy)]
pub struct TwoVolumeTest {
    pub state: u8,     // 0b00 = no hit, 0b01 = left hit, 0b10 = right hit, 0b11 = both hit
    pub left_t1: f64,  // left tnear if state & 0b01 != 0, else trash
    pub left_t2: f64,  // left tfar if state & 0b01 != 0, else trash
    pub right_t1: f64, // right tnear if state & 0b10 != 0, else trash
    pub right_t2: f64, // right tfar if state & 0b10 != 0, else trash
}

impl TwoVolumeTest {
    pub fn hit(&self) -> bool {
        self.state != 0
    }

    pub fn left_hit(&self) -> bool {
        self.state & 0b01 != 0
    }

    pub fn right_hit(&self) -> bool {
        self.state & 0b10 != 0
    }

    pub fn left_t(&self) -> (f64, f64) {
        (self.left_t1, self.left_t2)
    }

    pub fn right_t(&self) -> (f64, f64) {
        (self.right_t1, self.right_t2)
    }
}

#[derive(Debug, Clone)]
pub struct BVHNode {
    /// Bounding boxes of the left and right child nodes
    pub child_volumes: TwoVolume,

    /// Index to left child node, or geometry index if it contains geometry
    pub left: i32,

    /// Index to right child node, or geometry index if it contains geometry
    pub right: i32,
}

impl BVHNode {
    pub fn new(child_volumes: TwoVolume, left: i32, right: i32) -> Self {
        Self {
            child_volumes,
            left,
            right,
        }
    }

    pub fn is_leaf(&self) -> bool {
        self.left == -1
    }

    pub fn left_child(&self) -> i32 {
        self.left
    }

    pub fn right_child(&self) -> i32 {
        self.right
    }

    pub fn child_volumes(&self) -> &TwoVolume {
        &self.child_volumes
    }
}

/// A BVH.
///
/// Simply contains a list of nodes, and takes no ownership of the underlying geometry.
///
/// The BVH works like an indexing into the geometry, and the geometry is not modified or sorted in any way.
#[derive(Debug)]
pub struct BVH<T>
where
    T: BoundedGeometry,
{
    nodes: Vec<BVHNode>,
    _marker: std::marker::PhantomData<T>,
}

pub struct BVHTraverseResult {
    pub node_id: i32,
    pub primitive_id: i32,
    pub t: f64,
    pub nodes_visited: usize,
    pub nodes_intersected: usize,
    pub primitives_intersected: usize,
}

impl<T> BVH<T>
where
    T: BoundedGeometry,
{
    pub fn nodes(&self) -> &[BVHNode] {
        &self.nodes
    }

    pub fn build(primitives: &[T]) -> BVH<T> {
        type Partition = (i32, i32, i32, bool); // (left, right, parent, is_left)

        if primitives.is_empty() {
            return BVH {
                nodes: Vec::new(),
                _marker: std::marker::PhantomData,
            };
        }

        if primitives.len() > i32::MAX as usize {
            panic!("Too many primitives");
        }

        #[derive(Debug)]
        struct NodeData {
            bounding_box: AABB,
            parent_id: i32,
            left: i32,
            right: i32,
            is_left_child: bool,
        }

        let midpoints = primitives
            .iter()
            .map(|obj| obj.center_point())
            .collect::<Vec<_>>();

        let bboxes = primitives
            .iter()
            .map(|obj| obj.world_bounding_box())
            .collect::<Vec<_>>();

        // wörk
        let mut queue = Vec::<Partition>::with_capacity(primitives.len());

        // temporary nodes
        let mut nodes = Vec::<NodeData>::with_capacity(primitives.len() * 2);

        // leaf node references to build parent -> child refs
        let mut leaves = VecDeque::<i32>::with_capacity(primitives.len() / 2);

        // as to avoid having to sort the primitives, we instead sort this badboy,
        // and use it to access the primitives in the correct order
        // cAcHe lOcalIty is fUckEd bUt i doNt cArE
        // mesh instancing more important
        let mut refs = (0..primitives.len() as i32).collect::<Vec<_>>();

        queue.push((0, primitives.len() as i32, 0, false));

        // println!("primitives: {}", primitives.len());

        while let Some((left, right, parent, is_left)) = queue.pop() {
            let mut node = NodeData {
                bounding_box: AABB::null(),
                parent_id: parent,
                left: -1,
                right: -1,
                is_left_child: is_left,
            };

            if right - left >= 8192 * 3 / 2 {
                // we have a fucklarge parent node, so chunk the
                // primitives and compute its bbox in parallell
                node.bounding_box = refs[left as usize..right as usize]
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
                    // fuck you i use get_unchecked where i want
                    (left..right)
                        .map(|i| *refs.get_unchecked(i as usize))
                        .for_each(|i| {
                            node.bounding_box.contain_aabb(&bboxes[i as usize]);
                        })
                }
            }

            let node_id = nodes.len() as i32;

            if right - left <= 1 {
                // we are at a leaf
                node.left = -1;
                node.right = *refs.get(left as usize).unwrap();

                nodes.push(node);
                leaves.push_back(node_id);
            } else {
                // need to split

                let longest_axis = {
                    let size = node.bounding_box.max - node.bounding_box.min;
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

                nodes.push(node);

                // split down the middle like a real monket
                let mid = (left + right) / 2;
                queue.push((left, mid, node_id, true));
                queue.push((mid, right, node_id, false));
            }
        }

        // build parent -> child refs
        // this starts at each leaf node, and walks up the tree until it reaches the root
        let mut done = HashSet::<i32>::with_capacity(leaves.len());

        while leaves.len() > 0 {
            let id = leaves.pop_back().unwrap();
            done.insert(id);

            let (is_left_child, parent_id) = unsafe {
                let node = nodes.get_unchecked(id as usize);
                (node.is_left_child, node.parent_id)
            };

            #[cfg(debug_assertions)]
            if parent_id < 0 || parent_id as usize >= nodes.len() {
                panic!("Parent index out of bounds");
            }

            let parent = unsafe { nodes.get_unchecked_mut(parent_id as usize) };

            if is_left_child {
                parent.left = id;
            } else {
                parent.right = id;
            }

            if parent_id != 0 && !done.contains(&parent_id) {
                leaves.push_back(parent_id);
            }
        }

        for node in &mut nodes {
            if node.left == -1 && node.right == -1 {
                panic!("leaf: {:?}", node);
            }
        }

        BVH {
            nodes: nodes
                .iter()
                .enumerate()
                .map(|(i, node)| {
                    let volume = unsafe {
                        if i == 0 {
                            // root node doesnt have a parent, so just slap its own bbox in there twice
                            TwoVolume::new(node.bounding_box, node.bounding_box)
                        } else if node.left != -1 {
                            TwoVolume::new(
                                nodes.get_unchecked(node.left as usize).bounding_box,
                                nodes.get_unchecked(node.right as usize).bounding_box,
                            )
                        } else {
                            TwoVolume::new(AABB::null(), AABB::null())
                        }
                    };

                    BVHNode::new(volume, node.left, node.right)
                })
                .collect::<Vec<_>>(),
            _marker: std::marker::PhantomData,
        }
    }

    pub fn dump_graphviz<W>(&self, writer: &mut W) -> std::io::Result<()>
    where
        W: std::io::Write,
    {
        writer.write_all("digraph G {\n".as_bytes())?;

        for (i, node) in self.nodes.iter().enumerate() {
            if node.is_leaf() {
                writer.write_all(format!("{} [label=\"{}\"];\n", i, node.right).as_bytes())?;
            } else {
                writer.write_all(format!("{} [label=\"{}\"];\n", i, i).as_bytes())?;
            }

            if !node.is_leaf() {
                writer.write_all(format!("{} -> {};\n", i, node.left).as_bytes())?;
                writer.write_all(format!("{} -> {};\n", i, node.right).as_bytes())?;
            }
        }

        writer.write_all("}".as_bytes())?;

        Ok(())
    }
}

impl<T> BVH<T>
where
    T: BoundedGeometry + RaytracableGeometry,
{
    pub fn traverse(&self, ray: &Ray, max_t: f64, primitives: &[T]) -> BVHTraverseResult {
        let two_ray = TwoRay::new(*ray);

        let mut nodes_visited = 0;
        let mut nodes_intersected = 0;
        let mut primitives_intersected = 0;

        let mut stack: [i32; 128] = [0; 128];
        let mut stack_ptr = 1;

        let mut max_t = max_t;
        let mut node_id = -1i32;
        let mut primitive_id = -1i32;

        while stack_ptr > 0 {
            nodes_visited += 1;

            debug_assert!(stack_ptr < stack.len());

            stack_ptr -= 1;
            let id = unsafe { stack.get_unchecked(stack_ptr) };
            let node = unsafe { self.nodes.get_unchecked(*id as usize) };

            debug_assert!(primitives.len() > node.right as usize);

            if !node.is_leaf() {
                nodes_intersected += 1;
                let isect = unsafe { node.child_volumes().test(&two_ray, max_t) };

                match isect.state {
                    0 => {
                        // no hit
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
                debug_assert!(node.right >= 0 && node.right < primitives.len() as i32);

                primitives_intersected += 1;
                let primitive = unsafe { primitives.get_unchecked(node.right as usize) };

                if let Some(t) = primitive.thin_intersection(ray, max_t, false) {
                    if t.t < max_t {
                        max_t = t.t;
                        node_id = *id;
                        primitive_id = node.right;
                    }
                }
            }
        }

        BVHTraverseResult {
            node_id,
            primitive_id,
            t: max_t,
            nodes_visited,
            nodes_intersected,
            primitives_intersected,
        }
    }
}

///
/// no-simd bvh
///

#[cfg(feature = "no-simd")]
pub fn hi() {
    println!("bvh no-simd");
}

#[cfg(feature = "no-simd")]
pub struct TwoRay(pub Ray);

#[cfg(feature = "no-simd")]
impl TwoRay {
    pub fn new(ray: Ray) -> Self {
        Self(ray)
    }
}

#[cfg(feature = "no-simd")]
#[derive(Debug, Clone)]
pub struct TwoVolume {
    pub left: AABB,
    pub right: AABB,
}

#[cfg(feature = "no-simd")]
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

    /// Extracts the AABB at index `I` from the TwoVolume.
    ///
    /// I = 0 extracts the left AABB, I = 1 extracts the right AABB.
    pub fn extract_aabb<const I: usize>(&self) -> AABB {
        if I == 0 {
            self.left
        } else {
            self.right
        }
    }
}

#[cfg(feature = "no-simd")]
impl Into<(AABB, AABB)> for TwoVolume {
    fn into(self) -> (AABB, AABB) {
        (self.left, self.right)
    }
}

///
/// avx2 bvh
///

#[cfg(all(feature = "avx2", not(feature = "no-simd"), not(feature = "avx512")))]
pub fn hi() {
    println!("bvh avx2");
}

#[cfg(all(feature = "avx2", not(feature = "no-simd"), not(feature = "avx512")))]
use crate::render::{Ray, AABB};

#[cfg(all(feature = "avx2", not(feature = "no-simd"), not(feature = "avx512")))]
pub struct TwoRay {
    pub origin: [__m128d; 3],
    pub inv_dir: [__m128d; 3],
}

#[cfg(all(feature = "avx2", not(feature = "no-simd"), not(feature = "avx512")))]
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

#[cfg(all(feature = "avx2", not(feature = "no-simd"), not(feature = "avx512")))]
#[derive(Debug, Clone)]
pub struct TwoVolume {
    pub min: [__m128d; 3], // a min x, b min x, a min y, b min y, a min z, b min z
    pub max: [__m128d; 3], // a max x, b max x, a max y, b max y, a max z, b max z
}

#[cfg(all(feature = "avx2", not(feature = "no-simd"), not(feature = "avx512")))]
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

#[cfg(all(feature = "avx2", not(feature = "no-simd"), not(feature = "avx512")))]
impl Into<(AABB, AABB)> for TwoVolume {
    fn into(self) -> (AABB, AABB) {
        (self.extract_aabb::<0>(), self.extract_aabb::<1>())
    }
}

///
/// avx512 bvh
///
#[cfg(feature = "avx512")]
pub fn hi() {
    println!("bvh avx512");
}

#[cfg(feature = "avx512")]
pub struct TwoRay {
    pub origin: __m512d,
    pub inv_dir: __m512d,
}

#[cfg(feature = "avx512")]
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

#[cfg(feature = "avx512")]
#[derive(Debug, Clone)]
pub struct TwoVolume {
    pub min: __m512d,
    pub max: __m512d,
}

#[cfg(feature = "avx512")]
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
