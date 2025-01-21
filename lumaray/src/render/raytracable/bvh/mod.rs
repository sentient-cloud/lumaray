//! Bounding Volume Hierarchy (BVH) implementation.
//!
//! Anything that implements the `BoundedGeometry` trait can be used to build a BVH.
//!
//! The implementation for the `RaytracableGeometry` trait is optional, and is split
//! in order to specialize the BVH for meshes.

#[cfg(feature = "no-simd")]
mod no_simd;

#[cfg(all(feature = "avx2", not(feature = "no-simd"), not(feature = "avx512")))]
mod avx2;

#[cfg(feature = "avx512")]
mod avx512;

use std::collections::{HashSet, VecDeque};

#[cfg(feature = "no-simd")]
#[allow(unused_imports)]
pub use no_simd::*;

#[cfg(all(feature = "avx2", not(feature = "no-simd"), not(feature = "avx512")))]
#[allow(unused_imports)]
pub use avx2::*;

#[cfg(feature = "avx512")]
#[allow(unused_imports)]
pub use avx512::*;

use crate::render::AABB;

use super::{BoundedGeometry, Ray, RaytracableGeometry};

use rayon::prelude::*;
use rayon::slice::ParallelSlice;

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

        println!("primitives: {}", primitives.len());

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
