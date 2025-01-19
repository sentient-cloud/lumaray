#[cfg(feature = "no-simd")]
mod no_simd;

#[cfg(all(feature = "avx2", not(feature = "no-simd"), not(feature = "avx512")))]
mod avx2;

#[cfg(feature = "avx512")]
mod avx512;

use std::collections::{BTreeSet, HashSet, VecDeque};

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

    /// Index to left child node, -1 if leaf
    pub left: i32,

    /// Index to right child node, or,
    /// index to the first primitive in the node, if its a leaf
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
    T: BoundedGeometry + RaytracableGeometry,
{
    nodes: Vec<BVHNode>,
    _marker: std::marker::PhantomData<T>,
}

pub struct BVHTraverseResult {
    pub node_id: i32,
    pub primitive_id: i32,
    pub t: f64,
    pub nodes_visited: usize,
}

impl<T> BVH<T>
where
    T: BoundedGeometry + RaytracableGeometry,
{
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

        struct NodeData {
            bounding_box: AABB,
            parent_id: i32,
            is_leaf: bool,
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
        let mut leaves = BTreeSet::<i32>::new();

        // as to avoid having to sort the primitives, we instead sort this badboy,
        // and use it to access the primitives in the correct order
        // cAcHe lOcalIty is fUckEd bUt i doNt cArE
        // mesh instancing more important
        let mut refs = (0..primitives.len() as i32).collect::<Vec<_>>();

        queue.push((0, refs.len() as i32, 0, false));

        while let Some((left, right, parent, is_left)) = queue.pop() {
            let mut node = NodeData {
                bounding_box: AABB::null(),
                parent_id: parent,
                is_leaf: false,
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
                // we are at a leaf, so write the primitive index to node.right
                node.is_leaf = true;
                node.right = *refs.get(left as usize).unwrap();
                nodes.push(node);
                leaves.insert(node_id);
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

                // split down the middle like a real monket
                let mid = (left + right) / 2;
                queue.push((left, mid, node_id, true));
                queue.push((mid, right, node_id, false));
            }
        }

        // build parent -> child refs
        // this starts at each leaf node, and walks up the tree until it reaches the root
        let mut todo = VecDeque::<i32>::with_capacity(leaves.len());
        let mut done = HashSet::<i32>::with_capacity(leaves.len());

        leaves.iter().for_each(|&leaf| {
            todo.push_back(leaf);
        });

        while todo.len() > 0 {
            let id = todo.pop_front().unwrap();
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
                todo.push_back(parent_id);
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
                        } else {
                            TwoVolume::new(
                                nodes.get_unchecked(node.left as usize).bounding_box,
                                nodes.get_unchecked(node.right as usize).bounding_box,
                            )
                        }
                    };

                    BVHNode::new(volume, node.left, node.right)
                })
                .collect::<Vec<_>>(),
            _marker: std::marker::PhantomData,
        }
    }

    pub fn traverse(&self, ray: &Ray, max_t: f64, primitives: &[T]) -> BVHTraverseResult {
        let two_ray = TwoRay::new(*ray);

        let mut nodes_visited = 0;
        let mut stack: [i32; 128] = [0; 128];
        let mut stack_ptr = 1;

        let mut max_t = max_t;
        let mut node_id = -1i32;
        let mut primitive_id = -1i32;

        while stack_ptr > 0 {
            nodes_visited += 1;

            #[cfg(debug_assertions)]
            if stack_ptr >= stack.len() {
                panic!("Stack overflow");
            }

            stack_ptr -= 1;
            let id = unsafe { stack.get_unchecked(stack_ptr) };
            let node = unsafe { self.nodes.get_unchecked(*id as usize) };

            #[cfg(debug_assertions)]
            if primitives.len() < node.right as usize {
                panic!("Primitive index out of bounds");
            }

            if !node.is_leaf() {
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
                #[cfg(debug_assertions)]
                if node.right < 0 || node.right > primitives.len() as i32 {
                    panic!("Primitive index out of bounds");
                }

                let primitive = unsafe { primitives.get_unchecked(node.right as usize) };

                if let Some(t) = primitive.thin_intersection(ray, false) {
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
        }
    }
}
