pub mod cube;

use cgmath::*;

use super::render::PaintedFaceVertex;

pub fn vertex(p: [i8; 3], c: [i8; 3], i: u32) -> PaintedFaceVertex {
    PaintedFaceVertex {
        position: [p[0] as f32, p[1] as f32, p[2] as f32, 1.0],
        color: [c[0] as f32, c[1] as f32, c[2] as f32, 1.0],
        normals: {
            let n = Vector3::<f32>::new(p[0] as f32, p[1] as f32, p[2] as f32).normalize();
            [n.x, n.y, n.z, 0.0]
        },
        index: i,
    }
}

pub fn create_vertices(positions: &Vec<[i8; 3]>, colors: &Vec<[i8; 3]>) -> Vec<PaintedFaceVertex> {
    let mut data: Vec<PaintedFaceVertex> = Vec::with_capacity(positions.len());
    for i in 0..positions.len() {
        data.push(vertex(positions[i], colors[i], (i as u32) / 3));
    }
    data.to_vec()
}
