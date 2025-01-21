use std::sync::Arc;

use crate::utils::HashindMap;

use super::{raytracable::mesh::Mesh, Texture};

pub struct Scene {
    textures: HashindMap<String, Arc<Texture>>,
    meshes: HashindMap<String, Arc<Mesh>>,
}

impl Scene {
    pub fn new() -> Self {
        Self {
            textures: HashindMap::new(),
            meshes: HashindMap::new(),
        }
    }

    pub fn textures(&self) -> &HashindMap<String, Arc<Texture>> {
        &self.textures
    }

    pub fn textures_mut(&mut self) -> &mut HashindMap<String, Arc<Texture>> {
        &mut self.textures
    }
}
