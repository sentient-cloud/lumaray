use std::sync::Arc;

use crate::utils::HashindMap;

use super::Texture;

pub struct Scene {
    textures: HashindMap<String, Arc<Texture>>,
}

impl Scene {
    pub fn new() -> Self {
        Self {
            textures: HashindMap::new(),
        }
    }

    pub fn textures(&self) -> &HashindMap<String, Arc<Texture>> {
        &self.textures
    }

    pub fn textures_mut(&mut self) -> &mut HashindMap<String, Arc<Texture>> {
        &mut self.textures
    }
}
