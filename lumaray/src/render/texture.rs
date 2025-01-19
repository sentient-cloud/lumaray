use std::sync::Arc;

use crate::{
    compositor::{Image, RGBA},
    utils::HashindMap,
};

use ultraviolet::DVec3;

pub enum Texture {
    /// A solid color texture.
    SolidColor(RGBA),
    /// An image texture.
    Image(Image),
    /// An implicit texture, fn takes (u, v, w) in range [0.0, 1.0] and returns a color.
    Implicit(Box<dyn Fn(&DVec3) -> RGBA + Send + Sync>),
}

impl Texture {
    pub fn new_solid_color(color: RGBA) -> Self {
        Self::SolidColor(color)
    }

    pub fn new_image(image: Image) -> Self {
        Self::Image(image)
    }

    pub fn new_implicit(f: impl Fn(&DVec3) -> RGBA + Send + Sync + 'static) -> Self {
        Self::Implicit(Box::new(f))
    }

    pub fn sample(&self, point: &DVec3) -> RGBA {
        match self {
            Texture::SolidColor(color) => *color,
            Texture::Image(image) => {
                let x = point.x * (image.width as f64);
                let y = point.y * (image.height as f64);
                *image.get(x as usize, y as usize)
            }
            Texture::Implicit(f) => f(point),
        }
    }
}

pub struct TexturePool {
    texures: HashindMap<String, Arc<Texture>>,
}

impl TexturePool {
    pub fn new() -> Self {
        Self {
            texures: HashindMap::new(),
        }
    }

    /// Gets a texture from the pool.
    ///
    /// Returns `None` if the texture is not found.
    pub fn get_by_key(&self, name: String) -> Option<Arc<Texture>> {
        self.texures.get_by_key(name).map(|texture| texture.clone())
    }

    /// Inserts a texture into the pool.
    ///
    /// The name cannot start with '@'.
    pub fn insert(&mut self, name: String, texture: Texture) -> Option<(usize, Arc<Texture>)> {
        if name.starts_with('@') {
            None
        } else {
            let arc = Arc::new(texture);
            let hashind = self.texures.insert(name, arc.clone());
            Some((hashind, arc))
        }
    }

    /// Inserts an unnamed texture into the pool.
    ///
    /// The name is generated from the pointer to the texture.
    pub fn insert_unnamed(&mut self, texture: Texture) -> (String, (usize, Arc<Texture>)) {
        let arc = Arc::new(texture);
        let name = format!("@unnamed_texture_{:p}", arc);
        let hashind = self.texures.insert(name.clone(), arc.clone());
        (name, (hashind, arc))
    }
}
