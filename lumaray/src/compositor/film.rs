use super::RGBA;

pub struct Film {
    pub width: usize,
    pub height: usize,
    pub pixels: Vec<(RGBA, usize)>,
}

impl Film {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            pixels: vec![(RGBA::from(0.0), 0); width * height],
        }
    }

    pub fn get(&self, x: usize, y: usize) -> &(RGBA, usize) {
        &self.pixels[y * self.width + x]
    }

    pub fn get_mut(&mut self, x: usize, y: usize) -> &mut (RGBA, usize) {
        &mut self.pixels[y * self.width + x]
    }

    pub fn get_rgba(&self, x: usize, y: usize) -> RGBA {
        self.pixels[y * self.width + x].0 / (self.pixels[y * self.width + x].1 as f64).max(1.0)
    }

    pub fn splat(&mut self, x: usize, y: usize, color: RGBA) {
        let (mut rgba, mut count) = self.get_mut(x, y);
        rgba += color;
        count += 1;
        self.pixels[y * self.width + x] = (rgba, count);
    }
}
