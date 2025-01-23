use itertools::Itertools;

use super::RGBA;

#[derive(Debug, Clone)]
pub struct Film {
    width: usize,
    height: usize,
    pixels: Vec<(RGBA, usize)>,
}

unsafe impl Send for Film {}
unsafe impl Sync for Film {}

impl Film {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            pixels: vec![(RGBA::from(0.0), 0); width * height],
        }
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
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
        let (rgba, count) = self.get(x, y);
        let rgba = *rgba + color;
        let count = count + 1;
        self.pixels[y * self.width + x] = (rgba, count);
    }

    pub fn fill(&mut self, color: RGBA) {
        for y in 0..self.height {
            for x in 0..self.width {
                self.splat(x, y, color);
            }
        }
    }

    /// Splats a chunk of film onto this film.
    ///
    /// The FilmChunk must fit within the bounds of this film, or this will panic.
    pub fn splat_chunk(&mut self, chunk: FilmChunk) {
        for y in 0..chunk.height() {
            for x in 0..chunk.width() {
                let (rgba, count) = self.get(chunk.x + x, chunk.y + y);
                let (sub_rgba, sub_count) = chunk.film.get(x, y);
                self.pixels[(y + chunk.y) * self.width + (x + chunk.x)] =
                    (*rgba + *sub_rgba, *count + *sub_count);
            }
        }
    }

    /// Subdivides the film into chunks, chunks_x is an approximate number of chunks in the x direction.
    /// The chunks will be as square as possible.
    pub fn subdivide(&self, chunks_x: usize) -> Vec<FilmChunk> {
        if chunks_x <= 1 {
            return vec![FilmChunk {
                x: 0,
                y: 0,
                film: Film::new(self.width, self.height),
            }];
        }

        let find_optimal_size = |width: usize, divisions: usize| {
            let initial_width = width / divisions;

            let mut best_size = 0;
            let mut best_score = usize::MAX;

            for size in (initial_width / 2)..(initial_width * 3 / 2) {
                let score_x = 1 + size - (self.width % size);
                let score_y = 1 + size - (self.height % size);
                let score_z = 1 + (initial_width as i64 - size as i64).abs() as usize;

                let score = score_x * score_y + score_z;

                if score < best_score {
                    best_score = score;
                    best_size = size;
                }
            }

            best_size
        };

        let chunk_size = find_optimal_size(self.width, chunks_x);

        let last_size_x = self.width % chunk_size;
        let last_size_y = self.height % chunk_size;

        let mut chunks = Vec::new();

        let mut x = 0;
        let mut y = 0;

        loop {
            let width = if x + chunk_size > self.width {
                last_size_x
            } else {
                chunk_size
            };

            let height = if y + chunk_size > self.height {
                last_size_y
            } else {
                chunk_size
            };

            chunks.push(FilmChunk {
                x,
                y,
                film: Film::new(width, height),
            });

            x += chunk_size;

            if x >= self.width {
                x = 0;
                y += chunk_size;

                if y >= self.height {
                    break;
                }
            }
        }

        chunks
    }
}

#[derive(Debug, Clone)]
pub struct FilmChunk {
    x: usize,
    y: usize,
    film: Film,
}

impl FilmChunk {
    pub fn x(&self) -> usize {
        self.x
    }

    pub fn y(&self) -> usize {
        self.y
    }

    pub fn splat(&mut self, x: usize, y: usize, color: RGBA) {
        let x = x - self.x;
        let y = y - self.y;
        self.film.splat(x, y, color);
    }

    pub fn fill(&mut self, color: RGBA) {
        for y in 0..self.height() {
            for x in 0..self.width() {
                self.splat(self.x + x, self.y + y, color);
            }
        }
    }

    pub fn width(&self) -> usize {
        self.film.width
    }

    pub fn height(&self) -> usize {
        self.film.height
    }

    pub fn iter(&self) -> impl Iterator<Item = (usize, usize)> {
        (self.y..(self.y + self.height()))
            .cartesian_product(self.x..(self.x + self.width()))
            .map(|(y, x)| (x, y))
    }
}

unsafe impl Send for FilmChunk {}
unsafe impl Sync for FilmChunk {}
