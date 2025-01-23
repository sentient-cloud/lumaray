use std::{fs::File, io::BufWriter, path::Path};

use super::{Film, RGBA};

pub struct Image {
    pub width: usize,
    pub height: usize,
    pub pixels: Vec<RGBA>,
}

impl Image {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            pixels: vec![RGBA::from(0.0); width * height],
        }
    }

    pub fn load_from_png<P>(path: P) -> Result<Self, png::DecodingError>
    where
        P: AsRef<Path>,
    {
        let file = File::open(path).unwrap();

        let mut decoder = png::Decoder::new(file);
        decoder.set_ignore_text_chunk(true);

        let mut reader = decoder.read_info()?;

        if reader.info().palette.is_some() {
            return Err(png::DecodingError::IoError(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Paletted images are not supported",
            )));
        }

        let mut image = Image::new(reader.info().width as usize, reader.info().height as usize);

        if reader.info().is_animated() {
            println!("Warning: APNG is not supported, only the first frame will be loaded");
        }

        let bytes_per_sample = match reader.info().bit_depth {
            png::BitDepth::Eight => 1,
            png::BitDepth::Sixteen => 2,
            _ => {
                return Err(png::DecodingError::IoError(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "Unsupported bit depth",
                )))
            }
        };

        let samples_per_pixel = match reader.info().color_type.samples() {
            3 => 3,
            4 => 4,
            _ => {
                return Err(png::DecodingError::IoError(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "Unsupported color type",
                )))
            }
        };

        let mut data = vec![0; reader.output_buffer_size()];
        reader.next_frame(&mut data)?;

        if data.len() < bytes_per_sample * samples_per_pixel * image.width * image.height {
            return Err(png::DecodingError::IoError(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Invalid image data size",
            )));
        }

        let mut position = 0;
        for y in 0..image.height {
            for x in 0..image.width {
                let pixel_slice = &data[position..position + bytes_per_sample * samples_per_pixel];
                position += bytes_per_sample * samples_per_pixel;

                let mut pixel = [1.0; 4];

                for i in 0..samples_per_pixel {
                    let pos = i * bytes_per_sample;
                    let sample_slice = &pixel_slice[pos..pos + bytes_per_sample];
                    let sample = match bytes_per_sample {
                        1 => u16::from_be_bytes([sample_slice[0], 0]),
                        2 => u16::from_be_bytes([sample_slice[0], sample_slice[1]]),
                        _ => unreachable!(),
                    };

                    pixel[i] = sample as f64 / 65535.0;
                }

                image.set(x, y, RGBA::from(pixel));
            }
        }

        Ok(image)
    }

    /// Sets the color of the pixel at the given coordinates.
    ///
    /// If the coordinates are out of bounds, this function does nothing.
    pub fn set(&mut self, x: usize, y: usize, color: RGBA) {
        if x >= self.width || y >= self.height {
            return;
        }

        *self.get_mut(x, y) = color;
    }

    /// Returns the color of the pixel at the given coordinates.
    ///
    /// If the coordinates are out of bounds, the coordinates are wrapped around.
    pub fn get(&self, x: usize, y: usize) -> &RGBA {
        let x = x % self.width;
        let y = y % self.height;
        &self.pixels[y * self.width + x]
    }

    /// Returns a mutable reference to the color of the pixel at the given coordinates.
    ///
    /// If the coordinates are out of bounds, the coordinates are wrapped around.
    pub fn get_mut(&mut self, x: usize, y: usize) -> &mut RGBA {
        let x = x % self.width;
        let y = y % self.height;
        &mut self.pixels[y * self.width + x]
    }

    /// Applies the given function for each pixel in the image.
    ///
    /// Example: `image.for_each_pixel(|_, _, color| color.sqrt())` will apply the square root operation to each pixel in the image.
    pub fn for_each_pixel(&mut self, f: impl Fn(usize, usize, RGBA) -> RGBA) {
        for y in 0..self.height {
            for x in 0..self.width {
                let pixel = self.get(x, y);
                self.set(x, y, f(x, y, *pixel));
            }
        }
    }

    /// Blits the given image onto this image at the given coordinates.
    pub fn blit(&mut self, x: usize, y: usize, image: &Image) {
        for j in 0..image.height {
            for i in 0..image.width {
                *self.get_mut(i + x, j + y) = *image.get(i, j);
            }
        }
    }

    pub fn output_as_png<P>(&self, path: P) -> Result<(), png::EncodingError>
    where
        P: AsRef<Path>,
    {
        let file = File::create(path).unwrap();
        let ref mut w = BufWriter::new(file);

        let mut encoder = png::Encoder::new(w, self.width as u32, self.height as u32);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Sixteen);

        encoder.add_text_chunk("meow".into(), ":3".into())?;

        let mut writer = encoder.write_header().unwrap();

        writer
            .write_image_data(
                &self
                    .pixels
                    .iter()
                    .map(|color| color.clamp(0.0, 1.0))
                    .map(|color| {
                        [
                            (color.r * 65535.0).floor() as u16,
                            (color.g * 65535.0).floor() as u16,
                            (color.b * 65535.0).floor() as u16,
                            (color.a * 65535.0).floor() as u16,
                        ]
                    })
                    .map(|[r, g, b, a]| {
                        [
                            (r >> 8) as u8,
                            (r & 0xff) as u8,
                            (g >> 8) as u8,
                            (g & 0xff) as u8,
                            (b >> 8) as u8,
                            (b & 0xff) as u8,
                            (a >> 8) as u8,
                            (a & 0xff) as u8,
                        ]
                    })
                    .flatten()
                    .collect::<Vec<_>>(),
            )
            .unwrap();

        writer.finish()
    }
}

impl From<Film> for Image {
    fn from(film: Film) -> Self {
        let mut image = Image::new(film.width(), film.height());
        for y in 0..film.height() {
            for x in 0..film.width() {
                *image.get_mut(x, y) = film.get_rgba(x, y);
            }
        }
        image
    }
}
