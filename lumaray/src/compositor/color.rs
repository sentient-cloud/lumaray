use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

use ultraviolet::DVec3;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RGBA {
    pub r: f64,
    pub g: f64,
    pub b: f64,
    pub a: f64,
}

impl RGBA {
    pub fn new(r: f64, g: f64, b: f64, a: f64) -> Self {
        Self { r, g, b, a }
    }

    pub fn new_normal(normal: DVec3) -> Self {
        Self {
            r: (normal.x + 1.0) / 2.0,
            g: (normal.y + 1.0) / 2.0,
            b: (normal.z + 1.0) / 2.0,
            a: 1.0,
        }
    }

    /// Create a new RGBA color from a 24-bit hex value.
    ///
    /// hex: 0xRRGGBB
    pub fn from_hex24(hex: u32) -> Self {
        let r = ((hex >> 16) & 0xff) as f64 / 255.0;
        let g = ((hex >> 8) & 0xff) as f64 / 255.0;
        let b = (hex & 0xff) as f64 / 255.0;
        let a = 1.0;
        Self { r, g, b, a }
    }

    /// Convert the color to a 24-bit hex value.
    ///
    /// Returns: 0xRRGGBB
    pub fn to_hex24(&self) -> u32 {
        let r = (self.r * 255.0).clamp(0.0, 255.0) as u32;
        let g = (self.g * 255.0).clamp(0.0, 255.0) as u32;
        let b = (self.b * 255.0).clamp(0.0, 255.0) as u32;
        (r << 16) | (g << 8) | b
    }

    /// Create a new RGBA color from a 32-bit hex value.
    ///
    /// hex: 0xAARRGGBB
    pub fn from_hex32(hex: u32) -> Self {
        let r = ((hex >> 16) & 0xff) as f64 / 255.0;
        let g = ((hex >> 8) & 0xff) as f64 / 255.0;
        let b = (hex & 0xff) as f64 / 255.0;
        let a = ((hex >> 24) & 0xff) as f64 / 255.0;
        Self { r, g, b, a }
    }

    /// Convert the color to a 32-bit hex value.
    ///
    /// Returns: 0xAARRGGBB
    pub fn to_hex32(&self) -> u32 {
        let r = (self.r * 255.0).clamp(0.0, 255.0) as u32;
        let g = (self.g * 255.0).clamp(0.0, 255.0) as u32;
        let b = (self.b * 255.0).clamp(0.0, 255.0) as u32;
        let a = (self.a * 255.0).clamp(0.0, 255.0) as u32;
        (a << 24) | (r << 16) | (g << 8) | b
    }

    /// Returns the linear luminance of the color.
    pub fn luminance(&self) -> f64 {
        0.2126 * self.r + 0.7152 * self.g + 0.0722 * self.b
    }

    /// Returns the square root of the color components.
    pub fn sqrt(&self) -> Self {
        Self::new(self.r.sqrt(), self.g.sqrt(), self.b.sqrt(), self.a.sqrt())
    }

    /// Returns the absolute value of the color components.
    pub fn abs(&self) -> Self {
        Self::new(self.r.abs(), self.g.abs(), self.b.abs(), self.a.abs())
    }

    /// Raises the color components to the power of exp.
    pub fn pow(&self, exp: f64) -> Self {
        Self::new(
            self.r.powf(exp),
            self.g.powf(exp),
            self.b.powf(exp),
            self.a.powf(exp),
        )
    }

    /// Clamps the color components to the range [min, max].
    pub fn clamp(&self, min: f64, max: f64) -> Self {
        Self::new(
            self.r.clamp(min, max),
            self.g.clamp(min, max),
            self.b.clamp(min, max),
            self.a.clamp(min, max),
        )
    }

    /// Copies the alpha of other to self
    ///
    /// This is useful when applying operations such as sqrt, abs, pow.
    ///
    /// Example: `color = color.copy_alpha(color.sqrt(), color)` will apply the square root operation to the color, but keep the original alpha channel.
    pub fn copy_alpha(this: Self, other: Self) -> Self {
        Self::new(this.r, this.g, this.b, other.a)
    }
}

impl Add for RGBA {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self::new(
            self.r + other.r,
            self.g + other.g,
            self.b + other.b,
            self.a + other.a,
        )
    }
}

impl Add<f64> for RGBA {
    type Output = Self;

    fn add(self, other: f64) -> Self {
        Self::new(
            self.r + other,
            self.g + other,
            self.b + other,
            self.a + other,
        )
    }
}

impl AddAssign for RGBA {
    fn add_assign(&mut self, other: Self) {
        *self = *self + other;
    }
}

impl AddAssign<f64> for RGBA {
    fn add_assign(&mut self, other: f64) {
        *self = *self + other;
    }
}

impl Sub for RGBA {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self::new(
            self.r - other.r,
            self.g - other.g,
            self.b - other.b,
            self.a - other.a,
        )
    }
}

impl Sub<f64> for RGBA {
    type Output = Self;

    fn sub(self, other: f64) -> Self {
        Self::new(
            self.r - other,
            self.g - other,
            self.b - other,
            self.a - other,
        )
    }
}

impl SubAssign for RGBA {
    fn sub_assign(&mut self, other: Self) {
        *self = *self - other;
    }
}

impl SubAssign<f64> for RGBA {
    fn sub_assign(&mut self, other: f64) {
        *self = *self - other;
    }
}

impl Mul for RGBA {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        Self::new(
            self.r * other.r,
            self.g * other.g,
            self.b * other.b,
            self.a * other.a,
        )
    }
}

impl Mul<f64> for RGBA {
    type Output = Self;

    fn mul(self, other: f64) -> Self {
        Self::new(
            self.r * other,
            self.g * other,
            self.b * other,
            self.a * other,
        )
    }
}

impl MulAssign for RGBA {
    fn mul_assign(&mut self, other: Self) {
        *self = *self * other;
    }
}

impl MulAssign<f64> for RGBA {
    fn mul_assign(&mut self, other: f64) {
        *self = *self * other;
    }
}

impl Div for RGBA {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        Self::new(
            self.r / other.r,
            self.g / other.g,
            self.b / other.b,
            self.a / other.a,
        )
    }
}

impl Div<f64> for RGBA {
    type Output = Self;

    fn div(self, other: f64) -> Self {
        Self::new(
            self.r / other,
            self.g / other,
            self.b / other,
            self.a / other,
        )
    }
}

impl DivAssign for RGBA {
    fn div_assign(&mut self, other: Self) {
        *self = *self / other;
    }
}

impl DivAssign<f64> for RGBA {
    fn div_assign(&mut self, other: f64) {
        *self = *self / other;
    }
}

impl num_traits::SaturatingAdd for RGBA {
    fn saturating_add(&self, v: &Self) -> Self {
        Self::new(
            (self.r + v.r).min(1.0),
            (self.g + v.r).min(1.0),
            (self.b + v.r).min(1.0),
            (self.a + v.r).min(1.0),
        )
    }
}

impl num_traits::SaturatingSub for RGBA {
    fn saturating_sub(&self, v: &Self) -> Self {
        Self::new(
            (self.r - v.r).max(0.0),
            (self.g - v.g).max(0.0),
            (self.b - v.b).max(0.0),
            (self.a - v.a).max(0.0),
        )
    }
}

impl num_traits::SaturatingMul for RGBA {
    fn saturating_mul(&self, v: &Self) -> Self {
        Self::new(
            (self.r * v.r).clamp(0.0, 1.0),
            (self.g * v.g).clamp(0.0, 1.0),
            (self.b * v.b).clamp(0.0, 1.0),
            (self.a * v.a).clamp(0.0, 1.0),
        )
    }
}

impl Default for RGBA {
    fn default() -> Self {
        Self::new(0.0, 0.0, 0.0, 1.0)
    }
}

impl From<(f64, f64, f64, f64)> for RGBA {
    /// Convert a tuple of f64 values to an RGBA color.
    fn from((r, g, b, a): (f64, f64, f64, f64)) -> Self {
        Self::new(r, g, b, a)
    }
}

impl From<[f64; 4]> for RGBA {
    /// Convert an array of f64 values to an RGBA color.
    fn from([r, g, b, a]: [f64; 4]) -> Self {
        Self::new(r, g, b, a)
    }
}

impl From<RGBA> for (f64, f64, f64, f64) {
    /// Convert an RGBA color to a tuple of f64 values.
    fn from(rgba: RGBA) -> Self {
        (rgba.r, rgba.g, rgba.b, rgba.a)
    }
}

impl From<RGBA> for [f64; 4] {
    /// Convert an RGBA color to an array of f64 values.
    fn from(rgba: RGBA) -> Self {
        [rgba.r, rgba.g, rgba.b, rgba.a]
    }
}

impl From<u32> for RGBA {
    /// Convert a u32 to an RGBA color.
    ///
    /// Note: The u32 is expected to be in the format 0xAARRGGBB, but 24 bit colors are also supported.
    /// If the alpha channel is 0, it is assumed to be fully opaque.
    fn from(hex: u32) -> Self {
        let mut res = Self::from_hex32(hex);
        if res.a == 0.0 {
            res.a = 1.0;
            res
        } else {
            res
        }
    }
}

impl From<RGBA> for u32 {
    /// Convert an RGBA color to a u32.
    fn from(rgba: RGBA) -> Self {
        rgba.to_hex32()
    }
}

impl From<(u32, f64)> for RGBA {
    /// Convert a tuple of a u32 and f64 to an RGBA color.
    fn from((hex, a): (u32, f64)) -> Self {
        let mut rgba = RGBA::from_hex24(hex);
        rgba.a = a;
        rgba
    }
}

impl From<RGBA> for (u32, f64) {
    /// Convert an RGBA color to a tuple of a u32 and f64.
    fn from(rgba: RGBA) -> Self {
        (rgba.to_hex24(), rgba.a)
    }
}

impl From<f64> for RGBA {
    /// Convert a f64 to an RGBA color.
    fn from(v: f64) -> Self {
        Self::new(v, v, v, 1.0)
    }
}

impl From<(f64, f64, f64)> for RGBA {
    /// Convert a tuple of three f64 values to an RGBA color.
    fn from((r, g, b): (f64, f64, f64)) -> Self {
        Self::new(r, g, b, 1.0)
    }
}

impl From<[f64; 3]> for RGBA {
    /// Convert an array of three f64 values to an RGBA color.
    fn from([r, g, b]: [f64; 3]) -> Self {
        Self::new(r, g, b, 1.0)
    }
}

impl From<RGBA> for (f64, f64, f64) {
    /// Convert an RGBA color to a tuple of three f64 values.
    fn from(rgba: RGBA) -> Self {
        (rgba.r, rgba.g, rgba.b)
    }
}

impl From<RGBA> for [f64; 3] {
    /// Convert an RGBA color to an array of three f64 values.
    fn from(rgba: RGBA) -> Self {
        [rgba.r, rgba.g, rgba.b]
    }
}
