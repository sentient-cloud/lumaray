#[repr(align(16))]
#[derive(Debug, Clone, Copy)]
pub struct AlignedF64x2(pub [f64; 2]);

#[repr(align(32))]
#[derive(Debug, Clone, Copy)]
pub struct AlignedF64x4(pub [f64; 4]);

#[repr(align(64))]
#[derive(Debug, Clone, Copy)]
pub struct AlignedF64x8(pub [f64; 8]);

#[repr(align(64))]
#[derive(Debug, Clone, Copy)]
pub struct AlignedF64x16(pub [f64; 16]);

#[repr(align(64))]
#[derive(Debug, Clone, Copy)]
pub struct AlignedF32x16(pub [f32; 16]);

#[repr(align(64))]
#[derive(Debug, Clone, Copy)]
pub struct AlignedI32x16(pub [i32; 16]);
