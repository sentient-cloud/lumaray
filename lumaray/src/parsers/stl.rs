use std::io::{Read, Seek};

use ultraviolet::Vec3;

#[derive(Debug, Clone, Copy)]
pub struct STLTriangle {
    pub vertices: [Vec3; 3],
    pub normal: Vec3,
    pub attribute: u16,
}

#[derive(Debug)]
pub struct STL {
    pub header: [u8; 80],
    pub triangles: Vec<STLTriangle>,
}

impl STL {
    pub fn new_from_bufreader<R>(reader: &mut R) -> Result<Self, std::io::Error>
    where
        R: Read + Seek,
    {
        let mut triangles = Vec::new();

        let mut header = [0u8; 80];
        reader.read_exact(&mut header)?;

        let mut num_tris_buf = [0u8; 4];
        reader.read_exact(&mut num_tris_buf)?;

        let num_tris = u32::from_le_bytes(num_tris_buf);

        if num_tris == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "STL is empty",
            ));
        }

        if num_tris > i32::MAX as u32 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "STL is too large",
            ));
        }

        for _ in 0..num_tris {
            let mut tri_buf = [0u8; 50];
            reader.read_exact(&mut tri_buf).map_err(|e| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("Failed to read triangle: {}", e),
                )
            })?;

            let normal = Vec3::new(
                f32::from_le_bytes([tri_buf[0], tri_buf[1], tri_buf[2], tri_buf[3]]),
                f32::from_le_bytes([tri_buf[4], tri_buf[5], tri_buf[6], tri_buf[7]]),
                f32::from_le_bytes([tri_buf[8], tri_buf[9], tri_buf[10], tri_buf[11]]),
            );

            let v1 = Vec3::new(
                f32::from_le_bytes([tri_buf[12], tri_buf[13], tri_buf[14], tri_buf[15]]),
                f32::from_le_bytes([tri_buf[16], tri_buf[17], tri_buf[18], tri_buf[19]]),
                f32::from_le_bytes([tri_buf[20], tri_buf[21], tri_buf[22], tri_buf[23]]),
            );

            let v2 = Vec3::new(
                f32::from_le_bytes([tri_buf[24], tri_buf[25], tri_buf[26], tri_buf[27]]),
                f32::from_le_bytes([tri_buf[28], tri_buf[29], tri_buf[30], tri_buf[31]]),
                f32::from_le_bytes([tri_buf[32], tri_buf[33], tri_buf[34], tri_buf[35]]),
            );

            let v3 = Vec3::new(
                f32::from_le_bytes([tri_buf[36], tri_buf[37], tri_buf[38], tri_buf[39]]),
                f32::from_le_bytes([tri_buf[40], tri_buf[41], tri_buf[42], tri_buf[43]]),
                f32::from_le_bytes([tri_buf[44], tri_buf[45], tri_buf[46], tri_buf[47]]),
            );

            let attribute = u16::from_le_bytes([tri_buf[48], tri_buf[49]]);

            triangles.push(STLTriangle {
                normal,
                vertices: [v1, v2, v3],
                attribute,
            });
        }

        Ok(STL { header, triangles })
    }

    pub fn write<W>(&self, writer: &mut W) -> Result<(), std::io::Error>
    where
        W: std::io::Write,
    {
        let header_str = "lumaray!!~~";
        let mut header = [0u8; 80];
        header[..header_str.len()].copy_from_slice(header_str.as_bytes());

        writer.write_all(&header)?;

        let num_tris = self.triangles.len() as u32;
        writer.write_all(&num_tris.to_le_bytes())?;

        for tri in &self.triangles {
            writer.write_all(&tri.normal.x.to_le_bytes())?;
            writer.write_all(&tri.normal.y.to_le_bytes())?;
            writer.write_all(&tri.normal.z.to_le_bytes())?;

            for v in &tri.vertices {
                writer.write_all(&v.x.to_le_bytes())?;
                writer.write_all(&v.y.to_le_bytes())?;
                writer.write_all(&v.z.to_le_bytes())?;
            }

            writer.write_all(&tri.attribute.to_le_bytes())?;
        }

        Ok(())
    }
}
