use super::error::Error;
use std::io::{Read, Write};
use std::path::Path;

pub struct QuantizationParams {
    pub mins: Vec<f32>,
    pub maxs: Vec<f32>,
    pub dimensions: usize,
}

impl QuantizationParams {
    /// Calibrate quantization parameters from a flat buffer of vectors.
    pub fn calibrate(vectors: &[f32], dimensions: usize) -> Self {
        let num_vectors = vectors.len() / dimensions;
        let mut mins = vec![f32::INFINITY; dimensions];
        let mut maxs = vec![f32::NEG_INFINITY; dimensions];

        for i in 0..num_vectors {
            let offset = i * dimensions;
            for d in 0..dimensions {
                let v = vectors[offset + d];
                if v < mins[d] {
                    mins[d] = v;
                }
                if v > maxs[d] {
                    maxs[d] = v;
                }
            }
        }

        // Handle edge case: if all values are the same in a dimension, set a small range
        for d in 0..dimensions {
            if (maxs[d] - mins[d]).abs() < f32::EPSILON {
                maxs[d] = mins[d] + 1.0;
            }
        }

        Self {
            mins,
            maxs,
            dimensions,
        }
    }

    /// Quantize a single vector into the output buffer.
    pub fn quantize(&self, vector: &[f32], output: &mut [i8]) {
        debug_assert_eq!(vector.len(), self.dimensions);
        debug_assert_eq!(output.len(), self.dimensions);
        for d in 0..self.dimensions {
            let range = self.maxs[d] - self.mins[d];
            let normalized = (vector[d] - self.mins[d]) / range;
            let scaled = (normalized * 254.0) - 127.0;
            output[d] = scaled.round().clamp(-127.0, 127.0) as i8;
        }
    }

    /// Quantize all vectors from a flat f32 buffer into a flat i8 buffer.
    pub fn quantize_all(&self, vectors: &[f32], dimensions: usize) -> Vec<i8> {
        let num_vectors = vectors.len() / dimensions;
        let mut output = vec![0i8; num_vectors * dimensions];
        for i in 0..num_vectors {
            let src_offset = i * dimensions;
            let dst_offset = i * dimensions;
            self.quantize(
                &vectors[src_offset..src_offset + dimensions],
                &mut output[dst_offset..dst_offset + dimensions],
            );
        }
        output
    }

    /// Write quantization parameters to a binary file.
    pub fn write_to_file(&self, path: &Path) -> Result<(), Error> {
        let mut file = std::fs::File::create(path)?;
        file.write_all(&(self.dimensions as u32).to_le_bytes())?;
        for &min in &self.mins {
            file.write_all(&min.to_le_bytes())?;
        }
        for &max in &self.maxs {
            file.write_all(&max.to_le_bytes())?;
        }
        file.sync_all()?;
        Ok(())
    }

    /// Read quantization parameters from a binary file.
    pub fn read_from_file(path: &Path) -> Result<Self, Error> {
        let mut file = std::fs::File::open(path)?;
        let mut buf4 = [0u8; 4];
        file.read_exact(&mut buf4)?;
        let dimensions = u32::from_le_bytes(buf4) as usize;

        let mut mins = vec![0.0f32; dimensions];
        for min in &mut mins {
            file.read_exact(&mut buf4)?;
            *min = f32::from_le_bytes(buf4);
        }

        let mut maxs = vec![0.0f32; dimensions];
        for max in &mut maxs {
            file.read_exact(&mut buf4)?;
            *max = f32::from_le_bytes(buf4);
        }

        Ok(Self {
            mins,
            maxs,
            dimensions,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_calibrate() {
        // 3 vectors of 2 dimensions
        let vectors = vec![
            1.0, 10.0, // vec 0
            3.0, 20.0, // vec 1
            5.0, 30.0, // vec 2
        ];
        let params = QuantizationParams::calibrate(&vectors, 2);
        assert_eq!(params.dimensions, 2);
        assert!((params.mins[0] - 1.0).abs() < 1e-6);
        assert!((params.maxs[0] - 5.0).abs() < 1e-6);
        assert!((params.mins[1] - 10.0).abs() < 1e-6);
        assert!((params.maxs[1] - 30.0).abs() < 1e-6);
    }

    #[test]
    fn test_quantize_range() {
        let vectors = vec![0.0, 1.0];
        let params = QuantizationParams::calibrate(&vectors, 1);
        let mut output = [0i8; 1];

        // min value -> -127
        params.quantize(&[0.0], &mut output);
        assert_eq!(output[0], -127);

        // max value -> 127
        params.quantize(&[1.0], &mut output);
        assert_eq!(output[0], 127);

        // midpoint -> 0
        params.quantize(&[0.5], &mut output);
        assert_eq!(output[0], 0);
    }

    #[test]
    fn test_quantize_all() {
        let vectors = vec![0.0, 1.0, 0.5, 0.5];
        let params = QuantizationParams::calibrate(&vectors, 2);
        let quantized = params.quantize_all(&vectors, 2);
        assert_eq!(quantized.len(), 4);
    }

    #[test]
    fn test_write_read_roundtrip() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("quantization.bin");

        let vectors = vec![1.0, 10.0, 5.0, 30.0];
        let params = QuantizationParams::calibrate(&vectors, 2);
        params.write_to_file(&path).unwrap();

        let loaded = QuantizationParams::read_from_file(&path).unwrap();
        assert_eq!(loaded.dimensions, params.dimensions);
        assert_eq!(loaded.mins, params.mins);
        assert_eq!(loaded.maxs, params.maxs);
    }

    #[test]
    fn test_calibrate_constant_dimension() {
        // All values same in dimension 0
        let vectors = vec![5.0, 10.0, 5.0, 20.0, 5.0, 30.0];
        let params = QuantizationParams::calibrate(&vectors, 2);
        // Should not produce zero range
        assert!(params.maxs[0] > params.mins[0]);
    }
}
