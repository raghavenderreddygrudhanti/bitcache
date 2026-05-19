//! Binary quantization: float vectors → packed bit arrays.
//!
//! Sign-bit quantization: each dimension becomes 1 if positive, 0 otherwise.
//! Packed into u8 arrays for efficient XOR + popcount.
//!
//! A 1536-dim float32 vector (6144 bytes) becomes 192 bytes (32x compression).

/// Quantize a single float vector to packed binary representation.
///
/// Each coordinate maps to one bit: 1 if positive, 0 otherwise.
/// Packed MSB-first into bytes (matching numpy.packbits behavior).
pub fn quantize(vector: &[f32]) -> Vec<u8> {
    let n_bytes = (vector.len() + 7) / 8;
    let mut codes = vec![0u8; n_bytes];

    for (i, &val) in vector.iter().enumerate() {
        if val > 0.0 {
            codes[i / 8] |= 1 << (7 - (i % 8));
        }
    }

    codes
}

/// Quantize a batch of float vectors.
///
/// `vectors` is a flat slice of length `n * dim`.
/// Returns a flat Vec<u8> of length `n * n_bytes` where `n_bytes = ceil(dim/8)`.
pub fn quantize_batch(vectors: &[f32], n: usize, dim: usize) -> Vec<u8> {
    let n_bytes = (dim + 7) / 8;
    let mut codes = vec![0u8; n * n_bytes];

    for i in 0..n {
        let vec_start = i * dim;
        let code_start = i * n_bytes;
        for j in 0..dim {
            if vectors[vec_start + j] > 0.0 {
                codes[code_start + j / 8] |= 1 << (7 - (j % 8));
            }
        }
    }

    codes
}

/// L2-normalize a vector in place. Returns the original norm.
pub fn normalize(vector: &mut [f32]) -> f32 {
    let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-10 {
        for x in vector.iter_mut() {
            *x /= norm;
        }
    }
    norm
}

/// L2-normalize a batch of vectors in place (flat layout, n * dim).
pub fn normalize_batch(vectors: &mut [f32], n: usize, dim: usize) -> Vec<f32> {
    let mut norms = Vec::with_capacity(n);
    for i in 0..n {
        let start = i * dim;
        let end = start + dim;
        let norm: f32 = vectors[start..end].iter().map(|x| x * x).sum::<f32>().sqrt();
        norms.push(norm);
        if norm > 1e-10 {
            for x in vectors[start..end].iter_mut() {
                *x /= norm;
            }
        }
    }
    norms
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantize_basic() {
        let v = vec![1.0, -1.0, 0.5, -0.5, 0.1, -0.1, 0.0, 0.9];
        let code = quantize(&v);
        // Positive: indices 0, 2, 4, 7 → bits 7,5,3,0 of byte 0
        // Expected: 10101001 = 0xA9... wait let's be precise
        // Bit layout (MSB first): bit7=v[0]=1, bit6=v[1]=0, bit5=v[2]=1, bit4=v[3]=0
        //                          bit3=v[4]=1, bit2=v[5]=0, bit1=v[6]=0, bit0=v[7]=1
        // = 0b10101001 = 0xA9 = 169
        assert_eq!(code.len(), 1);
        assert_eq!(code[0], 0b10101001);
    }

    #[test]
    fn test_quantize_multi_byte() {
        let v = vec![1.0; 16]; // all positive
        let code = quantize(&v);
        assert_eq!(code.len(), 2);
        assert_eq!(code[0], 0xFF);
        assert_eq!(code[1], 0xFF);
    }

    #[test]
    fn test_normalize() {
        let mut v = vec![3.0, 4.0];
        let norm = normalize(&mut v);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((v[0] - 0.6).abs() < 1e-6);
        assert!((v[1] - 0.8).abs() < 1e-6);
    }
}
