//! MD5 hash — RFC 1321, direct transcode of upstream `md5.c`.
//!
//! Used by the upstream compressor as the file-integrity checksum on
//! compressed output. Public-domain implementation; bit-exact match
//! to the upstream + RFC 1321 test vectors.

/// 64-element T table from RFC 1321 §3.4: `T[i] = ⌊2³² · |sin(i+1)|⌋`.
const T: [u32; 64] = [
    0xd76a_a478,
    0xe8c7_b756,
    0x2420_70db,
    0xc1bd_ceee,
    0xf57c_0faf,
    0x4787_c62a,
    0xa830_4613,
    0xfd46_9501,
    0x6980_98d8,
    0x8b44_f7af,
    0xffff_5bb1,
    0x895c_d7be,
    0x6b90_1122,
    0xfd98_7193,
    0xa679_438e,
    0x49b4_0821,
    0xf61e_2562,
    0xc040_b340,
    0x265e_5a51,
    0xe9b6_c7aa,
    0xd62f_105d,
    0x0244_1453,
    0xd8a1_e681,
    0xe7d3_fbc8,
    0x21e1_cde6,
    0xc337_07d6,
    0xf4d5_0d87,
    0x455a_14ed,
    0xa9e3_e905,
    0xfcef_a3f8,
    0x676f_02d9,
    0x8d2a_4c8a,
    0xfffa_3942,
    0x8771_f681,
    0x6d9d_6122,
    0xfde5_380c,
    0xa4be_ea44,
    0x4bde_cfa9,
    0xf6bb_4b60,
    0xbebf_bc70,
    0x289b_7ec6,
    0xeaa1_27fa,
    0xd4ef_3085,
    0x0488_1d05,
    0xd9d4_d039,
    0xe6db_99e5,
    0x1fa2_7cf8,
    0xc4ac_5665,
    0xf429_2244,
    0x432a_ff97,
    0xab94_23a7,
    0xfc93_a039,
    0x655b_59c3,
    0x8f0c_cc92,
    0xffef_f47d,
    0x8584_5dd1,
    0x6fa8_7e4f,
    0xfe2c_e6e0,
    0xa301_4314,
    0x4e08_11a1,
    0xf753_7e82,
    0xbd3a_f235,
    0x2ad7_d2bb,
    0xeb86_d391,
];

/// 64-element per-round shift table.
const S: [u32; 64] = [
    7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9,
    14, 20, 5, 9, 14, 20, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 6, 10, 15,
    21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
];

#[inline]
const fn round_f(x: u32, y: u32, z: u32) -> u32 {
    (x & y) | (!x & z)
}
#[inline]
const fn round_g(x: u32, y: u32, z: u32) -> u32 {
    (x & z) | (y & !z)
}
#[inline]
const fn round_h(x: u32, y: u32, z: u32) -> u32 {
    x ^ y ^ z
}
#[inline]
const fn round_i(x: u32, y: u32, z: u32) -> u32 {
    y ^ (x | !z)
}

fn transform(state: &mut [u32; 4], block: &[u8; 64]) {
    let mut m = [0u32; 16];
    for (i, chunk) in block.chunks_exact(4).enumerate() {
        m[i] = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }
    let (mut a, mut b, mut c, mut d) = (state[0], state[1], state[2], state[3]);
    for i in 0..64usize {
        let (f, g) = if i < 16 {
            (round_f(b, c, d), i)
        } else if i < 32 {
            (round_g(b, c, d), (5 * i + 1) % 16)
        } else if i < 48 {
            (round_h(b, c, d), (3 * i + 5) % 16)
        } else {
            (round_i(b, c, d), (7 * i) % 16)
        };
        let tmp = d;
        d = c;
        c = b;
        b = b.wrapping_add(
            a.wrapping_add(f)
                .wrapping_add(T[i])
                .wrapping_add(m[g])
                .rotate_left(S[i]),
        );
        a = tmp;
    }
    state[0] = state[0].wrapping_add(a);
    state[1] = state[1].wrapping_add(b);
    state[2] = state[2].wrapping_add(c);
    state[3] = state[3].wrapping_add(d);
}

/// Incremental MD5 hasher matching the upstream `md5_ctx_t` interface.
#[derive(Debug, Clone)]
pub struct Md5 {
    state: [u32; 4],
    count: u64,
    buffer: [u8; 64],
}

impl Default for Md5 {
    fn default() -> Self {
        Self::new()
    }
}

impl Md5 {
    /// Initialize a fresh hasher.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            state: [0x6745_2301, 0xefcd_ab89, 0x98ba_dcfe, 0x1032_5476],
            count: 0,
            buffer: [0u8; 64],
        }
    }

    /// Absorb `data` into the running hash state.
    pub fn update(&mut self, data: &[u8]) {
        let mut idx = (self.count & 63) as usize;
        self.count = self.count.wrapping_add(data.len() as u64);
        for &b in data {
            self.buffer[idx] = b;
            idx += 1;
            if idx == 64 {
                let block = self.buffer;
                transform(&mut self.state, &block);
                idx = 0;
            }
        }
    }

    /// Finalize and produce the 16-byte digest.
    #[must_use]
    pub fn finalize(mut self) -> [u8; 16] {
        let bits = self.count.wrapping_mul(8);
        self.update(&[0x80u8]);
        let pad_zero = [0u8; 1];
        while (self.count & 63) != 56 {
            self.update(&pad_zero);
        }
        self.update(&bits.to_le_bytes());

        let mut digest = [0u8; 16];
        for i in 0..4 {
            let bytes = self.state[i].to_le_bytes();
            digest[i * 4..i * 4 + 4].copy_from_slice(&bytes);
        }
        digest
    }
}

/// One-shot convenience: hash `data` and return the 16-byte digest.
#[must_use]
pub fn md5(data: &[u8]) -> [u8; 16] {
    let mut hasher = Md5::new();
    hasher.update(data);
    hasher.finalize()
}

#[cfg(test)]
mod tests {
    use super::md5;

    fn hex(d: [u8; 16]) -> String {
        let mut s = String::with_capacity(32);
        for b in d {
            s.push_str(&format!("{b:02x}"));
        }
        s
    }

    /// RFC 1321 §A.5 test suite.
    #[test]
    fn rfc1321_empty() {
        assert_eq!(hex(md5(b"")), "d41d8cd98f00b204e9800998ecf8427e");
    }
    #[test]
    fn rfc1321_a() {
        assert_eq!(hex(md5(b"a")), "0cc175b9c0f1b6a831c399e269772661");
    }
    #[test]
    fn rfc1321_abc() {
        assert_eq!(hex(md5(b"abc")), "900150983cd24fb0d6963f7d28e17f72");
    }
    #[test]
    fn rfc1321_message_digest() {
        assert_eq!(
            hex(md5(b"message digest")),
            "f96b697d7cb7938d525a2f31aaf161d0"
        );
    }
    #[test]
    fn rfc1321_alphabet() {
        assert_eq!(
            hex(md5(b"abcdefghijklmnopqrstuvwxyz")),
            "c3fcd3d76192e4007dfb496cca67e13b"
        );
    }
    #[test]
    fn rfc1321_alphanumeric() {
        assert_eq!(
            hex(md5(
                b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
            )),
            "d174ab98d277d9f5a5611c2c9f419d9f"
        );
    }
    #[test]
    fn rfc1321_long_digits() {
        assert_eq!(
            hex(md5(
                b"12345678901234567890123456789012345678901234567890123456789012345678901234567890"
            )),
            "57edf4a22be3c955ac49da2e2107b67a"
        );
    }
    #[test]
    fn incremental_matches_one_shot() {
        let data = b"the quick brown fox jumps over the lazy dog";
        let one_shot = md5(data);
        let mut h = super::Md5::new();
        for chunk in data.chunks(7) {
            h.update(chunk);
        }
        assert_eq!(one_shot, h.finalize());
    }
}
