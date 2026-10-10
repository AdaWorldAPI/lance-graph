//! Clifford fold. Bit planes in, parity out. A T gate is a refusal.
//!
//! Two-qubit tableau, four generators stored as bit rows. H, S, and CNOT
//! rewrite rows. Measurement of a stabilizer is the phase bit, not a
//! statevector. This is the proof of concept named in
//! `.grok/board/CLIFFORD_FOLD_POC.md`. It is not a general emulator.

/// A Pauli row: phase, then X and Z bits for two qubits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Row {
    r: u8,
    x: u8,
    z: u8,
}

impl Row {
    fn h(mut self, q: u8) -> Self {
        let bit = 1 << q;
        let x = (self.x & bit) != 0;
        let z = (self.z & bit) != 0;
        if x && z {
            self.r ^= 1;
        }
        self.x = (self.x & !bit) | (u8::from(z) << q);
        self.z = (self.z & !bit) | (u8::from(x) << q);
        self
    }

    fn s(mut self, q: u8) -> Self {
        let bit = 1 << q;
        if (self.x & bit) != 0 && (self.z & bit) != 0 {
            self.r ^= 1;
        }
        if (self.x & bit) != 0 {
            self.z ^= bit;
        }
        self
    }

    fn cnot(mut self, c: u8, t: u8) -> Self {
        let xc = (self.x >> c) & 1;
        let xt = (self.x >> t) & 1;
        let zc = (self.z >> c) & 1;
        let zt = (self.z >> t) & 1;
        self.r ^= xc & zt & (xt ^ zc ^ 1);
        if xc == 1 {
            self.x ^= 1 << t;
        }
        if zt == 1 {
            self.z ^= 1 << c;
        }
        self
    }
}

/// `|00⟩` as two Z stabilizers. Destabilizers are omitted: this PoC only
/// folds the stabilizer rows a measurement will read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tableau {
    rows: [Row; 2],
}

impl Tableau {
    pub fn zero() -> Self {
        Self {
            rows: [
                Row {
                    r: 0,
                    x: 0,
                    z: 0b01,
                },
                Row {
                    r: 0,
                    x: 0,
                    z: 0b10,
                },
            ],
        }
    }

    pub fn h(mut self, q: u8) -> Self {
        self.rows = [self.rows[0].h(q), self.rows[1].h(q)];
        self
    }

    pub fn s(mut self, q: u8) -> Self {
        self.rows = [self.rows[0].s(q), self.rows[1].s(q)];
        self
    }

    pub fn cnot(mut self, c: u8, t: u8) -> Self {
        self.rows = [self.rows[0].cnot(c, t), self.rows[1].cnot(c, t)];
        self
    }

    /// T is not a Clifford. The fold refuses rather than materializing amplitudes.
    pub fn t(self, _q: u8) -> Result<Self, &'static str> {
        let _ = self;
        Err("T gate leaves the Clifford fold")
    }

    /// Parity of a Pauli that is already a stabilizer row. `None` if it is not.
    pub fn parity(&self, x: u8, z: u8) -> Option<u8> {
        self.rows
            .iter()
            .find(|row| row.x == x && row.z == z)
            .map(|row| row.r)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bell_zz_parity_is_even() {
        let bell = Tableau::zero().h(0).cnot(0, 1);
        assert_eq!(
            bell.parity(0b00, 0b11),
            Some(0),
            "ZZ even on the Bell state"
        );
        assert_eq!(
            bell.parity(0b11, 0b00),
            Some(0),
            "XX even on the Bell state"
        );
    }

    #[test]
    fn t_gate_is_refused() {
        assert_eq!(Tableau::zero().t(0), Err("T gate leaves the Clifford fold"));
    }

    #[test]
    fn h_swaps_the_bit_planes() {
        let spun = Tableau::zero().h(0);
        assert_eq!(spun.parity(0b01, 0b00), Some(0), "H|0> is a +X stabilizer");
    }
}
