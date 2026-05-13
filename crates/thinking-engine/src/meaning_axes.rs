//! 48 Canonical Meaning Axes + Gestalt I/Thou/It + Volition + Council.
//!
//! The full semantic coordinate system from ladybug-rs, migrated as
//! a clean module. The 17D QPL is an ICC profile lens on this space.
//! Pearson r = 0.9913 between Jina cosine and 48-axis Hamming similarity.

/// 48 bipolar meaning axes: (family, positive_pole, negative_pole).
pub const AXES_48: [(&str, &str, &str); 48] = [
    ("osgood","good","bad"),("osgood","strong","weak"),("osgood","active","passive"),
    ("physical","large","small"),("physical","heavy","light"),("physical","hard","soft"),
    ("physical","rough","smooth"),("physical","hot","cold"),("physical","wet","dry"),
    ("physical","fast","slow"),("physical","loud","quiet"),("physical","bright","dark"),
    ("physical","sharp","dull"),
    ("spatial","near","far"),("spatial","high","low"),("spatial","inside","outside"),
    ("spatial","new","old"),("spatial","permanent","temporary"),("spatial","sudden","gradual"),
    ("cognitive","simple","complex"),("cognitive","certain","uncertain"),
    ("cognitive","concrete","abstract"),("cognitive","familiar","unfamiliar"),
    ("cognitive","important","trivial"),
    ("emotional","happy","sad"),("emotional","calm","anxious"),("emotional","loving","hateful"),
    ("social","friendly","hostile"),("social","dominant","submissive"),("social","formal","informal"),
    ("evaluative","useful","useless"),("evaluative","beautiful","ugly"),
    ("evaluative","safe","dangerous"),("evaluative","clean","dirty"),
    ("evaluative","natural","artificial"),("evaluative","sacred","profane"),
    ("evaluative","real","imaginary"),("evaluative","whole","partial"),
    ("evaluative","open","closed"),("evaluative","free","constrained"),
    ("evaluative","ordered","chaotic"),("evaluative","alive","dead"),
    ("evaluative","growing","shrinking"),("evaluative","giving","taking"),
    ("evaluative","creating","destroying"),
    ("sensory","sweet","bitter"),("sensory","fragrant","foul"),("sensory","melodic","cacophonous"),
];

/// 48D axis activation. Each value in [-1.0, 1.0].
#[derive(Clone, Debug)]
pub struct AxisActivation { pub values: [f32; 48] }

impl AxisActivation {
    pub fn neutral() -> Self { Self { values: [0.0; 48] } }

    pub fn get(&self, label: &str) -> Option<f32> {
        AXES_48.iter().enumerate().find(|(_, (_, p, _))| *p == label).map(|(i, _)| self.values[i])
    }

    pub fn set(&mut self, label: &str, value: f32) {
        if let Some(i) = AXES_48.iter().position(|(_, p, _)| *p == label) {
            self.values[i] = value.clamp(-1.0, 1.0);
        }
    }

    /// Project 48D → 17D QPL (ICC profile rendering).
    pub fn to_qpl_17d(&self) -> [f32; 17] {
        let v = &self.values;
        [v[2], v[0], 1.0-v[25].abs(), (v[7].max(0.0)+v[26].max(0.0)).min(1.0),
         v[20], v[13], -v[21], v[9], -v[40], v[37], v[13]*v[26],
         -v[16], v[1], v[38], v[22], v[38], v[37]]
    }

    pub fn dominant_family(&self) -> &'static str {
        [("osgood",0,3),("physical",3,13),("spatial",13,19),("cognitive",19,24),
         ("emotional",24,27),("social",27,30),("evaluative",30,45),("sensory",45,48)]
        .iter()
        .max_by(|a, b| {
            let sa: f32 = self.values[a.1..a.2].iter().map(|v| v.abs()).sum();
            let sb: f32 = self.values[b.1..b.2].iter().map(|v| v.abs()).sum();
            sa.partial_cmp(&sb).unwrap()
        })
        .map(|f| f.0).unwrap_or("neutral")
    }
}

/// Processing fluidity — how freely computation flows between stages.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Viscosity {
    /// Fastest: streaming, no resistance.
    Water,
    /// Smooth: slight resistance, deliberate.
    Oil,
    /// Slow: significant resistance, iterating.
    Honey,
    /// Stalled: high resistance, blocked.
    Tar,
    /// Frozen: committed, no further flow.
    Ice,
}

impl Viscosity {
    pub fn speed_factor(&self) -> f32 {
        match self { Self::Water=>1.0, Self::Oil=>0.7, Self::Honey=>0.4, Self::Tar=>0.15, Self::Ice=>0.0 }
    }
}

/// Inner Council: Guardian (cautious) / Catalyst (curious) / Balanced.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Archetype { Guardian, Catalyst, Balanced }

#[derive(Clone, Debug)]
pub struct CouncilWeights { pub guardian: f32, pub catalyst: f32, pub balanced: f32 }

impl Default for CouncilWeights {
    fn default() -> Self { Self { guardian: 0.33, catalyst: 0.33, balanced: 0.34 } }
}

impl CouncilWeights {
    pub fn modulate(&self, raw: f32, free_energy: f32) -> f32 {
        self.guardian * raw * (1.0 - free_energy * 0.5)
        + self.catalyst * raw * (1.0 + free_energy * 0.5)
        + self.balanced * raw
    }

    pub fn shift_toward(&mut self, arch: Archetype, amount: f32) {
        let a = amount.clamp(0.0, 0.3);
        match arch {
            Archetype::Guardian => { self.guardian += a; self.catalyst -= a*0.5; self.balanced -= a*0.5; }
            Archetype::Catalyst => { self.catalyst += a; self.guardian -= a*0.5; self.balanced -= a*0.5; }
            Archetype::Balanced => { self.balanced += a; self.guardian -= a*0.5; self.catalyst -= a*0.5; }
        }
        let t = self.guardian + self.catalyst + self.balanced;
        if t > 0.01 { self.guardian /= t; self.catalyst /= t; self.balanced /= t; }
    }
}

/// Volition = free_energy × ghost_intensity × (1 - confidence) × rung_weight.
#[derive(Clone, Debug)]
pub struct VolitionalAct {
    pub atom: u16, pub free_energy: f32, pub ghost_intensity: f32,
    pub confidence: f32, pub rung_weight: f32, pub raw_score: f32, pub council_score: f32,
}

impl VolitionalAct {
    pub fn compute(atom: u16, fe: f32, ghost: f32, conf: f32,
        rung: &crate::cognitive_stack::RungLevel, council: &CouncilWeights) -> Self {
        let rw = 1.0 / (1.0 + rung.as_u8() as f32 * 0.3);
        let raw = fe * ghost * (1.0 - conf) * rw;
        Self { atom, free_energy: fe, ghost_intensity: ghost, confidence: conf,
            rung_weight: rw, raw_score: raw, council_score: council.modulate(raw, fe) }
    }
}

/// Gestalt role: I/Thou/It (Buber).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GestaltRole { Subject, Predicate, Object }

/// 3D resonance from unresolved perspectives. Disagreement IS awareness.
#[derive(Clone, Copy, Debug)]
pub struct HdrResonance { pub x: f32, pub y: f32, pub z: f32 }

impl HdrResonance {
    pub fn new(x: f32, y: f32, z: f32) -> Self { Self { x, y, z } }
    pub fn mean(&self) -> f32 { (self.x + self.y + self.z) / 3.0 }
    pub fn spread(&self) -> f32 { self.x.max(self.y).max(self.z) - self.x.min(self.y).min(self.z) }
    pub fn variance(&self) -> f32 {
        let m = self.mean();
        ((self.x-m).powi(2) + (self.y-m).powi(2) + (self.z-m).powi(2)) / 3.0
    }
    pub fn dominant(&self) -> Archetype {
        if self.x >= self.y && self.x >= self.z { Archetype::Guardian }
        else if self.y >= self.z { Archetype::Catalyst }
        else { Archetype::Balanced }
    }
    pub fn is_epiphany(&self) -> bool { self.spread() > 0.4 }
    pub fn is_unanimous(&self) -> bool { self.variance() < 0.01 }
    pub fn gate(&self) -> crate::cognitive_stack::GateState {
        crate::cognitive_stack::GateState::from_sd(self.variance().sqrt())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn axes_48_count() { assert_eq!(AXES_48.len(), 48); }
    #[test] fn axis_get_set() {
        let mut a = AxisActivation::neutral(); a.set("good", 0.8);
        assert!((a.get("good").unwrap() - 0.8).abs() < 0.01);
    }
    #[test] fn qpl_projection() {
        let mut a = AxisActivation::neutral(); a.set("active", 0.9); a.set("good", 0.7);
        let q = a.to_qpl_17d();
        assert!(q[0] > 0.5); assert!(q[1] > 0.5);
    }
    #[test] fn dominant_emotional() {
        let mut a = AxisActivation::neutral();
        a.set("happy", 0.9); a.set("calm", 0.8); a.set("loving", 0.7);
        assert_eq!(a.dominant_family(), "emotional");
    }
    #[test] fn viscosity_order() {
        assert!(Viscosity::Water.speed_factor() > Viscosity::Tar.speed_factor());
    }
    #[test] fn council_shift_renorm() {
        let mut c = CouncilWeights::default(); c.shift_toward(Archetype::Catalyst, 0.2);
        assert!(c.catalyst > c.guardian);
        assert!((c.guardian + c.catalyst + c.balanced - 1.0).abs() < 0.01);
    }
    #[test] fn hdr_epiphany() {
        let r = HdrResonance::new(0.9, 0.2, 0.3);
        assert!(r.is_epiphany()); assert_eq!(r.dominant(), Archetype::Guardian);
    }
    #[test] fn hdr_unanimous_flows() {
        let r = HdrResonance::new(0.5, 0.5, 0.5);
        assert!(r.is_unanimous());
        assert_eq!(r.gate(), crate::cognitive_stack::GateState::Flow);
    }
    #[test] fn volition_formula() {
        let c = CouncilWeights::default();
        let a = VolitionalAct::compute(42, 0.8, 0.6, 0.3, &crate::cognitive_stack::RungLevel::Surface, &c);
        assert!(a.raw_score > 0.3 && a.raw_score < 0.4);
    }
}
