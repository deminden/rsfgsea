use serde::{Deserialize, Serialize};
use std::fmt;
use std::path::PathBuf;
use std::str::FromStr;

pub const DECOR_BALANCED_ALPHA: f64 = 60.0;
pub const DECOR_BALANCED_THRESHOLD_TAU: f64 = 0.04;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ScoreType {
    Std,
    Pos,
    Neg,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum EnrichmentMethod {
    Classic,
    Decor,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct DecorOptions {
    pub alpha: f64,
    pub cache_path: Option<PathBuf>,
    pub expression_path: Option<PathBuf>,
    pub expression_has_header: bool,
    pub cache_mode: DecorCacheMode,
    pub correlation: DecorCorrelation,
    pub redundancy: DecorRedundancy,
    pub weight_formula: DecorWeightFormula,
    pub gamma: f64,
    pub threshold_tau: f64,
    pub penalty_floor: f64,
    pub scale_epsilon: f64,
}

impl Default for DecorOptions {
    fn default() -> Self {
        Self {
            alpha: DECOR_BALANCED_ALPHA,
            cache_path: None,
            expression_path: None,
            expression_has_header: true,
            cache_mode: DecorCacheMode::Auto,
            correlation: DecorCorrelation::Pearson,
            redundancy: DecorRedundancy::PositiveMean,
            weight_formula: DecorWeightFormula::ThresholdRational,
            gamma: 1.0,
            threshold_tau: DECOR_BALANCED_THRESHOLD_TAU,
            penalty_floor: 0.0,
            scale_epsilon: 1e-12,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum DecorPreset {
    Sensitive,
    Balanced,
    Specific,
    Strict,
}

impl DecorPreset {
    pub const SUPPORTED: &'static [&'static str] = &["sensitive", "balanced", "specific", "strict"];
}

impl fmt::Display for DecorPreset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let value = match self {
            DecorPreset::Sensitive => "sensitive",
            DecorPreset::Balanced => "balanced",
            DecorPreset::Specific => "specific",
            DecorPreset::Strict => "strict",
        };
        write!(f, "{value}")
    }
}

impl FromStr for DecorPreset {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.to_ascii_lowercase().as_str() {
            "sensitive" => Ok(Self::Sensitive),
            "balanced" => Ok(Self::Balanced),
            "specific" => Ok(Self::Specific),
            "strict" => Ok(Self::Strict),
            other => Err(format!(
                "Invalid decor preset '{other}'. Expected one of: {}.",
                Self::SUPPORTED.join(", ")
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct DecorPresetResolution {
    pub preset: DecorPreset,
    pub weight_formula: DecorWeightFormula,
    pub alpha: f64,
    pub threshold_tau: f64,
    pub gamma: f64,
    pub penalty_floor: f64,
    pub target_median_penalty: Option<f64>,
}

impl DecorOptions {
    // Presets are the public tuning contract for decor: they set formula knobs,
    // while cache construction and permutation calibration stay reproducible.
    pub fn apply_preset(&mut self, preset: DecorPreset) -> DecorPresetResolution {
        let resolved = resolve_decor_preset(preset);
        self.weight_formula = resolved.weight_formula;
        self.alpha = resolved.alpha;
        self.threshold_tau = resolved.threshold_tau;
        self.gamma = resolved.gamma;
        self.penalty_floor = resolved.penalty_floor;
        resolved
    }

    pub fn apply_stringency(
        &mut self,
        stringency: f64,
    ) -> Result<DecorStringencyResolution, String> {
        let resolved = resolve_decor_stringency(stringency)?;
        self.weight_formula = resolved.preset_resolution.weight_formula;
        self.alpha = resolved.preset_resolution.alpha;
        self.threshold_tau = resolved.preset_resolution.threshold_tau;
        self.gamma = resolved.preset_resolution.gamma;
        self.penalty_floor = resolved.preset_resolution.penalty_floor;
        Ok(resolved)
    }
}

// The release presets are intentionally named by behavior, not formula internals.
pub fn resolve_decor_preset(preset: DecorPreset) -> DecorPresetResolution {
    match preset {
        DecorPreset::Sensitive => DecorPresetResolution {
            preset,
            weight_formula: DecorWeightFormula::RawRational,
            alpha: 22.0,
            threshold_tau: 0.0,
            gamma: 1.0,
            penalty_floor: 0.0,
            target_median_penalty: None,
        },
        DecorPreset::Balanced => DecorPresetResolution {
            preset,
            weight_formula: DecorWeightFormula::ThresholdRational,
            alpha: DECOR_BALANCED_ALPHA,
            threshold_tau: DECOR_BALANCED_THRESHOLD_TAU,
            gamma: 1.0,
            penalty_floor: 0.0,
            target_median_penalty: None,
        },
        DecorPreset::Specific => DecorPresetResolution {
            preset,
            weight_formula: DecorWeightFormula::ThresholdRational,
            alpha: 65.0,
            threshold_tau: 0.05,
            gamma: 1.0,
            penalty_floor: 0.0,
            target_median_penalty: None,
        },
        DecorPreset::Strict => {
            let target_median_penalty: f64 = 0.10;
            DecorPresetResolution {
                preset,
                weight_formula: DecorWeightFormula::ExpScaled,
                alpha: -target_median_penalty.ln(),
                threshold_tau: 0.0,
                gamma: 1.0,
                penalty_floor: 0.0,
                target_median_penalty: Some(target_median_penalty),
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DecorStringencyResolution {
    pub stringency: f64,
    pub band: &'static str,
    pub preset_resolution: DecorPresetResolution,
}

// Stringency is a convenience ladder over presets. It avoids interpolating
// formulas, which would make runs harder to audit and reproduce.
pub fn resolve_decor_stringency(stringency: f64) -> Result<DecorStringencyResolution, String> {
    if !stringency.is_finite() || !(0.0..=100.0).contains(&stringency) {
        return Err(format!(
            "Invalid decor stringency '{stringency}'. Expected a finite value from 0 to 100."
        ));
    }

    let (preset, band) = if stringency < 35.0 {
        (DecorPreset::Sensitive, "0 <= stringency < 35")
    } else if stringency < 65.0 {
        (DecorPreset::Balanced, "35 <= stringency < 65")
    } else if stringency < 85.0 {
        (DecorPreset::Specific, "65 <= stringency < 85")
    } else {
        (DecorPreset::Strict, "85 <= stringency <= 100")
    };

    Ok(DecorStringencyResolution {
        stringency,
        band,
        preset_resolution: resolve_decor_preset(preset),
    })
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum DecorCacheMode {
    Auto,
    Reuse,
    Rebuild,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum DecorCorrelation {
    Pearson,
    Spearman,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum DecorRedundancy {
    PositiveMean,
    AbsMean,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum DecorWeightFormula {
    RawRational,
    ScaledRational,
    Q75ScaledRational,
    ExpScaled,
    OddsRational,
    ThresholdRational,
    QuantileRational,
    FloorScaledRational,
    PowerRetention,
}

impl DecorWeightFormula {
    pub const SUPPORTED: &'static [&'static str] =
        &["raw-rational", "exp-scaled", "threshold-rational"];
}

impl fmt::Display for DecorWeightFormula {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let value = match self {
            DecorWeightFormula::RawRational => "raw-rational",
            DecorWeightFormula::ScaledRational => "scaled-rational",
            DecorWeightFormula::Q75ScaledRational => "q75-scaled-rational",
            DecorWeightFormula::ExpScaled => "exp-scaled",
            DecorWeightFormula::OddsRational => "odds-rational",
            DecorWeightFormula::ThresholdRational => "threshold-rational",
            DecorWeightFormula::QuantileRational => "quantile-rational",
            DecorWeightFormula::FloorScaledRational => "floor-scaled-rational",
            DecorWeightFormula::PowerRetention => "power-retention",
        };
        write!(f, "{value}")
    }
}

impl FromStr for DecorWeightFormula {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.to_ascii_lowercase().as_str() {
            "raw-rational" => Ok(Self::RawRational),
            "exp-scaled" => Ok(Self::ExpScaled),
            "threshold-rational" => Ok(Self::ThresholdRational),
            other => Err(format!(
                "Invalid decor weight formula '{other}'. Expected one of: {}.",
                Self::SUPPORTED.join(", ")
            )),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankedList {
    pub genes: Vec<String>,
    pub scores: Vec<f64>,
}

impl RankedList {
    pub fn new(genes: Vec<String>, scores: Vec<f64>) -> Self {
        // Enforce sorting by score descending
        let mut indices: Vec<usize> = (0..genes.len()).collect();
        indices.sort_by(|&a, &b| {
            scores[b]
                .partial_cmp(&scores[a])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.cmp(&b))
        });

        let sorted_genes = indices.iter().map(|&i| genes[i].clone()).collect();
        let sorted_scores = indices.iter().map(|&i| scores[i]).collect();

        Self {
            genes: sorted_genes,
            scores: sorted_scores,
        }
    }

    pub fn len(&self) -> usize {
        self.genes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.genes.is_empty()
    }

    pub fn prepare(&self, gsea_param: f64) -> (Vec<f64>, Vec<i64>, i64) {
        let abs_scores: Vec<f64> = self
            .scores
            .iter()
            .map(|&s| s.abs().powf(gsea_param))
            .collect();
        // Scaling as in fgsea
        let sum_abs: f64 = abs_scores.iter().sum();
        let mut scale_coeff = (1i64 << 30) as f64 / sum_abs;
        if scale_coeff >= 1.0 {
            scale_coeff = scale_coeff.floor();
        }
        let scaled_scores: Vec<i64> = abs_scores
            .iter()
            .map(|&s| (s * scale_coeff).round_ties_even() as i64)
            .collect();
        let actual_sum: i64 = scaled_scores.iter().sum();
        (abs_scores, scaled_scores, actual_sum)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pathway {
    pub name: String,
    pub description: Option<String>,
    pub genes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathwayDb {
    pub pathways: Vec<Pathway>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnrichmentResult {
    pub pathway_name: String,
    pub size: usize,
    pub es: f64,
    pub nes: Option<f64>,
    pub p_value: f64,
    pub padj: Option<f64>,
    pub log2err: Option<f64>,
    pub leading_edge: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnrichmentResultExport {
    pub pathway: String,
    pub size: usize,
    pub es: f64,
    pub nes: Option<f64>,
    pub pval: f64,
    pub padj: Option<f64>,
    pub log2err: Option<f64>,
    pub leading_edge: Vec<String>,
}

impl EnrichmentResult {
    pub fn export(&self) -> EnrichmentResultExport {
        EnrichmentResultExport {
            pathway: self.pathway_name.clone(),
            size: self.size,
            es: self.es,
            nes: self.nes,
            pval: self.p_value,
            padj: self.padj,
            log2err: self.log2err,
            leading_edge: self.leading_edge.clone(),
        }
    }

    pub fn leading_edge_csv(&self) -> String {
        self.leading_edge.join(",")
    }
}
