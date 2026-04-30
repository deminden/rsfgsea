use serde::{Deserialize, Serialize};
use std::path::PathBuf;

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
pub struct DecorOptions {
    pub alpha: f64,
    pub cache_path: Option<PathBuf>,
    pub expression_path: Option<PathBuf>,
    pub cache_mode: DecorCacheMode,
    pub correlation: DecorCorrelation,
    pub redundancy: DecorRedundancy,
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
