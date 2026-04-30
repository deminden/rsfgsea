use crate::algo::{IntoSeed, resolve_rng_seed};
use crate::algo_support::{
    apply_bh_adjustment, build_gene_index, compute_nes, leading_edge, mode_fraction_count,
    selected_tail_count, simple_log2err, warn_prepare_stats,
};
use crate::core::{
    DecorCacheMode, DecorCorrelation, DecorOptions, DecorRedundancy, EnrichmentResult, Pathway,
    RankedList, ScoreType,
};
use crate::rng_compat::{RLecuyerCmrgSeedCompat, RSampleKind};
use anyhow::{Context, Result, bail};
use flate2::Compression;
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use rayon::prelude::*;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

const CACHE_FORMAT: &str = "rsfgsea-decor-cache";
const CACHE_VERSION: &str = "1";
const GENE_ID_MODE: &str = "verbatim";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecorCacheStatus {
    Reused,
    Built,
    Rebuilt,
}

impl fmt::Display for DecorCacheStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DecorCacheStatus::Reused => write!(f, "reused"),
            DecorCacheStatus::Built => write!(f, "built"),
            DecorCacheStatus::Rebuilt => write!(f, "rebuilt"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecorCacheMetadata {
    pub format: String,
    pub version: String,
    pub created_by: String,
    pub gmt_sha256: String,
    pub expression_sha256: String,
    pub correlation: DecorCorrelation,
    pub redundancy: DecorRedundancy,
    pub expression_gene_axis: String,
    pub expression_has_header: bool,
    pub gene_id_mode: String,
    pub n_pathways: usize,
    pub n_rows: usize,
}

#[derive(Debug, Clone)]
pub struct DecorPathwayScores {
    pub genes: Vec<String>,
    pub redundancy: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct DecorCache {
    pub metadata: DecorCacheMetadata,
    pub pathways: BTreeMap<String, DecorPathwayScores>,
}

#[derive(Debug, Clone)]
pub struct DecorCacheExpectedMetadata {
    pub gmt_sha256: String,
    pub expression_sha256: Option<String>,
    pub correlation: DecorCorrelation,
    pub redundancy: DecorRedundancy,
    pub expression_gene_axis: String,
    pub expression_has_header: bool,
    pub gene_id_mode: String,
}

#[derive(Debug, Clone)]
pub struct DecorCacheCompatibility {
    pub reasons: Vec<String>,
}

impl DecorCacheCompatibility {
    pub fn is_compatible(&self) -> bool {
        self.reasons.is_empty()
    }
}

impl fmt::Display for DecorCorrelation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DecorCorrelation::Pearson => write!(f, "pearson"),
            DecorCorrelation::Spearman => write!(f, "spearman"),
        }
    }
}

impl fmt::Display for DecorRedundancy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DecorRedundancy::PositiveMean => write!(f, "positive_mean"),
            DecorRedundancy::AbsMean => write!(f, "abs_mean"),
        }
    }
}

impl fmt::Display for DecorCacheMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DecorCacheMode::Auto => write!(f, "auto"),
            DecorCacheMode::Reuse => write!(f, "reuse"),
            DecorCacheMode::Rebuild => write!(f, "rebuild"),
        }
    }
}

fn parse_correlation(value: &str) -> Result<DecorCorrelation> {
    match value {
        "pearson" => Ok(DecorCorrelation::Pearson),
        "spearman" => Ok(DecorCorrelation::Spearman),
        other => bail!("unsupported decor correlation in cache metadata: {other}"),
    }
}

fn parse_redundancy(value: &str) -> Result<DecorRedundancy> {
    match value {
        "positive_mean" => Ok(DecorRedundancy::PositiveMean),
        "abs_mean" => Ok(DecorRedundancy::AbsMean),
        other => bail!("unsupported decor redundancy in cache metadata: {other}"),
    }
}

#[inline]
pub fn calculate_es_decor(
    stats: &[f64],
    hits: &[usize],
    penalty: &[f64],
    n_total: usize,
    score_type: ScoreType,
) -> Result<(f64, usize)> {
    if hits.len() != penalty.len() {
        bail!(
            "decor redundancy length mismatch: got {} redundancy values for {} hits",
            penalty.len(),
            hits.len()
        );
    }
    if hits.is_empty() {
        return Ok((0.0, 0));
    }

    let m = hits.len();
    if m == n_total {
        return Ok((0.0, hits[0]));
    }

    let mut nr = 0.0_f64;
    for i in 0..m {
        nr += stats[hits[i]].abs() * penalty[i];
    }

    let mut max_p = f64::NEG_INFINITY;
    let mut min_p = f64::INFINITY;
    let mut max_i = 0usize;
    let mut min_i = 0usize;
    let mut csum = 0.0;
    let denom = (n_total - m) as f64;
    let inv_m = 1.0 / m as f64;
    let nr_is_zero = nr == 0.0;

    for i in 0..m {
        let adj_i = stats[hits[i]].abs() * penalty[i];
        csum += adj_i;
        let r_cum = if nr_is_zero {
            (i + 1) as f64 * inv_m
        } else {
            csum / nr
        };
        let miss = (hits[i] - i) as f64 / denom;
        let top = r_cum - miss;
        let bottom = if nr_is_zero {
            top - inv_m
        } else {
            top - adj_i / nr
        };
        if top > max_p {
            max_p = top;
            max_i = i;
        }
        if bottom < min_p {
            min_p = bottom;
            min_i = i;
        }
    }

    Ok(match score_type {
        ScoreType::Std => {
            if max_p == -min_p {
                (0.0, hits[0])
            } else if max_p > -min_p {
                (max_p, hits[max_i])
            } else {
                (min_p, hits[min_i])
            }
        }
        ScoreType::Pos => (max_p, hits[max_i]),
        ScoreType::Neg => (min_p, hits[min_i]),
    })
}

pub fn file_sha256(path: &Path) -> Result<String> {
    let mut file =
        File::open(path).with_context(|| format!("Failed to open '{}'", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = file
            .read(&mut buf)
            .with_context(|| format!("Failed to read '{}'", path.display()))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

pub fn validate_decor_cache(
    cache_metadata: &DecorCacheMetadata,
    expected: &DecorCacheExpectedMetadata,
) -> DecorCacheCompatibility {
    let mut reasons = Vec::new();
    if cache_metadata.format != CACHE_FORMAT {
        reasons.push(format!(
            "cache format differs: cache={}",
            cache_metadata.format
        ));
    }
    if cache_metadata.version != CACHE_VERSION {
        reasons.push(format!(
            "cache format version is unsupported: cache={}",
            cache_metadata.version
        ));
    }
    if cache_metadata.gmt_sha256 != expected.gmt_sha256 {
        reasons.push("GMT SHA256 differs".to_string());
    }
    if let Some(expression_sha256) = &expected.expression_sha256
        && cache_metadata.expression_sha256 != *expression_sha256
    {
        reasons.push("expression SHA256 differs".to_string());
    }
    if cache_metadata.correlation != expected.correlation {
        reasons.push(format!(
            "correlation method differs: cache={}, requested={}",
            cache_metadata.correlation, expected.correlation
        ));
    }
    if cache_metadata.redundancy != expected.redundancy {
        reasons.push(format!(
            "redundancy mode differs: cache={}, requested={}",
            cache_metadata.redundancy, expected.redundancy
        ));
    }
    if cache_metadata.expression_gene_axis != expected.expression_gene_axis {
        reasons.push(format!(
            "expression gene axis differs: cache={}, requested={}",
            cache_metadata.expression_gene_axis, expected.expression_gene_axis
        ));
    }
    if cache_metadata.expression_has_header != expected.expression_has_header {
        reasons.push(format!(
            "expression header setting differs: cache={}, requested={}",
            cache_metadata.expression_has_header, expected.expression_has_header
        ));
    }
    if cache_metadata.gene_id_mode != expected.gene_id_mode {
        reasons.push(format!(
            "gene ID mode differs: cache={}, requested={}",
            cache_metadata.gene_id_mode, expected.gene_id_mode
        ));
    }
    DecorCacheCompatibility { reasons }
}

pub fn ensure_decor_cache_for_paths(
    pathways: &[Pathway],
    gmt_path: &Path,
    options: &DecorOptions,
    expression_has_header: bool,
) -> Result<(DecorCache, DecorCacheStatus)> {
    if options.alpha < 0.0 || !options.alpha.is_finite() {
        bail!("decor alpha must be a finite value >= 0");
    }
    if options.correlation == DecorCorrelation::Spearman {
        bail!("spearman decor correlation is not implemented yet");
    }

    let cache_path = options
        .cache_path
        .as_deref()
        .context("method decor requires --decor-cache")?;
    let gmt_sha256 = file_sha256(gmt_path)?;
    let expression_sha256 = options
        .expression_path
        .as_deref()
        .map(file_sha256)
        .transpose()?;
    let expected = DecorCacheExpectedMetadata {
        gmt_sha256,
        expression_sha256: expression_sha256.clone(),
        correlation: options.correlation,
        redundancy: options.redundancy,
        expression_gene_axis: "rows".to_string(),
        expression_has_header,
        gene_id_mode: GENE_ID_MODE.to_string(),
    };

    let cache_exists = cache_path.exists();
    if cache_exists && options.cache_mode != DecorCacheMode::Rebuild {
        match read_decor_cache(cache_path) {
            Ok(cache) => {
                let compatibility = validate_decor_cache(&cache.metadata, &expected);
                if compatibility.is_compatible() {
                    return Ok((cache, DecorCacheStatus::Reused));
                }
                if options.cache_mode == DecorCacheMode::Reuse {
                    bail!("{}", incompatible_message(&compatibility, true));
                }
                if options.expression_path.is_none() {
                    bail!("{}", incompatible_message(&compatibility, false));
                }
            }
            Err(err) => {
                if options.cache_mode == DecorCacheMode::Reuse {
                    bail!("Decor cache could not be read: {err}");
                }
                if options.expression_path.is_none() {
                    bail!(
                        "Decor cache could not be read and --decor-expression was not provided: {err}"
                    );
                }
            }
        }
    } else if !cache_exists && options.cache_mode == DecorCacheMode::Reuse {
        bail!("decor cache does not exist: {}", cache_path.display());
    }

    let expression_path = options.expression_path.as_deref().context(
        "decor cache does not exist or is incompatible and --decor-expression was not provided",
    )?;
    let expression_sha256 = expression_sha256.context("internal error: missing expression hash")?;
    let cache = build_decor_cache_from_expression(
        pathways,
        expression_path,
        DecorCacheExpectedMetadata {
            expression_sha256: Some(expression_sha256),
            ..expected
        },
    )?;
    write_decor_cache_atomic(cache_path, &cache)?;
    let status = if cache_exists {
        DecorCacheStatus::Rebuilt
    } else {
        DecorCacheStatus::Built
    };
    Ok((cache, status))
}

fn incompatible_message(compatibility: &DecorCacheCompatibility, suggest_rebuild: bool) -> String {
    let mut msg = String::from("Decor cache is incompatible:");
    for reason in &compatibility.reasons {
        msg.push_str("\n  - ");
        msg.push_str(reason);
    }
    if suggest_rebuild {
        msg.push_str("\nUse --decor-cache-mode rebuild with --decor-expression to rebuild it.");
    } else {
        msg.push_str(
            "\nProvide --decor-expression or use --decor-cache-mode rebuild to rebuild it.",
        );
    }
    msg
}

#[derive(Debug)]
struct ExpressionMatrix {
    genes: Vec<String>,
    values: Vec<Vec<f64>>,
}

fn read_expression_matrix(path: &Path, has_header: bool) -> Result<ExpressionMatrix> {
    let file = File::open(path).with_context(|| format!("Failed to open '{}'", path.display()))?;
    let reader: Box<dyn BufRead> = if path.extension().is_some_and(|ext| ext == "gz") {
        Box::new(BufReader::new(GzDecoder::new(file)))
    } else {
        Box::new(BufReader::new(file))
    };

    let mut genes = Vec::new();
    let mut values = Vec::new();
    let mut seen = HashSet::new();
    let mut expected_cols = None;

    for (line_idx, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        if has_header && line_idx == 0 {
            let cols = line.split('\t').count();
            if cols < 3 {
                bail!("expression matrix header must contain gene and at least two sample columns");
            }
            expected_cols = Some(cols - 1);
            continue;
        }

        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() < 3 {
            bail!(
                "Malformed expression matrix line {}: expected gene and at least two sample columns",
                line_idx + 1
            );
        }
        if let Some(cols) = expected_cols {
            if fields.len() - 1 != cols {
                bail!(
                    "Malformed expression matrix line {}: expected {} sample columns, found {}",
                    line_idx + 1,
                    cols,
                    fields.len() - 1
                );
            }
        } else {
            expected_cols = Some(fields.len() - 1);
        }

        let gene = fields[0].to_string();
        if gene.is_empty() {
            bail!(
                "expression matrix line {} has an empty gene identifier",
                line_idx + 1
            );
        }
        if !seen.insert(gene.clone()) {
            bail!("expression matrix has duplicate gene identifier: {gene}");
        }
        let row: Vec<f64> = fields[1..]
            .iter()
            .map(|value| {
                value.parse::<f64>().with_context(|| {
                    format!(
                        "Failed to parse expression value '{}' on line {}",
                        value,
                        line_idx + 1
                    )
                })
            })
            .collect::<Result<Vec<_>>>()?;
        if row.iter().any(|value| !value.is_finite()) {
            bail!(
                "expression matrix line {} contains a non-finite value",
                line_idx + 1
            );
        }
        genes.push(gene);
        values.push(row);
    }

    if values.is_empty() {
        bail!("expression matrix contains no gene rows");
    }
    if expected_cols.unwrap_or(0) < 2 {
        bail!("expression matrix must contain at least two sample columns");
    }
    Ok(ExpressionMatrix { genes, values })
}

fn standardized_rows(matrix: ExpressionMatrix) -> HashMap<String, Vec<f64>> {
    matrix
        .genes
        .into_iter()
        .zip(matrix.values)
        .map(|(gene, values)| {
            let n = values.len() as f64;
            let mean = values.iter().sum::<f64>() / n;
            let centered: Vec<f64> = values.iter().map(|value| value - mean).collect();
            let ss = centered.iter().map(|value| value * value).sum::<f64>();
            if ss <= 0.0 {
                (gene, vec![0.0; centered.len()])
            } else {
                let scale = ss.sqrt();
                (
                    gene,
                    centered.into_iter().map(|value| value / scale).collect(),
                )
            }
        })
        .collect()
}

pub fn build_decor_cache_from_expression(
    pathways: &[Pathway],
    expression_path: &Path,
    expected: DecorCacheExpectedMetadata,
) -> Result<DecorCache> {
    if expected.correlation == DecorCorrelation::Spearman {
        bail!("spearman decor correlation is not implemented yet");
    }
    let expression = read_expression_matrix(expression_path, expected.expression_has_header)?;
    let standardized = standardized_rows(expression);

    let mut rows: Vec<(String, DecorPathwayScores)> = pathways
        .par_iter()
        .map(|pathway| {
            let mut genes = Vec::new();
            let mut vectors = Vec::new();
            let mut seen = HashSet::new();
            for gene in &pathway.genes {
                if !seen.insert(gene) {
                    continue;
                }
                if let Some(vector) = standardized.get(gene) {
                    genes.push(gene.clone());
                    vectors.push(vector.as_slice());
                }
            }
            let redundancy = compute_redundancy(&vectors, expected.redundancy);
            (
                pathway.name.clone(),
                DecorPathwayScores { genes, redundancy },
            )
        })
        .collect();
    rows.sort_by(|a, b| a.0.cmp(&b.0));

    let n_rows = rows.iter().map(|(_, row)| row.genes.len()).sum();
    let metadata = DecorCacheMetadata {
        format: CACHE_FORMAT.to_string(),
        version: CACHE_VERSION.to_string(),
        created_by: "rsfgsea".to_string(),
        gmt_sha256: expected.gmt_sha256,
        expression_sha256: expected
            .expression_sha256
            .unwrap_or_else(|| "unknown".to_string()),
        correlation: expected.correlation,
        redundancy: expected.redundancy,
        expression_gene_axis: expected.expression_gene_axis,
        expression_has_header: expected.expression_has_header,
        gene_id_mode: expected.gene_id_mode,
        n_pathways: rows.len(),
        n_rows,
    };
    Ok(DecorCache {
        metadata,
        pathways: rows.into_iter().collect(),
    })
}

fn compute_redundancy(vectors: &[&[f64]], mode: DecorRedundancy) -> Vec<f32> {
    let m = vectors.len();
    if m < 2 {
        return vec![0.0; m];
    }
    let mut sums = vec![0.0_f64; m];
    for i in 0..m {
        for j in i + 1..m {
            let corr = vectors[i]
                .iter()
                .zip(vectors[j].iter())
                .map(|(a, b)| a * b)
                .sum::<f64>()
                .clamp(-1.0, 1.0);
            let value = match mode {
                DecorRedundancy::PositiveMean => corr.max(0.0),
                DecorRedundancy::AbsMean => corr.abs(),
            };
            sums[i] += value;
            sums[j] += value;
        }
    }
    let denom = (m - 1) as f64;
    sums.into_iter()
        .map(|sum| (sum / denom).clamp(0.0, 1.0) as f32)
        .collect()
}

pub fn read_decor_cache(path: &Path) -> Result<DecorCache> {
    let file = File::open(path).with_context(|| format!("Failed to open '{}'", path.display()))?;
    let reader: Box<dyn BufRead> = if path.extension().is_some_and(|ext| ext == "gz") {
        Box::new(BufReader::new(GzDecoder::new(file)))
    } else {
        Box::new(BufReader::new(file))
    };

    let mut metadata_map = BTreeMap::new();
    let mut pathways: BTreeMap<String, DecorPathwayScores> = BTreeMap::new();
    let mut saw_header = false;
    let mut row_count = 0usize;

    for (line_idx, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        if let Some(comment) = line.strip_prefix('#') {
            let Some((key, value)) = comment.trim().split_once('=') else {
                continue;
            };
            metadata_map.insert(key.trim().to_string(), value.trim().to_string());
            continue;
        }
        if !saw_header {
            if line != "pathway\tgene\tredundancy" {
                bail!("decor cache is missing required pathway/gene/redundancy header");
            }
            saw_header = true;
            continue;
        }
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() != 3 {
            bail!(
                "Malformed decor cache line {}: expected pathway, gene, redundancy",
                line_idx + 1
            );
        }
        let redundancy: f32 = fields[2].parse().with_context(|| {
            format!(
                "Failed to parse redundancy '{}' on cache line {}",
                fields[2],
                line_idx + 1
            )
        })?;
        if !redundancy.is_finite() {
            bail!(
                "decor cache line {} has non-finite redundancy",
                line_idx + 1
            );
        }
        let entry = pathways
            .entry(fields[0].to_string())
            .or_insert_with(|| DecorPathwayScores {
                genes: Vec::new(),
                redundancy: Vec::new(),
            });
        entry.genes.push(fields[1].to_string());
        entry.redundancy.push(redundancy.clamp(0.0, 1.0));
        row_count += 1;
    }

    if !saw_header {
        bail!("decor cache is missing tabular header");
    }
    let metadata = parse_cache_metadata(&metadata_map)?;
    if metadata.n_rows != row_count {
        bail!(
            "decor cache row count metadata mismatch: metadata={}, observed={}",
            metadata.n_rows,
            row_count
        );
    }
    Ok(DecorCache { metadata, pathways })
}

fn required_meta<'a>(map: &'a BTreeMap<String, String>, key: &str) -> Result<&'a str> {
    map.get(key)
        .map(String::as_str)
        .with_context(|| format!("decor cache is missing metadata key: {key}"))
}

fn parse_cache_metadata(map: &BTreeMap<String, String>) -> Result<DecorCacheMetadata> {
    Ok(DecorCacheMetadata {
        format: required_meta(map, "rsfgsea_decor_cache_format")?.to_string(),
        version: required_meta(map, "rsfgsea_decor_cache_version")?.to_string(),
        created_by: required_meta(map, "created_by")?.to_string(),
        gmt_sha256: required_meta(map, "gmt_sha256")?.to_string(),
        expression_sha256: required_meta(map, "expression_sha256")?.to_string(),
        correlation: parse_correlation(required_meta(map, "correlation")?)?,
        redundancy: parse_redundancy(required_meta(map, "redundancy")?)?,
        expression_gene_axis: required_meta(map, "expression_gene_axis")?.to_string(),
        expression_has_header: required_meta(map, "expression_has_header")?.parse()?,
        gene_id_mode: required_meta(map, "gene_id_mode")?.to_string(),
        n_pathways: required_meta(map, "n_pathways")?.parse()?,
        n_rows: required_meta(map, "n_rows")?.parse()?,
    })
}

pub fn write_decor_cache_atomic(path: &Path, cache: &DecorCache) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create cache directory '{}'", parent.display()))?;
    }
    let pid = std::process::id();
    let tmp_path = tmp_cache_path(path, pid);
    let result = (|| -> Result<()> {
        let file = File::create(&tmp_path)
            .with_context(|| format!("Failed to create '{}'", tmp_path.display()))?;
        if path.extension().is_some_and(|ext| ext == "gz") {
            let mut writer = GzEncoder::new(file, Compression::default());
            write_cache_contents(&mut writer, cache)?;
            writer.finish()?.sync_all()?;
        } else {
            let mut writer = BufWriter::new(file);
            write_cache_contents(&mut writer, cache)?;
            writer.flush()?;
            writer.get_ref().sync_all()?;
        }
        fs::rename(&tmp_path, path).with_context(|| {
            format!(
                "Failed to rename temporary cache '{}' to '{}'",
                tmp_path.display(),
                path.display()
            )
        })?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&tmp_path);
    }
    result
}

fn tmp_cache_path(path: &Path, pid: u32) -> PathBuf {
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("decor-cache");
    path.with_file_name(format!("{file_name}.tmp.{pid}"))
}

fn write_cache_contents<W: Write>(writer: &mut W, cache: &DecorCache) -> Result<()> {
    let md = &cache.metadata;
    writeln!(writer, "# rsfgsea_decor_cache_format={}", md.format)?;
    writeln!(writer, "# rsfgsea_decor_cache_version={}", md.version)?;
    writeln!(writer, "# created_by={}", md.created_by)?;
    writeln!(writer, "# gmt_sha256={}", md.gmt_sha256)?;
    writeln!(writer, "# expression_sha256={}", md.expression_sha256)?;
    writeln!(writer, "# correlation={}", md.correlation)?;
    writeln!(writer, "# redundancy={}", md.redundancy)?;
    writeln!(writer, "# expression_gene_axis={}", md.expression_gene_axis)?;
    writeln!(
        writer,
        "# expression_has_header={}",
        md.expression_has_header
    )?;
    writeln!(writer, "# gene_id_mode={}", md.gene_id_mode)?;
    writeln!(writer, "# n_pathways={}", md.n_pathways)?;
    writeln!(writer, "# n_rows={}", md.n_rows)?;
    writeln!(writer, "pathway\tgene\tredundancy")?;
    for (pathway, scores) in &cache.pathways {
        for (gene, redundancy) in scores.genes.iter().zip(scores.redundancy.iter()) {
            writeln!(writer, "{pathway}\t{gene}\t{redundancy:.8}")?;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn fgsea_decor_simple_with_sample_size<S: IntoSeed>(
    ranks: &RankedList,
    pathways: &[Pathway],
    cache: &DecorCache,
    alpha: f64,
    n_perm: usize,
    seed: S,
    min_size: usize,
    max_size: usize,
    eps: f64,
    score_type: ScoreType,
    gsea_param: f64,
    _sample_size: usize,
) -> Result<Vec<EnrichmentResult>> {
    let seed = resolve_rng_seed(seed.into_seed());
    run_decor_simple_internal(
        ranks, pathways, cache, alpha, n_perm, seed, min_size, max_size, eps, score_type,
        gsea_param,
    )
}

#[allow(clippy::too_many_arguments)]
fn run_decor_simple_internal(
    ranks: &RankedList,
    pathways: &[Pathway],
    cache: &DecorCache,
    alpha: f64,
    n_perm: usize,
    seed: u64,
    min_size: usize,
    max_size: usize,
    eps: f64,
    score_type: ScoreType,
    gsea_param: f64,
) -> Result<Vec<EnrichmentResult>> {
    if alpha < 0.0 || !alpha.is_finite() {
        bail!("decor alpha must be a finite value >= 0");
    }
    let gene_to_idx = build_gene_index(ranks);
    let n_total = ranks.len();
    warn_prepare_stats(ranks, score_type);
    let min_size = min_size.max(1);
    let max_size = max_size.min(n_total.saturating_sub(1));
    let eps = eps.clamp(0.0, 1.0);
    let (_abs_weights, scaled_scores, _ns_total) = ranks.prepare(gsea_param);
    let simple_stats: Vec<f64> = scaled_scores.iter().map(|&v| v as f64).collect();

    struct Working {
        pathway_name: String,
        size: usize,
        hits: Vec<usize>,
        penalty: Vec<f64>,
        es: f64,
        peak_idx: usize,
        n_le_es: usize,
        n_ge_es: usize,
        n_le_zero: usize,
        n_ge_zero: usize,
        le_zero_sum: f64,
        ge_zero_sum: f64,
        nes: Option<f64>,
        p_value: f64,
        padj: Option<f64>,
        log2err: Option<f64>,
    }

    let mut work: Vec<Working> = pathways
        .par_iter()
        .map(|pw| {
            let (hits, redundancy) = extract_decor_hits(pw, &gene_to_idx, cache);
            if hits.len() < min_size || hits.len() > max_size {
                return Ok(None);
            }
            let penalty: Vec<f64> = redundancy
                .into_iter()
                .map(|r| 1.0 / (1.0 + alpha * r))
                .collect();
            let (es, peak_idx) =
                calculate_es_decor(&simple_stats, &hits, &penalty, n_total, score_type)?;
            Ok(Some(Working {
                pathway_name: pw.name.clone(),
                size: hits.len(),
                hits,
                penalty,
                es,
                peak_idx,
                n_le_es: 0,
                n_ge_es: 0,
                n_le_zero: 0,
                n_ge_zero: 0,
                le_zero_sum: 0.0,
                ge_zero_sum: 0.0,
                nes: None,
                p_value: f64::NAN,
                padj: None,
                log2err: None,
            }))
        })
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .flatten()
        .collect();

    if n_perm > 0 && !work.is_empty() {
        let mut rng = RLecuyerCmrgSeedCompat::from_r_set_seed(seed as u32);
        for _ in 0..n_perm {
            for w in &mut work {
                let mut selected =
                    rng.sample_int_no_replace_with_kind(n_total, w.size, RSampleKind::Rejection);
                for idx in &mut selected {
                    *idx -= 1;
                }
                selected.sort_unstable();
                let (rand_es, _) =
                    calculate_es_decor(&simple_stats, &selected, &w.penalty, n_total, score_type)?;
                if rand_es <= w.es {
                    w.n_le_es += 1;
                }
                if rand_es >= w.es {
                    w.n_ge_es += 1;
                }
                if rand_es <= 0.0 {
                    w.n_le_zero += 1;
                    w.le_zero_sum += rand_es;
                }
                if rand_es >= 0.0 {
                    w.n_ge_zero += 1;
                    w.ge_zero_sum += rand_es;
                }
            }
        }
    }

    for w in &mut work {
        let le_zero_mean = if w.n_le_zero > 0 {
            w.le_zero_sum / w.n_le_zero as f64
        } else {
            0.0
        };
        let ge_zero_mean = if w.n_ge_zero > 0 {
            w.ge_zero_sum / w.n_ge_zero as f64
        } else {
            0.0
        };
        w.nes = compute_nes(w.es, score_type, le_zero_mean, ge_zero_mean);
        if w.nes.is_some() {
            let p_le = (w.n_le_es + 1) as f64 / (w.n_le_zero + 1) as f64;
            let p_ge = (w.n_ge_es + 1) as f64 / (w.n_ge_zero + 1) as f64;
            w.p_value = p_le.min(p_ge).max(eps);
        }
        let n_more_extreme =
            selected_tail_count(score_type, w.es, w.n_le_es as u64, w.n_ge_es as u64);
        let mode_fraction =
            mode_fraction_count(score_type, w.es, w.n_le_zero as u64, w.n_ge_zero as u64);
        w.log2err = if w.p_value.is_finite() && mode_fraction > 0 {
            simple_log2err(n_more_extreme, n_perm)
        } else {
            None
        };
    }

    let mut final_results: Vec<EnrichmentResult> = work
        .into_iter()
        .map(|w| EnrichmentResult {
            pathway_name: w.pathway_name,
            size: w.size,
            es: w.es,
            nes: w.nes,
            p_value: w.p_value,
            padj: w.padj,
            log2err: w.log2err,
            leading_edge: leading_edge(&w.hits, w.peak_idx, w.es, score_type, ranks),
        })
        .collect();
    apply_bh_adjustment(&mut final_results);
    final_results.sort_by(|a, b| a.pathway_name.cmp(&b.pathway_name));
    Ok(final_results)
}

fn extract_decor_hits(
    pathway: &Pathway,
    gene_to_idx: &HashMap<String, usize>,
    cache: &DecorCache,
) -> (Vec<usize>, Vec<f64>) {
    let Some(scores) = cache.pathways.get(&pathway.name) else {
        return (Vec::new(), Vec::new());
    };
    let redundancy_by_gene: HashMap<&str, f64> = scores
        .genes
        .iter()
        .zip(scores.redundancy.iter())
        .map(|(gene, redundancy)| (gene.as_str(), *redundancy as f64))
        .collect();
    let mut hit_pairs: Vec<(usize, f64)> = pathway
        .genes
        .iter()
        .filter_map(|gene| {
            let idx = gene_to_idx.get(gene)?;
            let redundancy = redundancy_by_gene.get(gene.as_str())?;
            Some((*idx, *redundancy))
        })
        .collect();
    hit_pairs.sort_by_key(|(idx, _)| *idx);
    hit_pairs.dedup_by_key(|(idx, _)| *idx);
    hit_pairs.into_iter().unzip()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algo::calculate_es_fgsea;
    use tempfile::tempdir;

    #[test]
    fn decor_es_matches_classic_when_redundancy_zero() {
        let stats = vec![4.0, 3.0, 2.0, -1.0, -2.0, -3.0];
        let hits = vec![0, 2, 5];
        let penalty = vec![1.0, 1.0, 1.0];
        let decor = calculate_es_decor(&stats, &hits, &penalty, stats.len(), ScoreType::Std)
            .unwrap()
            .0;
        let classic = calculate_es_fgsea(&stats, &hits, stats.len(), ScoreType::Std).0;
        assert!((decor - classic).abs() < 1e-12);
    }

    #[test]
    fn decor_alpha_zero_matches_classic() {
        let stats = vec![4.0, 3.0, 2.0, -1.0, -2.0, -3.0];
        let hits = vec![0, 2, 5];
        let redundancy = [0.9, 0.1, 0.4];
        let penalty: Vec<f64> = redundancy.iter().map(|r| 1.0 / (1.0 + 0.0 * r)).collect();
        let decor = calculate_es_decor(&stats, &hits, &penalty, stats.len(), ScoreType::Std)
            .unwrap()
            .0;
        let classic = calculate_es_fgsea(&stats, &hits, stats.len(), ScoreType::Std).0;
        assert!((decor - classic).abs() < 1e-12);
    }

    #[test]
    fn decor_downweights_high_redundancy_gene() {
        let stats = vec![5.0, 4.0, 1.0, -1.0, -2.0, -3.0];
        let hits = vec![0, 1, 5];
        let no_penalty = vec![1.0, 1.0, 1.0];
        let penalized = vec![1.0 / 2.0, 1.0, 1.0];
        let classic = calculate_es_decor(&stats, &hits, &no_penalty, stats.len(), ScoreType::Std)
            .unwrap()
            .0;
        let decor = calculate_es_decor(&stats, &hits, &penalized, stats.len(), ScoreType::Std)
            .unwrap()
            .0;
        assert_ne!(decor, classic);
        assert!(decor.is_finite());
    }

    #[test]
    fn invalid_redundancy_length_errors_cleanly() {
        let err = calculate_es_decor(&[1.0, 2.0], &[0, 1], &[1.0], 2, ScoreType::Std)
            .unwrap_err()
            .to_string();
        assert!(err.contains("length mismatch"));
    }

    #[test]
    fn cache_build_write_read_validate_round_trip() {
        let dir = tempdir().unwrap();
        let expr = dir.path().join("expression.tsv");
        fs::write(
            &expr,
            "gene\ts1\ts2\ts3\ts4\nA\t1\t2\t3\t4\nB\t1.1\t2.1\t3.1\t4.1\nC\t4\t3\t2\t1\nE\t1\t1\t1\t1\n",
        )
        .unwrap();
        let gmt = dir.path().join("test.gmt");
        fs::write(&gmt, "PW\tdesc\tA\tB\tC\tE\n").unwrap();
        let expected = DecorCacheExpectedMetadata {
            gmt_sha256: file_sha256(&gmt).unwrap(),
            expression_sha256: Some(file_sha256(&expr).unwrap()),
            correlation: DecorCorrelation::Pearson,
            redundancy: DecorRedundancy::PositiveMean,
            expression_gene_axis: "rows".to_string(),
            expression_has_header: true,
            gene_id_mode: GENE_ID_MODE.to_string(),
        };
        let pathways = vec![Pathway {
            name: "PW".to_string(),
            description: None,
            genes: vec!["A".into(), "B".into(), "C".into(), "E".into()],
        }];
        let cache = build_decor_cache_from_expression(&pathways, &expr, expected.clone()).unwrap();
        let path = dir.path().join("cache.decor.tsv");
        write_decor_cache_atomic(&path, &cache).unwrap();
        let loaded = read_decor_cache(&path).unwrap();
        assert!(validate_decor_cache(&loaded.metadata, &expected).is_compatible());
        let pw = loaded.pathways.get("PW").unwrap();
        assert_eq!(pw.genes, vec!["A", "B", "C", "E"]);
        assert!(pw.redundancy[0] > 0.3);
    }
}
