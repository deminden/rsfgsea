use anyhow::{Result, bail};
use clap::{Parser, ValueEnum};
use rsfgsea::prelude::*;
use rsfgsea::resolve_rng_seed;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the ranked list file (TSV/whitespace: gene, score)
    #[arg(short, long)]
    ranks: PathBuf,

    /// Path to the GMT file
    #[arg(short, long)]
    gmt: PathBuf,

    /// Number of permutations in simple fgsea stage
    #[arg(short = 'n', long = "nPermSimple", default_value_t = 1000)]
    n_perm_simple: usize,

    /// Optional fgsea-style simple-mode permutations (forces simple mode in fgsea wrapper mode)
    #[arg(long = "nperm")]
    nperm: Option<usize>,

    /// Random seed. Omit to generate a fresh seed for each run.
    #[arg(short, long)]
    seed: Option<u64>,

    /// Output TSV path
    #[arg(short, long)]
    output: PathBuf,

    /// Minimal size of a gene set to test (defaults to 1, or 5 in blitz mode)
    #[arg(long = "minSize", visible_alias = "min-size")]
    min_size: Option<usize>,

    /// Maximal size of a gene set to test (defaults to ranks length - 1)
    #[arg(long = "maxSize")]
    max_size: Option<usize>,

    /// Eps parameter for multilevel GSEA
    #[arg(long, default_value_t = 1e-50)]
    eps: f64,

    /// Multilevel sample size (R fgsea's sampleSize)
    #[arg(
        long = "sampleSize",
        visible_alias = "sample-size",
        default_value_t = 101
    )]
    sample_size: usize,

    /// Score type (std, pos, neg)
    #[arg(
        long = "scoreType",
        visible_alias = "score-type",
        value_enum,
        default_value_t = ScoreTypeArg::Std
    )]
    score_type: ScoreTypeArg,

    /// GSEA parameter value
    #[arg(
        long = "gseaParam",
        visible_alias = "gsea-param",
        default_value_t = 1.0
    )]
    gsea_param: f64,

    /// Execution mode: fgsea (wrapper semantics), multilevel, or simple
    #[arg(long, value_enum, default_value_t = CliMode::Fgsea)]
    mode: CliMode,

    /// Number of blitz calibration anchors
    #[arg(long = "blitz-anchors", default_value_t = 40)]
    blitz_anchors: usize,

    /// Force symmetric positive/negative blitz null fits
    #[arg(long = "blitz-symmetric")]
    blitz_symmetric: bool,

    /// Disable blitz signature centering
    #[arg(long = "blitz-no-center")]
    blitz_no_center: bool,

    /// Blitz normal-tail accuracy setting, kept for parity metadata
    #[arg(long = "blitz-accuracy", default_value_t = 40)]
    blitz_accuracy: usize,

    /// Blitz deep-tail accuracy setting, kept for parity metadata
    #[arg(long = "blitz-deep-accuracy", default_value_t = 50)]
    blitz_deep_accuracy: usize,

    /// Enrichment method: classic fgsea-compatible statistics or decor
    #[arg(long, value_enum, default_value_t = MethodArg::Classic)]
    method: MethodArg,

    /// Path to the decor redundancy cache
    #[arg(long = "decor-cache")]
    decor_cache: Option<PathBuf>,

    /// Path to a normalized expression matrix used to build the decor cache
    #[arg(long = "decor-expression")]
    decor_expression: Option<PathBuf>,

    /// Decor preset: sensitive, balanced, specific, or strict. Defaults to balanced.
    #[arg(long = "decor-preset", value_enum)]
    decor_preset: Option<DecorPresetArg>,

    /// Easy decor stringency control from 0 to 100; autoswitches calibrated presets.
    #[arg(long = "decor-stringency")]
    decor_stringency: Option<f64>,

    /// Override decor redundancy penalty strength
    #[arg(long = "decor-alpha", hide = true)]
    decor_alpha: Option<f64>,

    /// Decor cache handling mode
    #[arg(long = "decor-cache-mode", value_enum, default_value_t = DecorCacheModeArg::Auto)]
    decor_cache_mode: DecorCacheModeArg,

    /// Decor expression correlation method
    #[arg(
        long = "decor-correlation",
        value_enum,
        default_value_t = DecorCorrelationArg::Pearson,
        hide = true
    )]
    decor_correlation: DecorCorrelationArg,

    /// Decor redundancy score definition
    #[arg(
        long = "decor-redundancy",
        value_enum,
        default_value_t = DecorRedundancyArg::PositiveMean,
        hide = true
    )]
    decor_redundancy: DecorRedundancyArg,

    /// Override decor hit-weight formula
    #[arg(long = "decor-weight-formula", value_enum, hide = true)]
    decor_weight_formula: Option<DecorWeightFormulaArg>,

    /// Override threshold tau for threshold-rational decor weights
    #[arg(long = "decor-threshold", hide = true)]
    decor_threshold: Option<f64>,

    /// Small positive epsilon for scaled decor formulas
    #[arg(long = "decor-scale-epsilon", default_value_t = 1e-12, hide = true)]
    decor_scale_epsilon: f64,

    /// Decor expression matrix format
    #[arg(long = "decor-expression-format", value_enum, default_value_t = DecorExpressionFormatArg::Auto)]
    decor_expression_format: DecorExpressionFormatArg,

    /// Whether the decor expression matrix has a header row
    #[arg(long = "decor-expression-has-header", default_value_t = true)]
    decor_expression_has_header: bool,

    /// Number of workers (0 = default threadpool behavior)
    #[arg(long, default_value_t = 0)]
    nproc: usize,

    /// Enable GPU (requires gpu feature)
    #[arg(long)]
    gpu: bool,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
enum CliMode {
    Fgsea,
    Multilevel,
    Simple,
    Blitz,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
enum MethodArg {
    Classic,
    Decor,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
enum DecorCacheModeArg {
    Auto,
    Reuse,
    Rebuild,
}

impl From<DecorCacheModeArg> for DecorCacheMode {
    fn from(value: DecorCacheModeArg) -> Self {
        match value {
            DecorCacheModeArg::Auto => DecorCacheMode::Auto,
            DecorCacheModeArg::Reuse => DecorCacheMode::Reuse,
            DecorCacheModeArg::Rebuild => DecorCacheMode::Rebuild,
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
enum DecorCorrelationArg {
    Pearson,
    Spearman,
}

impl From<DecorCorrelationArg> for DecorCorrelation {
    fn from(value: DecorCorrelationArg) -> Self {
        match value {
            DecorCorrelationArg::Pearson => DecorCorrelation::Pearson,
            DecorCorrelationArg::Spearman => DecorCorrelation::Spearman,
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
enum DecorRedundancyArg {
    PositiveMean,
    AbsMean,
}

impl From<DecorRedundancyArg> for DecorRedundancy {
    fn from(value: DecorRedundancyArg) -> Self {
        match value {
            DecorRedundancyArg::PositiveMean => DecorRedundancy::PositiveMean,
            DecorRedundancyArg::AbsMean => DecorRedundancy::AbsMean,
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
enum DecorPresetArg {
    Sensitive,
    Balanced,
    Specific,
    Strict,
}

impl From<DecorPresetArg> for DecorPreset {
    fn from(value: DecorPresetArg) -> Self {
        match value {
            DecorPresetArg::Sensitive => DecorPreset::Sensitive,
            DecorPresetArg::Balanced => DecorPreset::Balanced,
            DecorPresetArg::Specific => DecorPreset::Specific,
            DecorPresetArg::Strict => DecorPreset::Strict,
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
enum DecorWeightFormulaArg {
    RawRational,
    ExpScaled,
    ThresholdRational,
}

impl From<DecorWeightFormulaArg> for DecorWeightFormula {
    fn from(value: DecorWeightFormulaArg) -> Self {
        match value {
            DecorWeightFormulaArg::RawRational => DecorWeightFormula::RawRational,
            DecorWeightFormulaArg::ExpScaled => DecorWeightFormula::ExpScaled,
            DecorWeightFormulaArg::ThresholdRational => DecorWeightFormula::ThresholdRational,
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
enum DecorExpressionFormatArg {
    Auto,
    Tsv,
    Csv,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
enum ScoreTypeArg {
    Std,
    Pos,
    Neg,
}

impl From<ScoreTypeArg> for ScoreType {
    fn from(value: ScoreTypeArg) -> Self {
        match value {
            ScoreTypeArg::Std => ScoreType::Std,
            ScoreTypeArg::Pos => ScoreType::Pos,
            ScoreTypeArg::Neg => ScoreType::Neg,
        }
    }
}

#[cfg(feature = "gpu")]
#[derive(Debug, Clone, Copy, PartialEq)]
struct GpuModeConfig {
    n_perm: usize,
    eps: f64,
    sample_size: usize,
    allow_multilevel: bool,
}

#[cfg(feature = "gpu")]
fn validate_gpu_mode_args(args: &Args) -> Result<GpuModeConfig> {
    if args.mode != CliMode::Fgsea {
        bail!("--gpu currently supports only --mode fgsea.");
    }

    Ok(GpuModeConfig {
        n_perm: args.nperm.unwrap_or(args.n_perm_simple),
        eps: args.eps,
        sample_size: args.sample_size,
        allow_multilevel: args.nperm.is_none(),
    })
}

fn main() -> Result<()> {
    let args = Args::parse();
    let seed = if args.mode == CliMode::Blitz {
        args.seed.unwrap_or(0)
    } else {
        resolve_rng_seed(args.seed)
    };

    if args.sample_size == 0 {
        bail!("--sampleSize must be greater than 0.");
    }
    if args.min_size.is_some_and(|min_size| min_size == 0) {
        bail!("--minSize must be greater than 0.");
    }
    if args.blitz_anchors == 0 {
        bail!("--blitz-anchors must be greater than 0.");
    }
    if args.blitz_accuracy == 0 || args.blitz_deep_accuracy == 0 {
        bail!("--blitz-accuracy and --blitz-deep-accuracy must be greater than 0.");
    }
    if args
        .decor_alpha
        .is_some_and(|alpha| alpha < 0.0 || !alpha.is_finite())
    {
        bail!("decor alpha must be >= 0.");
    }
    if args
        .decor_threshold
        .is_some_and(|threshold| !(0.0..1.0).contains(&threshold) || !threshold.is_finite())
    {
        bail!("decor threshold must be >= 0 and < 1.");
    }
    if args
        .decor_stringency
        .is_some_and(|stringency| !stringency.is_finite() || !(0.0..=100.0).contains(&stringency))
    {
        bail!("decor stringency must be a finite value from 0 to 100.");
    }
    if args.decor_preset.is_some() && args.decor_stringency.is_some() {
        bail!("use either --decor-preset or --decor-stringency, not both.");
    }
    if args.decor_scale_epsilon <= 0.0 || !args.decor_scale_epsilon.is_finite() {
        bail!("decor scale epsilon must be > 0.");
    }
    if args.decor_expression_format == DecorExpressionFormatArg::Csv {
        bail!("CSV decor expression format is not implemented yet; use tab-separated input.");
    }
    if args.method == MethodArg::Decor && args.gpu {
        bail!(
            "decor supports CPU fixed-permutation simple runs; use --mode simple or provide --nperm without --gpu."
        );
    }
    if args.mode == CliMode::Blitz {
        if args.gpu {
            bail!("gpu is not supported with --mode blitz.");
        }
        if args.method != MethodArg::Classic {
            bail!("--mode blitz supports only --method classic.");
        }
        if args.nperm.is_some() {
            bail!("--nperm is not supported with --mode blitz.");
        }
        if args.score_type != ScoreTypeArg::Std {
            bail!("--mode blitz supports only --scoreType std.");
        }
        if (args.gsea_param - 1.0).abs() > f64::EPSILON {
            bail!("--mode blitz supports only --gseaParam 1.");
        }
    }
    if args.method == MethodArg::Decor
        && (args.mode == CliMode::Multilevel
            || (args.mode == CliMode::Fgsea && args.nperm.is_none()))
    {
        bail!(
            "decor supports CPU fixed-permutation simple runs; use --mode simple or provide --nperm."
        );
    }

    if args.nproc > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(args.nproc)
            .build_global()?;
    }

    println!("Loading ranks from {}...", args.ranks.display());
    let ranks = read_ranked_list(&args.ranks)?;
    println!("Loaded {} genes.", ranks.len());

    println!("Loading pathways from {}...", args.gmt.display());
    let pd = read_gmt(&args.gmt)?;
    println!("Loaded {} pathways.", pd.pathways.len());

    println!(
        "Running method={} mode={} (nPermSimple={}, nperm={:?})...",
        match args.method {
            MethodArg::Classic => "classic",
            MethodArg::Decor => "decor",
        },
        match args.mode {
            CliMode::Fgsea => "fgsea",
            CliMode::Multilevel => "multilevel",
            CliMode::Simple => "simple",
            CliMode::Blitz => "blitz",
        },
        args.n_perm_simple,
        args.nperm
    );
    println!("Using RNG seed: {seed}");

    let score_type: ScoreType = args.score_type.into();

    let start = Instant::now();
    let min_size = args
        .min_size
        .unwrap_or(if args.mode == CliMode::Blitz { 5 } else { 1 });
    let max_size = args.max_size.unwrap_or_else(|| {
        if args.mode == CliMode::Blitz {
            4000
        } else {
            ranks.len().saturating_sub(1)
        }
    });
    let results = if args.method == MethodArg::Decor {
        let mut options = DecorOptions::default();
        let stringency_resolved = args
            .decor_stringency
            .map(|stringency| options.apply_stringency(stringency))
            .transpose()
            .map_err(anyhow::Error::msg)?;
        let resolved = if let Some(stringency_resolved) = stringency_resolved {
            stringency_resolved.preset_resolution
        } else {
            options.apply_preset(args.decor_preset.unwrap_or(DecorPresetArg::Balanced).into())
        };
        options.cache_path = args.decor_cache.clone();
        options.expression_path = args.decor_expression.clone();
        options.expression_has_header = args.decor_expression_has_header;
        options.cache_mode = args.decor_cache_mode.into();
        options.correlation = args.decor_correlation.into();
        options.redundancy = args.decor_redundancy.into();
        options.scale_epsilon = args.decor_scale_epsilon;
        if let Some(formula) = args.decor_weight_formula {
            options.weight_formula = formula.into();
        }
        if let Some(alpha) = args.decor_alpha {
            options.alpha = alpha;
        }
        if let Some(threshold_tau) = args.decor_threshold {
            options.threshold_tau = threshold_tau;
        }
        let cache_path = options
            .cache_path
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("method decor requires --decor-cache"))?;
        if cache_path.exists() {
            println!("Checking decor cache: {}", cache_path.display());
        } else {
            println!("Decor cache not found: {}", cache_path.display());
        }
        if let Some(expression_path) = &options.expression_path {
            println!("Decor expression: {}", expression_path.display());
        }
        let (cache, status) = ensure_decor_cache_for_paths(
            &pd.pathways,
            &args.gmt,
            &options,
            args.decor_expression_has_header,
        )?;
        match status {
            DecorCacheStatus::Reused => {
                println!("Reusing compatible decor cache: {}", cache_path.display());
            }
            DecorCacheStatus::Built => {
                println!(
                    "Decor cache built: {} pathways, {} pathway-gene scores",
                    cache.metadata.n_pathways, cache.metadata.n_rows
                );
            }
            DecorCacheStatus::Rebuilt => {
                println!(
                    "Decor cache rebuilt: {} pathways, {} pathway-gene scores",
                    cache.metadata.n_pathways, cache.metadata.n_rows
                );
            }
        }
        if let Some(stringency_resolved) = stringency_resolved {
            println!(
                "Decor stringency={} resolved to preset={} ({})",
                stringency_resolved.stringency,
                stringency_resolved.preset_resolution.preset,
                stringency_resolved.band
            );
        }
        println!("Decor preset={}", resolved.preset);
        println!(
            "Decor resolved preset: weight_formula={}, alpha={}, threshold_tau={}",
            resolved.weight_formula, resolved.alpha, resolved.threshold_tau
        );
        println!(
            "Decor effective preset: weight_formula={}, alpha={}, threshold_tau={}",
            options.weight_formula, options.alpha, options.threshold_tau
        );
        if let Some(target) = resolved.target_median_penalty {
            println!("Decor target median penalty={target}");
        }
        fgsea_decor_simple_with_options(
            &ranks,
            &pd.pathways,
            &cache,
            &options,
            args.nperm.unwrap_or(args.n_perm_simple),
            Some(seed),
            min_size,
            max_size,
            args.eps,
            score_type,
            args.gsea_param,
            args.sample_size,
        )?
    } else if args.gpu {
        run_gpu_mode(&args, &ranks, &pd.pathways, score_type, max_size, seed)?
    } else if args.mode == CliMode::Blitz {
        fgsea_blitz_with_options(
            &ranks,
            &pd.pathways,
            &BlitzOptions {
                permutations: args.n_perm_simple,
                anchors: args.blitz_anchors,
                min_size,
                max_size,
                processes: if args.nproc > 0 { args.nproc } else { 4 },
                symmetric: args.blitz_symmetric,
                seed,
                center: !args.blitz_no_center,
                accuracy: args.blitz_accuracy,
                deep_accuracy: args.blitz_deep_accuracy,
            },
        )?
    } else {
        match args.mode {
            CliMode::Fgsea => fgsea_with_sample_size(
                &ranks,
                &pd.pathways,
                args.nperm,
                args.n_perm_simple,
                Some(seed),
                min_size,
                max_size,
                args.eps,
                score_type,
                args.gsea_param,
                args.sample_size,
            ),
            CliMode::Multilevel => {
                if args.nperm.is_some() {
                    bail!("--nperm is only valid with --mode fgsea or --mode simple.");
                }
                fgsea_multilevel_with_sample_size(
                    &ranks,
                    &pd.pathways,
                    args.n_perm_simple,
                    Some(seed),
                    min_size,
                    max_size,
                    args.eps,
                    score_type,
                    args.gsea_param,
                    args.sample_size,
                )
            }
            CliMode::Simple => fgsea_simple_with_sample_size(
                &ranks,
                &pd.pathways,
                args.nperm.unwrap_or(args.n_perm_simple),
                Some(seed),
                min_size,
                max_size,
                args.eps,
                score_type,
                args.gsea_param,
                args.sample_size,
            ),
            CliMode::Blitz => unreachable!("blitz mode handled before fgsea-compatible modes"),
        }
    };
    let duration = start.elapsed();
    println!("GSEA computation took: {:.2?}", duration);
    println!("GSEA_COMP_TIME_MS: {}", duration.as_millis());

    println!("Writing results to {}...", args.output.display());
    write_results(&args.output, &results)?;

    println!("Done.");
    Ok(())
}

fn write_results(path: &Path, results: &[EnrichmentResult]) -> Result<()> {
    let mut out = File::create(path)?;
    writeln!(
        out,
        "pathway\tsize\tes\tnes\tpval\tpadj\tlog2err\tleading_edge"
    )?;
    for res in results {
        let export = res.export();
        writeln!(
            out,
            "{}\t{}\t{:.8}\t{}\t{:.8}\t{}\t{}\t{}",
            export.pathway,
            export.size,
            export.es,
            format_optional_float(export.nes),
            export.pval,
            format_optional_float(export.padj),
            format_optional_float(export.log2err),
            res.leading_edge_csv()
        )?;
    }
    Ok(())
}

fn format_optional_float(value: Option<f64>) -> String {
    value
        .map(|value| format!("{value:.8}"))
        .unwrap_or_else(|| "NA".to_string())
}

#[cfg(feature = "gpu")]
fn run_gpu_mode(
    args: &Args,
    ranks: &RankedList,
    pathways: &[Pathway],
    score_type: ScoreType,
    max_size: usize,
    seed: u64,
) -> Result<Vec<EnrichmentResult>> {
    let config = validate_gpu_mode_args(args)?;

    println!(
        "GPU hybrid path enabled: simple-stage screening on GPU, multilevel refinement on CPU."
    );
    if !config.allow_multilevel {
        println!(
            "GPU wrapper forced into simple-only mode via --nperm={}.",
            config.n_perm
        );
    }
    rsfgsea::algo::run_gsea_gpu_with_config(
        ranks,
        pathways,
        config.n_perm,
        Some(seed),
        args.min_size.unwrap_or(1),
        max_size,
        config.eps,
        score_type,
        args.gsea_param,
        config.sample_size,
        config.allow_multilevel,
    )
}

#[cfg(not(feature = "gpu"))]
fn run_gpu_mode(
    _args: &Args,
    _ranks: &RankedList,
    _pathways: &[Pathway],
    _score_type: ScoreType,
    _max_size: usize,
    _seed: u64,
) -> Result<Vec<EnrichmentResult>> {
    bail!("--gpu requires building the CLI with --features gpu.");
}

#[cfg(all(test, feature = "gpu"))]
mod tests {
    use super::*;

    fn base_args() -> Args {
        Args {
            ranks: PathBuf::from("ranks.tsv"),
            gmt: PathBuf::from("pathways.gmt"),
            n_perm_simple: 1000,
            nperm: None,
            seed: Some(42),
            output: PathBuf::from("out.tsv"),
            min_size: Some(1),
            max_size: None,
            eps: 1e-50,
            sample_size: 101,
            score_type: ScoreTypeArg::Std,
            gsea_param: 1.0,
            mode: CliMode::Fgsea,
            blitz_anchors: 40,
            blitz_symmetric: false,
            blitz_no_center: false,
            blitz_accuracy: 40,
            blitz_deep_accuracy: 50,
            method: MethodArg::Classic,
            decor_cache: None,
            decor_expression: None,
            decor_preset: None,
            decor_stringency: None,
            decor_alpha: None,
            decor_cache_mode: DecorCacheModeArg::Auto,
            decor_correlation: DecorCorrelationArg::Pearson,
            decor_redundancy: DecorRedundancyArg::PositiveMean,
            decor_weight_formula: None,
            decor_threshold: None,
            decor_scale_epsilon: 1e-12,
            decor_expression_format: DecorExpressionFormatArg::Auto,
            decor_expression_has_header: true,
            nproc: 0,
            gpu: true,
        }
    }

    #[test]
    fn gpu_validation_accepts_custom_sample_size_and_eps() {
        let mut args = base_args();
        args.sample_size = 151;
        args.eps = 1e-8;

        let config = validate_gpu_mode_args(&args).unwrap();
        assert_eq!(
            config,
            GpuModeConfig {
                n_perm: 1000,
                eps: 1e-8,
                sample_size: 151,
                allow_multilevel: true,
            }
        );
    }

    #[test]
    fn gpu_validation_allows_wrapper_nperm_override() {
        let mut args = base_args();
        args.nperm = Some(250);

        let config = validate_gpu_mode_args(&args).unwrap();
        assert_eq!(
            config,
            GpuModeConfig {
                n_perm: 250,
                eps: 1e-50,
                sample_size: 101,
                allow_multilevel: false,
            }
        );
    }

    #[test]
    fn gpu_validation_still_rejects_non_fgsea_mode() {
        let mut args = base_args();
        args.mode = CliMode::Simple;

        let err = validate_gpu_mode_args(&args).unwrap_err();
        assert!(
            err.to_string()
                .contains("--gpu currently supports only --mode fgsea.")
        );
    }
}
