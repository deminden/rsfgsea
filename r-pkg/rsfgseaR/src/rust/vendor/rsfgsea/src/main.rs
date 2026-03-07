use anyhow::{Result, bail};
use clap::{Parser, ValueEnum};
use rsfgsea::prelude::*;
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

    /// Random seed
    #[arg(short, long, default_value_t = 42)]
    seed: u64,

    /// Output TSV path
    #[arg(short, long)]
    output: PathBuf,

    /// Minimal size of a gene set to test
    #[arg(long = "minSize", visible_alias = "min-size", default_value_t = 1)]
    min_size: usize,

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

    if args.sample_size == 0 {
        bail!("--sampleSize must be greater than 0.");
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
        "Running mode={} (nPermSimple={}, nperm={:?})...",
        match args.mode {
            CliMode::Fgsea => "fgsea",
            CliMode::Multilevel => "multilevel",
            CliMode::Simple => "simple",
        },
        args.n_perm_simple,
        args.nperm
    );

    let score_type: ScoreType = args.score_type.into();

    let start = Instant::now();
    let max_size = args
        .max_size
        .unwrap_or_else(|| ranks.len().saturating_sub(1));
    let results = if args.gpu {
        run_gpu_mode(&args, &ranks, &pd.pathways, score_type, max_size)?
    } else {
        match args.mode {
            CliMode::Fgsea => fgsea_with_sample_size(
                &ranks,
                &pd.pathways,
                args.nperm,
                args.n_perm_simple,
                args.seed,
                args.min_size,
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
                    args.seed,
                    args.min_size,
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
                args.seed,
                args.min_size,
                max_size,
                args.eps,
                score_type,
                args.gsea_param,
                args.sample_size,
            ),
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
            "{}\t{}\t{:.8}\t{:.8}\t{:.8}\t{:.8}\t{:.8}\t{}",
            export.pathway,
            export.size,
            export.es,
            export.nes.unwrap_or(0.0),
            export.pval,
            export.padj.unwrap_or(1.0),
            export.log2err.unwrap_or(0.0),
            res.leading_edge_csv()
        )?;
    }
    Ok(())
}

#[cfg(feature = "gpu")]
fn run_gpu_mode(
    args: &Args,
    ranks: &RankedList,
    pathways: &[Pathway],
    score_type: ScoreType,
    max_size: usize,
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
        args.seed,
        args.min_size,
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
            seed: 42,
            output: PathBuf::from("out.tsv"),
            min_size: 1,
            max_size: None,
            eps: 1e-50,
            sample_size: 101,
            score_type: ScoreTypeArg::Std,
            gsea_param: 1.0,
            mode: CliMode::Fgsea,
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
