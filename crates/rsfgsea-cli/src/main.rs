use anyhow::{Result, bail};
use clap::{Parser, ValueEnum};
use rsfgsea::prelude::*;
use std::fs::File;
use std::io::Write;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the ranked list file (TSV/whitespace: gene, score)
    #[arg(short, long)]
    ranks: String,

    /// Path to the GMT file
    #[arg(short, long)]
    gmt: String,

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
    output: String,

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
        default_value = "std"
    )]
    score_type: String,

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

    println!("Loading ranks from {}...", args.ranks);
    let ranks = read_ranked_list(&args.ranks)?;
    println!("Loaded {} genes.", ranks.len());

    println!("Loading pathways from {}...", args.gmt);
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

    let score_type = match args.score_type.to_lowercase().as_str() {
        "std" => ScoreType::Std,
        "pos" => ScoreType::Pos,
        "neg" => ScoreType::Neg,
        other => bail!(
            "Invalid scoreType '{}'. Expected one of: std, pos, neg.",
            other
        ),
    };

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
                run_gsea_with_sample_size(
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
            CliMode::Simple => run_gsea_simple_with_sample_size(
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

    println!("Writing results to {}...", args.output);
    let mut out = File::create(&args.output)?;
    writeln!(
        out,
        "pathway\tsize\tes\tnes\tpval\tpadj\tlog2err\tleading_edge"
    )?;
    for res in results {
        writeln!(
            out,
            "{}\t{}\t{:.8}\t{:.8}\t{:.8}\t{:.8}\t{:.8}\t{}",
            res.pathway_name,
            res.size,
            res.es,
            res.nes.unwrap_or(0.0),
            res.p_value,
            res.padj.unwrap_or(1.0),
            res.log2err.unwrap_or(0.0),
            res.leading_edge.join(",")
        )?;
    }

    println!("Done.");
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
    if args.mode != CliMode::Fgsea {
        bail!("--gpu currently supports only --mode fgsea.");
    }
    if args.nperm.is_some() {
        bail!("--gpu does not support --nperm/simple-only forcing yet.");
    }
    if args.sample_size != 101 {
        bail!(
            "--gpu currently uses sampleSize=101 internally; custom --sampleSize is not supported."
        );
    }
    if args.eps != 1e-50 {
        bail!("--gpu currently uses eps=1e-50 internally; custom --eps is not supported.");
    }

    println!(
        "GPU hybrid path enabled: simple-stage screening on GPU, multilevel refinement on CPU."
    );
    rsfgsea::algo::run_gsea_gpu(
        ranks,
        pathways,
        args.n_perm_simple,
        args.seed,
        args.min_size,
        max_size,
        score_type,
        args.gsea_param,
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
