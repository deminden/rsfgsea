use anyhow::{Result, bail};
use clap::Parser;
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
    #[arg(
        short = 'n',
        long = "nPermSimple",
        visible_alias = "nperm",
        default_value_t = 1000
    )]
    n_perm_simple: usize,

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

    /// Number of workers (0 = default threadpool behavior)
    #[arg(long, default_value_t = 0)]
    nproc: usize,

    /// Enable GPU (requires gpu feature)
    #[arg(long)]
    gpu: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

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
        "Running GSEA with {} simple permutations...",
        args.n_perm_simple
    );

    // In the future, check if args.gpu and feature is enabled
    #[cfg(feature = "gpu")]
    {
        if args.gpu {
            // TODO: Implement GPU run
            println!("GPU support requested but not yet implemented in main.");
        }
    }

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
    let results = run_gsea(
        &ranks,
        &pd.pathways,
        args.n_perm_simple,
        args.seed,
        args.min_size,
        max_size,
        args.eps,
        score_type,
        args.gsea_param,
    );
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
