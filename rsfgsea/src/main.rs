use anyhow::Result;
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

    /// Number of permutations
    #[arg(short, long, default_value_t = 1000)]
    nperm: usize,

    /// Random seed
    #[arg(short, long, default_value_t = 42)]
    seed: u64,

    /// Output TSV path
    #[arg(short, long)]
    output: String,

    /// Minimum pathway size
    #[arg(long, default_value_t = 15)]
    min_size: usize,

    /// Maximum pathway size
    #[arg(long, default_value_t = 500)]
    max_size: usize,

    /// Eps parameter for multilevel GSEA
    #[arg(long, default_value_t = 1e-10)]
    eps: f64,

    /// Score type (Std, Pos, Neg)
    #[arg(long, default_value = "Std")]
    score_type: String,

    /// GSEA parameter (p)
    #[arg(long, default_value_t = 1.0)]
    gsea_param: f64,

    /// Number of threads (default: all cores)
    #[arg(short, long)]
    threads: Option<usize>,

    /// Enable GPU (requires gpu feature)
    #[arg(long)]
    gpu: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    if let Some(t) = args.threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(t)
            .build_global()?;
    }

    println!("Loading ranks from {}...", args.ranks);
    let ranks = read_ranked_list(&args.ranks)?;
    println!("Loaded {} genes.", ranks.len());

    println!("Loading pathways from {}...", args.gmt);
    let pd = read_gmt(&args.gmt)?;
    println!("Loaded {} pathways.", pd.pathways.len());

    println!("Running GSEA with {} permutations...", args.nperm);

    // In the future, check if args.gpu and feature is enabled
    #[cfg(feature = "gpu")]
    {
        if args.gpu {
            // TODO: Implement GPU run
            println!("GPU support requested but not yet implemented in main.");
        }
    }

    let score_type = match args.score_type.to_lowercase().as_str() {
        "pos" => ScoreType::Pos,
        "neg" => ScoreType::Neg,
        _ => ScoreType::Std,
    };

    let start = Instant::now();
    let results = run_gsea(
        &ranks,
        &pd.pathways,
        args.nperm,
        args.seed,
        args.min_size,
        args.max_size,
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
