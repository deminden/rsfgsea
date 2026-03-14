use anyhow::{Result, bail};
use clap::{Parser, ValueEnum};
use rsfgsea::prelude::*;
use rsfgsea::resolve_rng_seed;
use std::collections::HashMap;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Write a multi-pathway GSEA table plot as PNG"
)]
struct Args {
    #[arg(short, long)]
    ranks: PathBuf,

    #[arg(short, long)]
    gmt: PathBuf,

    #[arg(short, long, num_args = 1..)]
    pathway: Vec<String>,

    #[arg(short, long)]
    output: PathBuf,

    #[arg(short = 'n', long = "nPermSimple", default_value_t = 1000)]
    n_perm_simple: usize,

    #[arg(long = "nperm")]
    nperm: Option<usize>,

    #[arg(short, long)]
    seed: Option<u64>,

    #[arg(long = "minSize", visible_alias = "min-size", default_value_t = 1)]
    min_size: usize,

    #[arg(long = "maxSize")]
    max_size: Option<usize>,

    #[arg(long, default_value_t = 1e-50)]
    eps: f64,

    #[arg(
        long = "sampleSize",
        visible_alias = "sample-size",
        default_value_t = 101
    )]
    sample_size: usize,

    #[arg(long = "scoreType", visible_alias = "score-type", value_enum, default_value_t = ScoreTypeArg::Std)]
    score_type: ScoreTypeArg,

    #[arg(
        long = "gseaParam",
        visible_alias = "gsea-param",
        default_value_t = 1.0
    )]
    gsea_param: f64,

    #[arg(long, value_enum, default_value_t = CliMode::Fgsea)]
    mode: CliMode,

    #[arg(long, default_value_t = 0)]
    nproc: usize,

    #[arg(long = "width-in", visible_alias = "width", default_value_t = 5.6)]
    width_inches: f64,

    #[arg(long = "height-in", visible_alias = "height")]
    height_inches: Option<f64>,

    #[arg(long, default_value_t = 300)]
    dpi: u32,

    #[arg(long, default_value_t = false)]
    transparent_background: bool,
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

fn main() -> Result<()> {
    let args = Args::parse();
    let seed = resolve_rng_seed(args.seed);
    if args.sample_size == 0 {
        bail!("--sampleSize must be greater than 0.");
    }
    if args.gsea_param < 0.0 || !args.gsea_param.is_finite() {
        bail!("--gseaParam must be finite and >= 0.");
    }
    if args.nproc > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(args.nproc)
            .build_global()?;
    }

    let ranks = read_ranked_list(&args.ranks)?;
    let pathway_db = read_gmt(&args.gmt)?;
    let pathway_map: HashMap<String, Pathway> = pathway_db
        .pathways
        .iter()
        .cloned()
        .map(|pathway| (pathway.name.clone(), pathway))
        .collect();
    let selected_pathways: Vec<Pathway> = args
        .pathway
        .iter()
        .map(|name| {
            pathway_map.get(name).cloned().ok_or_else(|| {
                anyhow::anyhow!("Pathway '{}' was not found in {}", name, args.gmt.display())
            })
        })
        .collect::<Result<_>>()?;

    let score_type: ScoreType = args.score_type.into();
    let max_size = args
        .max_size
        .unwrap_or_else(|| ranks.len().saturating_sub(1));
    let results = match args.mode {
        CliMode::Fgsea => fgsea_with_sample_size(
            &ranks,
            &pathway_db.pathways,
            args.nperm,
            args.n_perm_simple,
            Some(seed),
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
                &pathway_db.pathways,
                args.n_perm_simple,
                Some(seed),
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
            &pathway_db.pathways,
            args.nperm.unwrap_or(args.n_perm_simple),
            Some(seed),
            args.min_size,
            max_size,
            args.eps,
            score_type,
            args.gsea_param,
            args.sample_size,
        ),
    };

    write_gsea_table_plot_png(
        &ranks,
        &selected_pathways,
        &results,
        &args.output,
        args.gsea_param,
        &GseaTablePlotOptions {
            width_inches: args.width_inches,
            height_inches: args.height_inches,
            dpi: args.dpi,
            transparent_background: args.transparent_background,
        },
    )?;

    Ok(())
}
