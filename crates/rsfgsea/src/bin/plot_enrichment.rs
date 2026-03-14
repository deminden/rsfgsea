use anyhow::{Result, bail};
use clap::{Parser, ValueEnum};
use rsfgsea::prelude::*;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Write a single-pathway enrichment plot as PNG"
)]
struct Args {
    #[arg(short, long)]
    ranks: PathBuf,

    #[arg(short, long)]
    gmt: PathBuf,

    #[arg(short, long)]
    pathway: String,

    #[arg(short, long)]
    output: PathBuf,

    #[arg(
        long = "scoreType",
        visible_alias = "score-type",
        value_enum,
        default_value_t = ScoreTypeArg::Std
    )]
    score_type: ScoreTypeArg,

    #[arg(
        long = "gseaParam",
        visible_alias = "gsea-param",
        default_value_t = 1.0
    )]
    gsea_param: f64,

    #[arg(long = "width-in", visible_alias = "width", default_value_t = 4.5)]
    width_inches: f64,

    #[arg(long = "height-in", visible_alias = "height", default_value_t = 3.2)]
    height_inches: f64,

    #[arg(long, default_value_t = 300)]
    dpi: u32,

    #[arg(long, default_value_t = false)]
    transparent_background: bool,

    #[arg(long)]
    title: Option<String>,
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
    let ranks = read_ranked_list(&args.ranks)?;
    let pathways = read_gmt(&args.gmt)?;
    let pathway = pathways
        .pathways
        .iter()
        .find(|pw| pw.name == args.pathway)
        .cloned()
        .ok_or_else(|| {
            anyhow::anyhow!(
                "Pathway '{}' was not found in {}",
                args.pathway,
                args.gmt.display()
            )
        })?;

    if args.gsea_param < 0.0 || !args.gsea_param.is_finite() {
        bail!("--gseaParam must be finite and >= 0.");
    }

    write_enrichment_plot_png(
        &ranks,
        &pathway,
        &args.output,
        args.score_type.into(),
        args.gsea_param,
        &EnrichmentPlotOptions {
            width_inches: args.width_inches,
            height_inches: args.height_inches,
            dpi: args.dpi,
            transparent_background: args.transparent_background,
            title: args.title,
        },
    )?;

    Ok(())
}
