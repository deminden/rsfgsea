#![allow(non_snake_case)]

use extendr_api::prelude::*;
use extendr_api::throw_r_error;
use rayon::ThreadPoolBuilder;
use rsfgsea::prelude::*;
use std::str::FromStr;

fn parse_score_type(score_type: &str) -> std::result::Result<ScoreType, String> {
    match score_type.to_lowercase().as_str() {
        "std" => Ok(ScoreType::Std),
        "pos" => Ok(ScoreType::Pos),
        "neg" => Ok(ScoreType::Neg),
        other => Err(format!(
            "Invalid scoreType '{other}'. Expected one of: std, pos, neg."
        )),
    }
}

fn parse_sample_kind(sample_kind: &str) -> std::result::Result<RSampleKind, String> {
    match sample_kind.to_lowercase().as_str() {
        "rejection" => Ok(RSampleKind::Rejection),
        "rounding" => Ok(RSampleKind::Rounding),
        other => Err(format!(
            "Invalid sampleKind '{other}'. Expected one of: Rejection, Rounding."
        )),
    }
}

fn parse_method(method: &str) -> std::result::Result<EnrichmentMethod, String> {
    match method.to_lowercase().as_str() {
        "classic" => Ok(EnrichmentMethod::Classic),
        "decor" => Ok(EnrichmentMethod::Decor),
        other => Err(format!(
            "Invalid method '{other}'. Expected one of: classic, decor."
        )),
    }
}

fn parse_decor_cache_mode(mode: &str) -> std::result::Result<DecorCacheMode, String> {
    match mode.to_lowercase().as_str() {
        "auto" => Ok(DecorCacheMode::Auto),
        "reuse" => Ok(DecorCacheMode::Reuse),
        "rebuild" => Ok(DecorCacheMode::Rebuild),
        other => Err(format!(
            "Invalid decor.cache.mode '{other}'. Expected one of: auto, reuse, rebuild."
        )),
    }
}

fn parse_decor_correlation(correlation: &str) -> std::result::Result<DecorCorrelation, String> {
    match correlation.to_lowercase().as_str() {
        "pearson" => Ok(DecorCorrelation::Pearson),
        "spearman" => Ok(DecorCorrelation::Spearman),
        other => Err(format!(
            "Invalid decor.correlation '{other}'. Expected one of: pearson, spearman."
        )),
    }
}

fn parse_decor_redundancy(redundancy: &str) -> std::result::Result<DecorRedundancy, String> {
    match redundancy.to_lowercase().as_str() {
        "positive_mean" => Ok(DecorRedundancy::PositiveMean),
        "abs_mean" => Ok(DecorRedundancy::AbsMean),
        other => Err(format!(
            "Invalid decor.redundancy '{other}'. Expected one of: positive_mean, abs_mean."
        )),
    }
}

fn parse_decor_weight_formula(formula: &str) -> std::result::Result<DecorWeightFormula, String> {
    DecorWeightFormula::from_str(formula)
}

fn normalize_seed(seed: Nullable<i32>) -> std::result::Result<Option<u64>, String> {
    match seed.into_option() {
        Some(seed) if seed < 0 => Err("seed must be greater than or equal to 0.".to_string()),
        Some(seed) => Ok(Some(seed as u64)),
        None => Ok(None),
    }
}

fn run_with_optional_thread_pool<T, F>(nproc: i32, f: F) -> std::result::Result<T, String>
where
    F: FnOnce() -> T + Send,
    T: Send,
{
    if nproc > 0 {
        let pool = ThreadPoolBuilder::new()
            .num_threads(nproc as usize)
            .build()
            .map_err(|err| format!("Failed to build Rayon thread pool: {err}"))?;
        Ok(pool.install(f))
    } else {
        Ok(f())
    }
}

#[allow(clippy::too_many_arguments)]
#[cfg(feature = "gpu")]
fn run_gpu_path(
    ranks: &RankedList,
    pathways: &[Pathway],
    n_perm: usize,
    seed: Option<u64>,
    min_size: usize,
    max_size: usize,
    eps: f64,
    score_type: ScoreType,
    gsea_param: f64,
    sample_size: usize,
    allow_multilevel: bool,
    nproc: i32,
) -> std::result::Result<Vec<EnrichmentResult>, String> {
    run_with_optional_thread_pool(nproc, || {
        run_gsea_gpu_with_config(
            ranks,
            pathways,
            n_perm,
            seed,
            min_size,
            max_size,
            eps,
            score_type,
            gsea_param,
            sample_size,
            allow_multilevel,
        )
    })
    .and_then(|res| res.map_err(|err| format!("GPU execution failed: {err}")))
}

#[allow(clippy::too_many_arguments)]
#[cfg(not(feature = "gpu"))]
fn run_gpu_path(
    _ranks: &RankedList,
    _pathways: &[Pathway],
    _n_perm: usize,
    _seed: Option<u64>,
    _min_size: usize,
    _max_size: usize,
    _eps: f64,
    _score_type: ScoreType,
    _gsea_param: f64,
    _sample_size: usize,
    _allow_multilevel: bool,
    _nproc: i32,
) -> std::result::Result<Vec<EnrichmentResult>, String> {
    Err(
        "GPU support is not enabled in this rsfgseaR build. Reinstall with RSFGSEAR_ENABLE_GPU=1 to enable it."
            .to_string(),
    )
}

/// Return the backend crate version.
/// @export
#[extendr]
fn rsfgsea_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Return the supported execution modes.
/// @export
#[extendr]
fn supported_modes() -> Vec<&'static str> {
    vec!["fgsea", "simple", "multilevel", "blitz"]
}

/// Write a single-pathway enrichment plot as PNG.
#[allow(clippy::too_many_arguments)]
#[extendr]
fn write_enrichment_plot(
    stats: Doubles,
    genes: Strings,
    pathway_genes: Strings,
    output_path: &str,
    pathway_name: &str,
    score_type: &str,
    gsea_param: f64,
    width_inches: f64,
    height_inches: f64,
    dpi: i32,
    transparent_background: bool,
    title: &str,
) -> Robj {
    let result = (|| -> std::result::Result<(), String> {
        if stats.len() != genes.len() {
            return Err("stats and gene names must have the same length.".to_string());
        }
        if width_inches <= 0.0 || height_inches <= 0.0 || dpi <= 0 {
            return Err("width_inches, height_inches, and dpi must be greater than 0.".to_string());
        }

        let genes: Vec<String> = genes.iter().map(|gene| gene.to_string()).collect();
        let scores: Vec<f64> = stats.iter().map(|v| v.0).collect();
        let ranks = RankedList::new(genes, scores);
        let pathway = Pathway {
            name: pathway_name.to_string(),
            description: None,
            genes: pathway_genes.iter().map(|gene| gene.to_string()).collect(),
        };
        let score_type = parse_score_type(score_type)?;

        write_enrichment_plot_png(
            &ranks,
            &pathway,
            output_path,
            score_type,
            gsea_param,
            &EnrichmentPlotOptions {
                width_inches,
                height_inches,
                dpi: dpi as u32,
                transparent_background,
                title: if title.is_empty() {
                    None
                } else {
                    Some(title.to_string())
                },
            },
        )
        .map_err(|err| err.to_string())
    })();

    match result {
        Ok(()) => output_path.into(),
        Err(err) => throw_r_error(err),
    }
}

/// Write a multi-pathway GSEA table plot as PNG.
#[allow(clippy::too_many_arguments)]
#[extendr]
fn write_gsea_table_plot(
    stats: Doubles,
    genes: Strings,
    pathway_names: Strings,
    pathway_gene_lists: List,
    result_pathways: Strings,
    result_nes: Doubles,
    result_pval: Doubles,
    result_padj: Doubles,
    output_path: &str,
    gsea_param: f64,
    width_inches: f64,
    height_inches: Nullable<f64>,
    dpi: i32,
    transparent_background: bool,
) -> Robj {
    let result = (|| -> std::result::Result<(), String> {
        if stats.len() != genes.len() {
            return Err("stats and gene names must have the same length.".to_string());
        }
        if pathway_names.len() != pathway_gene_lists.len() {
            return Err(
                "pathway_names and pathway_gene_lists must have the same length.".to_string(),
            );
        }
        if result_pathways.len() != result_nes.len()
            || result_pathways.len() != result_pval.len()
            || result_pathways.len() != result_padj.len()
        {
            return Err(
                "result_pathways, result_nes, result_pval, and result_padj must have the same length."
                    .to_string(),
            );
        }
        let height_inches = height_inches.into_option();
        if width_inches <= 0.0 || height_inches.is_some_and(|h| h <= 0.0) || dpi <= 0 {
            return Err("width_inches, height_inches, and dpi must be greater than 0.".to_string());
        }

        let genes: Vec<String> = genes.iter().map(|gene| gene.to_string()).collect();
        let scores: Vec<f64> = stats.iter().map(|v| v.0).collect();
        let ranks = RankedList::new(genes, scores);

        let pathway_names_vec: Vec<String> =
            pathway_names.iter().map(|name| name.to_string()).collect();
        let pathways: Vec<Pathway> = pathway_names_vec
            .into_iter()
            .zip(pathway_gene_lists.iter())
            .map(|(name, (_, genes_obj))| {
                let genes = genes_obj
                    .as_str_vector()
                    .ok_or_else(|| format!("pathway '{name}' genes must be a character vector."))?
                    .iter()
                    .map(|gene: &&str| (*gene).to_string())
                    .collect();
                Ok(Pathway {
                    name,
                    description: None,
                    genes,
                })
            })
            .collect::<std::result::Result<Vec<_>, String>>()?;

        let results: Vec<EnrichmentResult> = result_pathways
            .iter()
            .zip(result_nes.iter())
            .zip(result_pval.iter())
            .zip(result_padj.iter())
            .map(|(((pathway_name, nes), pval), padj)| EnrichmentResult {
                pathway_name: pathway_name.to_string(),
                size: 0,
                es: 0.0,
                nes: Some(nes.0),
                p_value: pval.0,
                padj: Some(padj.0),
                log2err: None,
                leading_edge: Vec::new(),
            })
            .collect();

        write_gsea_table_plot_png(
            &ranks,
            &pathways,
            &results,
            output_path,
            gsea_param,
            &GseaTablePlotOptions {
                width_inches,
                height_inches,
                dpi: dpi as u32,
                transparent_background,
            },
        )
        .map_err(|err| err.to_string())
    })();

    match result {
        Ok(()) => output_path.into(),
        Err(err) => throw_r_error(err),
    }
}

/// Run fgsea-compatible enrichment using a named stats vector and a GMT file.
#[allow(clippy::too_many_arguments)]
fn fgsea_rust_impl(
    stats: Doubles,
    genes: Strings,
    gmt_path: &str,
    n_perm_simple: i32,
    seed: Nullable<i32>,
    nproc: i32,
    min_size: i32,
    max_size: i32,
    eps: f64,
    score_type: &str,
    gsea_param: f64,
    mode: &str,
    nperm: i32,
    sample_size: i32,
    sample_kind: &str,
    gpu: bool,
    method: &str,
    decor_cache: Nullable<String>,
    decor_expression: Nullable<String>,
    decor_alpha: f64,
    decor_cache_mode: &str,
    decor_correlation: &str,
    decor_redundancy: &str,
    decor_weight_formula: &str,
    decor_threshold: f64,
    decor_gamma: f64,
    decor_penalty_floor: f64,
    decor_scale_epsilon: f64,
    blitz_anchors: i32,
    blitz_symmetric: bool,
    blitz_center: bool,
    blitz_accuracy: i32,
    blitz_deep_accuracy: i32,
    blitz_signature_cache: bool,
) -> std::result::Result<Robj, String> {
    if stats.len() != genes.len() {
        return Err("stats and gene names must have the same length.".to_string());
    }
    if n_perm_simple <= 0 {
        return Err("nPermSimple must be greater than 0.".to_string());
    }
    if nproc < 0 {
        return Err("nproc must be greater than or equal to 0.".to_string());
    }
    if min_size == 0 {
        return Err("minSize must be greater than 0.".to_string());
    }
    if max_size == 0 {
        return Err("maxSize must be greater than 0 when provided.".to_string());
    }
    if sample_size <= 0 {
        return Err("sampleSize must be greater than 0.".to_string());
    }
    if nperm == 0 {
        return Err("nperm must be greater than 0 when provided.".to_string());
    }
    if blitz_anchors <= 0 || blitz_accuracy <= 0 || blitz_deep_accuracy <= 0 {
        return Err(
            "blitz.anchors, blitz.accuracy, and blitz.deep.accuracy must be greater than 0."
                .to_string(),
        );
    }

    let genes: Vec<String> = genes.iter().map(|gene| gene.to_string()).collect();
    let scores: Vec<f64> = stats.iter().map(|v| v.0).collect();
    let ranks = RankedList::new(genes, scores);
    let pathways =
        read_gmt(gmt_path).map_err(|err| format!("Failed to read GMT file '{gmt_path}': {err}"))?;

    let score_type = parse_score_type(score_type)?;
    let sample_kind = parse_sample_kind(sample_kind)?;
    let method = parse_method(method)?;
    let seed = normalize_seed(seed)?;
    let mode = parse_interface_mode(mode).map_err(|err| err.to_string())?;
    let nperm = (nperm >= 0).then_some(nperm as usize);
    let min_size = if min_size > 0 {
        min_size as usize
    } else if mode == InterfaceMode::Blitz {
        5
    } else {
        1
    };
    if mode == InterfaceMode::Blitz {
        if gpu {
            return Err("gpu is not supported with mode = 'blitz'.".to_string());
        }
        if method != EnrichmentMethod::Classic {
            return Err("mode = 'blitz' supports only method = 'classic'.".to_string());
        }
        if nperm.is_some() {
            return Err("nperm is not supported with mode = 'blitz'.".to_string());
        }
        if score_type != ScoreType::Std {
            return Err("mode = 'blitz' supports only scoreType = 'std'.".to_string());
        }
        if (gsea_param - 1.0).abs() > f64::EPSILON {
            return Err("mode = 'blitz' supports only gseaParam = 1.".to_string());
        }
    }
    if decor_alpha < 0.0 || !decor_alpha.is_finite() {
        return Err("decor.alpha must be a finite numeric value >= 0.".to_string());
    }
    if decor_gamma < 0.0 || !decor_gamma.is_finite() {
        return Err("decor.gamma must be a finite numeric value >= 0.".to_string());
    }
    if !(0.0..1.0).contains(&decor_threshold) || !decor_threshold.is_finite() {
        return Err("decor.threshold must be a finite numeric value >= 0 and < 1.".to_string());
    }
    if !(0.0..1.0).contains(&decor_penalty_floor) || !decor_penalty_floor.is_finite() {
        return Err("decor.penalty.floor must be a finite numeric value >= 0 and < 1.".to_string());
    }
    if decor_scale_epsilon <= 0.0 || !decor_scale_epsilon.is_finite() {
        return Err("decor.scale.epsilon must be a finite numeric value > 0.".to_string());
    }
    if method == EnrichmentMethod::Decor {
        if gpu {
            return Err(
                "decor supports CPU execution only; gpu is not supported with method = 'decor'."
                    .to_string(),
            );
        }
        if mode == InterfaceMode::Multilevel && nperm.is_some() {
            return Err("nperm is only valid with mode = 'fgsea' or mode = 'simple'.".to_string());
        }
        let cache_path = decor_cache
            .into_option()
            .ok_or_else(|| "method = 'decor' requires decor.cache.".to_string())?;
        let options = DecorOptions {
            alpha: decor_alpha,
            cache_path: Some(std::path::PathBuf::from(cache_path)),
            expression_path: decor_expression.into_option().map(std::path::PathBuf::from),
            cache_mode: parse_decor_cache_mode(decor_cache_mode)?,
            correlation: parse_decor_correlation(decor_correlation)?,
            redundancy: parse_decor_redundancy(decor_redundancy)?,
            weight_formula: parse_decor_weight_formula(decor_weight_formula)?,
            gamma: decor_gamma,
            threshold_tau: decor_threshold,
            penalty_floor: decor_penalty_floor,
            scale_epsilon: decor_scale_epsilon,
            ..DecorOptions::default()
        };
        let (cache, _) = ensure_decor_cache_for_paths(
            &pathways.pathways,
            std::path::Path::new(gmt_path),
            &options,
            true,
        )
        .map_err(|err| err.to_string())?;
        let max_size = if max_size > 0 {
            max_size as usize
        } else {
            ranks.len().saturating_sub(1)
        };
        let decor_multilevel =
            mode == InterfaceMode::Multilevel || (mode == InterfaceMode::Fgsea && nperm.is_none());
        let results = run_with_optional_thread_pool(nproc, || {
            if decor_multilevel {
                fgsea_decor_multilevel_with_options(
                    &ranks,
                    &pathways.pathways,
                    &cache,
                    &options,
                    n_perm_simple as usize,
                    seed,
                    min_size,
                    max_size,
                    eps,
                    score_type,
                    gsea_param,
                    sample_size as usize,
                )
            } else {
                fgsea_decor_simple_with_options(
                    &ranks,
                    &pathways.pathways,
                    &cache,
                    &options,
                    nperm.unwrap_or(n_perm_simple as usize),
                    seed,
                    min_size,
                    max_size,
                    eps,
                    score_type,
                    gsea_param,
                    sample_size as usize,
                )
            }
        })?
        .map_err(|err| err.to_string())?;

        return results_to_robj(results);
    }
    let exec_mode = resolve_execution_plan(mode, gpu, nperm, n_perm_simple as usize)
        .map_err(|e| e.to_string())?;
    let max_size = if max_size > 0 {
        max_size as usize
    } else if mode == InterfaceMode::Blitz {
        4000
    } else {
        ranks.len().saturating_sub(1)
    };

    if gpu && mode != InterfaceMode::Fgsea {
        return Err("gpu currently supports only mode = 'fgsea'.".to_string());
    }

    let results = match exec_mode {
        ExecutionPlan::Gpu {
            n_perm,
            allow_multilevel,
            ..
        } => run_gpu_path(
            &ranks,
            &pathways.pathways,
            n_perm,
            seed,
            min_size as usize,
            max_size,
            eps,
            score_type,
            gsea_param,
            sample_size as usize,
            allow_multilevel,
            nproc,
        )?,
        _ => run_with_optional_thread_pool(
            nproc,
            || -> std::result::Result<Vec<EnrichmentResult>, String> {
                match exec_mode {
                    ExecutionPlan::Cpu(InterfaceMode::Fgsea) => {
                        Ok(fgsea_with_sample_size_and_kind(
                            &ranks,
                            &pathways.pathways,
                            nperm,
                            n_perm_simple as usize,
                            seed,
                            min_size,
                            max_size,
                            eps,
                            score_type,
                            gsea_param,
                            sample_size as usize,
                            sample_kind,
                        ))
                    }
                    ExecutionPlan::Cpu(InterfaceMode::Simple) => {
                        Ok(fgsea_simple_with_sample_size_and_kind(
                            &ranks,
                            &pathways.pathways,
                            nperm.unwrap_or(n_perm_simple as usize),
                            seed,
                            min_size,
                            max_size,
                            eps,
                            score_type,
                            gsea_param,
                            sample_size as usize,
                            sample_kind,
                        ))
                    }
                    ExecutionPlan::Cpu(InterfaceMode::Multilevel) => {
                        Ok(fgsea_multilevel_with_sample_size_and_kind(
                            &ranks,
                            &pathways.pathways,
                            n_perm_simple as usize,
                            seed,
                            min_size,
                            max_size,
                            eps,
                            score_type,
                            gsea_param,
                            sample_size as usize,
                            sample_kind,
                        ))
                    }
                    ExecutionPlan::Cpu(InterfaceMode::Blitz) => fgsea_blitz_with_options(
                        &ranks,
                        &pathways.pathways,
                        &BlitzOptions {
                            permutations: n_perm_simple as usize,
                            anchors: blitz_anchors as usize,
                            min_size,
                            max_size,
                            processes: if nproc > 0 { nproc as usize } else { 4 },
                            symmetric: blitz_symmetric,
                            seed: seed.unwrap_or(0),
                            center: blitz_center,
                            accuracy: blitz_accuracy as usize,
                            deep_accuracy: blitz_deep_accuracy as usize,
                            signature_cache: blitz_signature_cache,
                        },
                    )
                    .map_err(|err| err.to_string()),
                    ExecutionPlan::Gpu { .. } => unreachable!(),
                }
            },
        )??,
    };

    results_to_robj(results)
}

fn results_to_robj(results: Vec<EnrichmentResult>) -> std::result::Result<Robj, String> {
    let pathway: Vec<String> = results.iter().map(|row| row.pathway_name.clone()).collect();
    let size: Vec<i32> = results.iter().map(|row| row.size as i32).collect();
    let es: Vec<f64> = results.iter().map(|row| row.es).collect();
    let nes: Vec<f64> = results
        .iter()
        .map(|row| row.nes.unwrap_or(f64::NAN))
        .collect();
    let pval: Vec<f64> = results.iter().map(|row| row.p_value).collect();
    let padj: Vec<f64> = results
        .iter()
        .map(|row| row.padj.unwrap_or(f64::NAN))
        .collect();
    let log2err: Vec<f64> = results
        .iter()
        .map(|row| row.log2err.unwrap_or(f64::NAN))
        .collect();
    let leading_edge: Vec<String> = results
        .iter()
        .map(|row| row.leading_edge.join(","))
        .collect();

    Ok(list!(
        pathway = pathway,
        size = size,
        es = es,
        nes = nes,
        pval = pval,
        padj = padj,
        log2err = log2err,
        leadingEdge = leading_edge
    )
    .into())
}

#[allow(clippy::too_many_arguments)]
#[extendr]
fn fgsea_rust(
    stats: Doubles,
    genes: Strings,
    gmt_path: &str,
    n_perm_simple: i32,
    seed: Nullable<i32>,
    nproc: i32,
    min_size: i32,
    max_size: i32,
    eps: f64,
    score_type: &str,
    gsea_param: f64,
    mode: &str,
    nperm: i32,
    sample_size: i32,
    sample_kind: &str,
    gpu: bool,
    method: &str,
    decor_cache: Nullable<String>,
    decor_expression: Nullable<String>,
    decor_alpha: f64,
    decor_cache_mode: &str,
    decor_correlation: &str,
    decor_redundancy: &str,
    decor_weight_formula: &str,
    decor_threshold: f64,
    decor_gamma: f64,
    decor_penalty_floor: f64,
    decor_scale_epsilon: f64,
    blitz_anchors: i32,
    blitz_symmetric: bool,
    blitz_center: bool,
    blitz_accuracy: i32,
    blitz_deep_accuracy: i32,
    blitz_signature_cache: bool,
) -> Robj {
    match fgsea_rust_impl(
        stats,
        genes,
        gmt_path,
        n_perm_simple,
        seed,
        nproc,
        min_size,
        max_size,
        eps,
        score_type,
        gsea_param,
        mode,
        nperm,
        sample_size,
        sample_kind,
        gpu,
        method,
        decor_cache,
        decor_expression,
        decor_alpha,
        decor_cache_mode,
        decor_correlation,
        decor_redundancy,
        decor_weight_formula,
        decor_threshold,
        decor_gamma,
        decor_penalty_floor,
        decor_scale_epsilon,
        blitz_anchors,
        blitz_symmetric,
        blitz_center,
        blitz_accuracy,
        blitz_deep_accuracy,
        blitz_signature_cache,
    ) {
        Ok(obj) => obj,
        Err(err) => throw_r_error(err),
    }
}

extendr_module! {
    mod rsfgseaR;
    fn rsfgsea_version;
    fn supported_modes;
    fn fgsea_rust;
    fn write_enrichment_plot;
    fn write_gsea_table_plot;
}
