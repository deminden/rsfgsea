#![allow(non_snake_case)]

use extendr_api::prelude::*;
use extendr_api::throw_r_error;
use rayon::ThreadPoolBuilder;
use rsfgsea::prelude::*;

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
    vec!["fgsea", "simple", "multilevel"]
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
    gpu: bool,
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
    if min_size <= 0 {
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

    let genes: Vec<String> = genes.iter().map(|gene| gene.to_string()).collect();
    let scores: Vec<f64> = stats.iter().map(|v| v.inner()).collect();
    let ranks = RankedList::new(genes, scores);
    let pathways =
        read_gmt(gmt_path).map_err(|err| format!("Failed to read GMT file '{gmt_path}': {err}"))?;

    let score_type = parse_score_type(score_type)?;
    let seed = normalize_seed(seed)?;
    let mode = parse_interface_mode(mode).map_err(|err| err.to_string())?;
    let nperm = (nperm >= 0).then_some(nperm as usize);
    let exec_mode =
        resolve_execution_plan(mode, gpu, nperm, n_perm_simple as usize).map_err(|e| e.to_string())?;
    let max_size = if max_size > 0 {
        max_size as usize
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
        _ => run_with_optional_thread_pool(nproc, || match exec_mode {
            ExecutionPlan::Cpu(InterfaceMode::Fgsea) => fgsea_with_sample_size(
                &ranks,
                &pathways.pathways,
                nperm,
                n_perm_simple as usize,
                seed,
                min_size as usize,
                max_size,
                eps,
                score_type,
                gsea_param,
                sample_size as usize,
            ),
            ExecutionPlan::Cpu(InterfaceMode::Simple) => fgsea_simple_with_sample_size(
                &ranks,
                &pathways.pathways,
                nperm.unwrap_or(n_perm_simple as usize),
                seed,
                min_size as usize,
                max_size,
                eps,
                score_type,
                gsea_param,
                sample_size as usize,
            ),
            ExecutionPlan::Cpu(InterfaceMode::Multilevel) => fgsea_multilevel_with_sample_size(
                &ranks,
                &pathways.pathways,
                n_perm_simple as usize,
                seed,
                min_size as usize,
                max_size,
                eps,
                score_type,
                gsea_param,
                sample_size as usize,
            ),
            ExecutionPlan::Gpu { .. } => unreachable!(),
        })?,
    };

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
    gpu: bool,
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
        gpu,
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
}
