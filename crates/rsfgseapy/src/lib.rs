use pyo3::Py;
use pyo3::prelude::*;
use pyo3::types::PyAny;
use rsfgsea::prelude::*;
use std::collections::HashMap;
use std::path::PathBuf;
use std::str::FromStr;

fn parse_score_type(score_type: &str) -> PyResult<ScoreType> {
    match score_type.to_lowercase().as_str() {
        "std" => Ok(ScoreType::Std),
        "pos" => Ok(ScoreType::Pos),
        "neg" => Ok(ScoreType::Neg),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Invalid scoreType '{}'. Expected one of: std, pos, neg.",
            other
        ))),
    }
}

fn parse_method(method: &str) -> PyResult<EnrichmentMethod> {
    match method.to_lowercase().as_str() {
        "classic" => Ok(EnrichmentMethod::Classic),
        "decor" => Ok(EnrichmentMethod::Decor),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Invalid method '{}'. Expected one of: classic, decor.",
            other
        ))),
    }
}

fn parse_decor_cache_mode(mode: &str) -> PyResult<DecorCacheMode> {
    match mode.to_lowercase().as_str() {
        "auto" => Ok(DecorCacheMode::Auto),
        "reuse" => Ok(DecorCacheMode::Reuse),
        "rebuild" => Ok(DecorCacheMode::Rebuild),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Invalid decor_cache_mode '{}'. Expected one of: auto, reuse, rebuild.",
            other
        ))),
    }
}

fn parse_decor_correlation(correlation: &str) -> PyResult<DecorCorrelation> {
    match correlation.to_lowercase().as_str() {
        "pearson" => Ok(DecorCorrelation::Pearson),
        "spearman" => Ok(DecorCorrelation::Spearman),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Invalid decor_correlation '{}'. Expected one of: pearson, spearman.",
            other
        ))),
    }
}

fn parse_decor_redundancy(redundancy: &str) -> PyResult<DecorRedundancy> {
    match redundancy.to_lowercase().as_str() {
        "positive_mean" => Ok(DecorRedundancy::PositiveMean),
        "abs_mean" => Ok(DecorRedundancy::AbsMean),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Invalid decor_redundancy '{}'. Expected one of: positive_mean, abs_mean.",
            other
        ))),
    }
}

fn parse_decor_preset(preset: &str) -> PyResult<DecorPreset> {
    DecorPreset::from_str(preset).map_err(pyo3::exceptions::PyValueError::new_err)
}

fn apply_decor_release_tuning(
    options: &mut DecorOptions,
    decor_preset: Option<&str>,
    decor_stringency: Option<f64>,
) -> PyResult<()> {
    if decor_preset.is_some() && decor_stringency.is_some() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Use either decor_preset or decor_stringency, not both.",
        ));
    }

    if let Some(stringency) = decor_stringency {
        options
            .apply_stringency(stringency)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
    } else {
        options.apply_preset(parse_decor_preset(decor_preset.unwrap_or("balanced"))?);
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
#[allow(non_snake_case)]
#[pyfunction]
#[pyo3(signature = (ranks, gmt_path, nPermSimple=1000, seed=None, nproc=0, minSize=1, maxSize=None, eps=1e-50, scoreType="std", gseaParam=1.0, mode="fgsea", nperm=None, sampleSize=101, gpu=false, method="classic", decor_cache=None, decor_expression=None, decor_preset=None, decor_stringency=None, decor_cache_mode="auto", decor_correlation="pearson", decor_redundancy="positive_mean"))]
fn run_gsea_py(
    py: Python<'_>,
    ranks: HashMap<String, f64>,
    gmt_path: String,
    nPermSimple: usize,
    seed: Option<u64>,
    nproc: usize,
    minSize: usize,
    maxSize: Option<usize>,
    eps: f64,
    scoreType: &str,
    gseaParam: f64,
    mode: &str,
    nperm: Option<usize>,
    sampleSize: usize,
    gpu: bool,
    method: &str,
    decor_cache: Option<String>,
    decor_expression: Option<String>,
    decor_preset: Option<&str>,
    decor_stringency: Option<f64>,
    decor_cache_mode: &str,
    decor_correlation: &str,
    decor_redundancy: &str,
) -> PyResult<Vec<HashMap<String, Py<PyAny>>>> {
    if sampleSize == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "sampleSize must be greater than 0.",
        ));
    }

    if nproc > 0 {
        let _ = rayon::ThreadPoolBuilder::new()
            .num_threads(nproc)
            .build_global();
    }

    let mut genes = Vec::new();
    let mut scores = Vec::new();
    for (g, s) in ranks {
        genes.push(g);
        scores.push(s);
    }

    let rs_ranks = RankedList::new(genes, scores);
    let pd =
        read_gmt(&gmt_path).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

    let st = parse_score_type(scoreType)?;
    let method = parse_method(method)?;
    let mode = parse_interface_mode(mode).map_err(pyo3::exceptions::PyValueError::new_err)?;
    if method == EnrichmentMethod::Decor {
        if gpu
            || mode == InterfaceMode::Multilevel
            || (mode == InterfaceMode::Fgsea && nperm.is_none())
        {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "decor supports CPU fixed-permutation simple runs; use mode='simple' or provide nperm without gpu.",
            ));
        }
        let cache_path = decor_cache.ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("method='decor' requires decor_cache.")
        })?;
        let mut options = DecorOptions::default();
        apply_decor_release_tuning(&mut options, decor_preset, decor_stringency)?;
        options.cache_path = Some(PathBuf::from(cache_path));
        options.expression_path = decor_expression.map(PathBuf::from);
        options.cache_mode = parse_decor_cache_mode(decor_cache_mode)?;
        options.correlation = parse_decor_correlation(decor_correlation)?;
        options.redundancy = parse_decor_redundancy(decor_redundancy)?;
        let (cache, _) = ensure_decor_cache_for_paths(
            &pd.pathways,
            PathBuf::from(&gmt_path).as_path(),
            &options,
            true,
        )
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        let max_size = maxSize.unwrap_or_else(|| rs_ranks.len().saturating_sub(1));
        let results = fgsea_decor_simple_with_options(
            &rs_ranks,
            &pd.pathways,
            &cache,
            &options,
            nperm.unwrap_or(nPermSimple),
            seed,
            minSize,
            max_size,
            eps,
            st,
            gseaParam,
            sampleSize,
        )
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        return results_to_py(py, results);
    } else if decor_cache.is_some() || decor_expression.is_some() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "decor arguments require method='decor'.",
        ));
    }
    let exec_mode = if gpu {
        #[cfg(not(feature = "gpu"))]
        {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "gpu=True requires building rsfgseapy with the 'gpu' feature.",
            ));
        }

        #[cfg(feature = "gpu")]
        {
            resolve_execution_plan(mode, true, nperm, nPermSimple)
                .map_err(pyo3::exceptions::PyValueError::new_err)?
        }
    } else {
        resolve_execution_plan(mode, false, nperm, nPermSimple)
            .map_err(pyo3::exceptions::PyValueError::new_err)?
    };
    let max_size = maxSize.unwrap_or_else(|| rs_ranks.len().saturating_sub(1));
    let results = match exec_mode {
        ExecutionPlan::Cpu(InterfaceMode::Fgsea) => fgsea_with_sample_size(
            &rs_ranks,
            &pd.pathways,
            nperm,
            nPermSimple,
            seed,
            minSize,
            max_size,
            eps,
            st,
            gseaParam,
            sampleSize,
        ),
        ExecutionPlan::Cpu(InterfaceMode::Multilevel) => fgsea_multilevel_with_sample_size(
            &rs_ranks,
            &pd.pathways,
            nPermSimple,
            seed,
            minSize,
            max_size,
            eps,
            st,
            gseaParam,
            sampleSize,
        ),
        ExecutionPlan::Cpu(InterfaceMode::Simple) => fgsea_simple_with_sample_size(
            &rs_ranks,
            &pd.pathways,
            nperm.unwrap_or(nPermSimple),
            seed,
            minSize,
            max_size,
            eps,
            st,
            gseaParam,
            sampleSize,
        ),
        #[cfg(feature = "gpu")]
        ExecutionPlan::Gpu {
            n_perm,
            allow_multilevel,
            ..
        } => run_gsea_gpu_with_config(
            &rs_ranks,
            &pd.pathways,
            n_perm,
            seed,
            minSize,
            max_size,
            eps,
            st,
            gseaParam,
            sampleSize,
            allow_multilevel,
        )
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?,
        #[cfg(not(feature = "gpu"))]
        ExecutionPlan::Gpu { .. } => {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "gpu=True requires building rsfgseapy with the 'gpu' feature.",
            ));
        }
    };

    results_to_py(py, results)
}

fn results_to_py(
    py: Python<'_>,
    results: Vec<EnrichmentResult>,
) -> PyResult<Vec<HashMap<String, Py<PyAny>>>> {
    let mut py_results = Vec::new();
    for res in results {
        let export = res.export();
        let mut map = HashMap::new();
        map.insert(
            "pathway".to_string(),
            export.pathway.into_pyobject(py)?.unbind().into(),
        );
        map.insert(
            "size".to_string(),
            export.size.into_pyobject(py)?.unbind().into(),
        );
        map.insert(
            "es".to_string(),
            export.es.into_pyobject(py)?.unbind().into(),
        );
        map.insert("nes".to_string(), export.nes.into_pyobject(py)?.unbind());
        map.insert(
            "pval".to_string(),
            export.pval.into_pyobject(py)?.unbind().into(),
        );
        map.insert("padj".to_string(), export.padj.into_pyobject(py)?.unbind());
        map.insert(
            "log2err".to_string(),
            export.log2err.into_pyobject(py)?.unbind(),
        );
        map.insert(
            "leading_edge".to_string(),
            export.leading_edge.into_pyobject(py)?.unbind(),
        );
        py_results.push(map);
    }

    Ok(py_results)
}

#[allow(clippy::too_many_arguments)]
#[allow(non_snake_case)]
#[pyfunction]
#[pyo3(signature = (ranks, pathway_genes, output_path, pathway_name="pathway", scoreType="std", gseaParam=1.0, width_inches=3.0, height_inches=2.2, dpi=300, transparent_background=false, title=None))]
fn write_enrichment_plot_png_py(
    ranks: HashMap<String, f64>,
    pathway_genes: Vec<String>,
    output_path: String,
    pathway_name: &str,
    scoreType: &str,
    gseaParam: f64,
    width_inches: f64,
    height_inches: f64,
    dpi: u32,
    transparent_background: bool,
    title: Option<String>,
) -> PyResult<()> {
    let st = parse_score_type(scoreType)?;
    let mut genes = Vec::new();
    let mut scores = Vec::new();
    for (g, s) in ranks {
        genes.push(g);
        scores.push(s);
    }

    let rs_ranks = RankedList::new(genes, scores);
    let pathway = Pathway {
        name: pathway_name.to_string(),
        description: None,
        genes: pathway_genes,
    };

    write_enrichment_plot_png(
        &rs_ranks,
        &pathway,
        output_path,
        st,
        gseaParam,
        &EnrichmentPlotOptions {
            width_inches,
            height_inches,
            dpi,
            transparent_background,
            title,
        },
    )
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

#[allow(clippy::too_many_arguments)]
#[allow(non_snake_case)]
#[pyfunction]
#[pyo3(signature = (ranks, pathways, results, output_path, gseaParam=1.0, width_inches=5.6, height_inches=None, dpi=300, transparent_background=false))]
fn write_gsea_table_plot_png_py(
    py: Python<'_>,
    ranks: HashMap<String, f64>,
    pathways: Vec<(String, Vec<String>)>,
    results: Vec<HashMap<String, Py<PyAny>>>,
    output_path: String,
    gseaParam: f64,
    width_inches: f64,
    height_inches: Option<f64>,
    dpi: u32,
    transparent_background: bool,
) -> PyResult<()> {
    let mut genes = Vec::new();
    let mut scores = Vec::new();
    for (g, s) in ranks {
        genes.push(g);
        scores.push(s);
    }
    let rs_ranks = RankedList::new(genes, scores);

    let pathways: Vec<Pathway> = pathways
        .into_iter()
        .map(|(name, genes)| Pathway {
            name,
            description: None,
            genes,
        })
        .collect();
    let results = parse_results_for_plot_py(py, results)?;

    write_gsea_table_plot_png(
        &rs_ranks,
        &pathways,
        &results,
        output_path,
        gseaParam,
        &GseaTablePlotOptions {
            width_inches,
            height_inches,
            dpi,
            transparent_background,
        },
    )
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

fn parse_results_for_plot_py(
    py: Python<'_>,
    results: Vec<HashMap<String, Py<PyAny>>>,
) -> PyResult<Vec<EnrichmentResult>> {
    let mut parsed = Vec::with_capacity(results.len());
    for row in results {
        let pathway_name = extract_required_py_string(py, &row, "pathway")?;
        let nes = extract_required_py_f64(py, &row, "nes")?;
        let pval = extract_required_py_f64(py, &row, "pval")?;
        let padj = extract_required_py_f64(py, &row, "padj")?;
        parsed.push(EnrichmentResult {
            pathway_name,
            size: 0,
            es: 0.0,
            nes: Some(nes),
            p_value: pval,
            padj: Some(padj),
            log2err: None,
            leading_edge: Vec::new(),
        });
    }
    Ok(parsed)
}

fn extract_required_py_string(
    py: Python<'_>,
    row: &HashMap<String, Py<PyAny>>,
    key: &str,
) -> PyResult<String> {
    row.get(key)
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!("Result row missing '{key}'."))
        })?
        .bind(py)
        .extract::<String>()
}

fn extract_required_py_f64(
    py: Python<'_>,
    row: &HashMap<String, Py<PyAny>>,
    key: &str,
) -> PyResult<f64> {
    row.get(key)
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!("Result row missing '{key}'."))
        })?
        .bind(py)
        .extract::<f64>()
}

#[pymodule]
fn rsfgseapy(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_gsea_py, m)?)?;
    m.add_function(wrap_pyfunction!(write_enrichment_plot_png_py, m)?)?;
    m.add_function(wrap_pyfunction!(write_gsea_table_plot_png_py, m)?)?;
    Ok(())
}
