use pyo3::prelude::*;
use rsfgsea::prelude::*;
use std::collections::HashMap;

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

#[allow(clippy::too_many_arguments)]
#[allow(non_snake_case)]
#[pyfunction]
#[pyo3(signature = (ranks, gmt_path, nPermSimple=1000, seed=None, nproc=0, minSize=1, maxSize=None, eps=1e-50, scoreType="std", gseaParam=1.0, mode="fgsea", nperm=None, sampleSize=101, gpu=false))]
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
) -> PyResult<Vec<HashMap<String, PyObject>>> {
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
    let mode = parse_interface_mode(mode).map_err(pyo3::exceptions::PyValueError::new_err)?;
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
#[pyo3(signature = (ranks, pathway_genes, output_path, pathway_name="pathway", scoreType="std", gseaParam=1.0, width_inches=4.5, height_inches=3.2, dpi=300, transparent_background=false, title=None))]
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

#[pymodule]
fn rsfgseapy(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_gsea_py, m)?)?;
    m.add_function(wrap_pyfunction!(write_enrichment_plot_png_py, m)?)?;
    Ok(())
}
