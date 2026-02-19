use pyo3::prelude::*;
use rsfgsea::prelude::*;
use std::collections::HashMap;

#[allow(clippy::too_many_arguments)]
#[allow(non_snake_case)]
#[pyfunction]
#[pyo3(signature = (ranks, gmt_path, nPermSimple=1000, seed=42, nproc=0, minSize=1, maxSize=None, eps=1e-50, scoreType="std", gseaParam=1.0, multilevel_engine="esruler"))]
fn run_gsea_py(
    py: Python<'_>,
    ranks: HashMap<String, f64>,
    gmt_path: String,
    nPermSimple: usize,
    seed: u64,
    nproc: usize,
    minSize: usize,
    maxSize: Option<usize>,
    eps: f64,
    scoreType: &str,
    gseaParam: f64,
    multilevel_engine: &str,
) -> PyResult<Vec<HashMap<String, PyObject>>> {
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

    let st = match scoreType.to_lowercase().as_str() {
        "pos" => ScoreType::Pos,
        "neg" => ScoreType::Neg,
        _ => ScoreType::Std,
    };

    unsafe {
        std::env::set_var(
            "RSFGSEA_MULTILEVEL_ENGINE",
            multilevel_engine.to_lowercase(),
        );
    }

    let max_size = maxSize.unwrap_or_else(|| rs_ranks.len().saturating_sub(1));
    let results = run_gsea(
        &rs_ranks,
        &pd.pathways,
        nPermSimple,
        seed,
        minSize,
        max_size,
        eps,
        st,
        gseaParam,
    );

    let mut py_results = Vec::new();
    for res in results {
        let mut map = HashMap::new();
        map.insert(
            "pathway".to_string(),
            res.pathway_name.into_pyobject(py).unwrap().unbind().into(),
        );
        map.insert(
            "size".to_string(),
            res.size.into_pyobject(py).unwrap().unbind().into(),
        );
        map.insert(
            "es".to_string(),
            res.es.into_pyobject(py).unwrap().unbind().into(),
        );
        map.insert(
            "nes".to_string(),
            res.nes.into_pyobject(py).unwrap().unbind(),
        );
        map.insert(
            "pval".to_string(),
            res.p_value.into_pyobject(py).unwrap().unbind().into(),
        );
        map.insert(
            "padj".to_string(),
            res.padj.into_pyobject(py).unwrap().unbind(),
        );
        map.insert(
            "log2err".to_string(),
            res.log2err.into_pyobject(py).unwrap().unbind(),
        );
        map.insert(
            "leading_edge".to_string(),
            res.leading_edge.into_pyobject(py).unwrap().unbind(),
        );
        py_results.push(map);
    }

    Ok(py_results)
}

#[pymodule]
fn rsfgseapy(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_gsea_py, m)?)?;
    Ok(())
}
