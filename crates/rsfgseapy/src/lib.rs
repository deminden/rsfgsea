use pyo3::prelude::*;
use rsfgsea::prelude::*;
use std::collections::HashMap;

#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (ranks, gmt_path, n_perm=1000, seed=42, threads=None, min_size=15, max_size=500, eps=1e-10, score_type="Std", gsea_param=1.0))]
fn run_gsea_py(
    py: Python<'_>,
    ranks: HashMap<String, f64>,
    gmt_path: String,
    n_perm: usize,
    seed: u64,
    threads: Option<usize>,
    min_size: usize,
    max_size: usize,
    eps: f64,
    score_type: &str,
    gsea_param: f64,
) -> PyResult<Vec<HashMap<String, PyObject>>> {
    if let Some(t) = threads {
        let _ = rayon::ThreadPoolBuilder::new()
            .num_threads(t)
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

    let st = match score_type.to_lowercase().as_str() {
        "pos" => ScoreType::Pos,
        "neg" => ScoreType::Neg,
        _ => ScoreType::Std,
    };

    let results = run_gsea(
        &rs_ranks,
        &pd.pathways,
        n_perm,
        seed,
        min_size,
        max_size,
        eps,
        st,
        gsea_param,
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
