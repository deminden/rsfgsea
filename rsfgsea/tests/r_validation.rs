#[cfg(test)]
mod tests {
    use rsfgsea::prelude::*;
    use std::collections::HashMap;
    use std::fs::File;
    use std::io::{BufRead, BufReader};

    #[test]
    fn test_cross_validation_with_r_fgsea() {
        let ref_path = "tests/r_fgsea_reference.csv";
        let ranks_path = "../data/pearson_symbols.rnk";
        let gmt_path = "../data/h.all.v2025.1.Hs.symbols.gmt";

        if !std::path::Path::new(ref_path).exists() || !std::path::Path::new(ranks_path).exists() {
            println!("Skipping R cross-validation: data files not found (expected in CI).");
            return;
        }

        let ref_es = read_r_csv(ref_path);
        let ranks = read_ranked_list(ranks_path).unwrap();
        let pd = read_gmt(gmt_path).unwrap();

        let results = run_gsea(
            &ranks,
            &pd.pathways,
            100,
            42,
            1,
            500,
            1e-10,
            ScoreType::Std,
            1.0,
        );
        let rs_results: HashMap<String, f64> = results
            .into_iter()
            .map(|r| (r.pathway_name, r.es))
            .collect();

        for (pathway, r_es) in ref_es {
            if let Some(&rs_es) = rs_results.get(&pathway) {
                let diff = (r_es - rs_es).abs();
                assert!(
                    diff < 2e-2,
                    "ES mismatch for pathway {}: R={} RS={}",
                    pathway,
                    r_es,
                    rs_es
                );
            }
        }
    }

    fn read_r_csv(path: &str) -> HashMap<String, f64> {
        let file = File::open(path).expect("Failed to open R reference");
        let reader = BufReader::new(file);
        let mut map = HashMap::new();
        let lines = reader.lines().skip(1); // skip header

        for line in lines {
            let line = line.unwrap();
            let parts: Vec<&str> = line.split(',').collect();
            // R CSV: pathway,pval,padj,ES,NES,nMoreExtreme,size,leadingEdge
            if parts.len() >= 4 {
                let pathway = parts[0].trim_matches('"').to_string();
                let es: f64 = parts[3].parse().unwrap();
                map.insert(pathway, es);
            }
        }
        map
    }
}
