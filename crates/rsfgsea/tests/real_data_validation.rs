#[cfg(test)]
mod tests {
    use rsfgsea::prelude::*;
    use std::collections::HashMap;
    use std::fs::File;
    use std::io::{BufRead, BufReader};
    use std::path::Path;

    #[test]
    fn test_muscle_data_comparison() {
        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let root = Path::new(&manifest_dir);
        let data_dir = root.join("tests/data/muscle_comparison");
        let ref_csv_path = data_dir.join("r_fgsea_results.csv");
        let gmt_path = root.join("../../data/h.all.v2025.1.Hs.symbols.gmt");

        if !ref_csv_path.exists() {
            println!("Skipping muscle data comparison: data not found.");
            return;
        }

        // 1. Load Reference Data
        let ref_data = read_r_csv(ref_csv_path.to_str().unwrap());
        // Map: Gene -> (Pathway -> (ES, Pval))
        let mut ref_map: HashMap<String, HashMap<String, (f64, f64)>> = HashMap::new();

        for (gene, pathway, es, pval) in ref_data {
            ref_map.entry(gene).or_default().insert(pathway, (es, pval));
        }

        println!("Loaded R results for {} genes.", ref_map.len());

        // 2. Load Pathways
        let pathway_db =
            rsfgsea::io::read_gmt(gmt_path.to_str().unwrap()).expect("Failed to load GMT");

        // 3. Process each gene
        let mut all_es_errors = Vec::new();
        let mut all_pval_diffs = Vec::new();
        let mut total_pathways_checked = 0;

        let mut sorted_genes: Vec<_> = ref_map.keys().cloned().collect();
        sorted_genes.sort();

        for gene in sorted_genes {
            let rnk_path = data_dir.join(format!("{}.rnk", gene));
            if !rnk_path.exists() {
                continue;
            }

            let ranks = rsfgsea::io::read_ranked_list(rnk_path.to_str().unwrap())
                .expect("Failed to read ranks");

            // Run GSEA
            // Using same params as R script: min=15, max=500, perm=1000
            #[cfg(feature = "gpu")]
            let results = rsfgsea::algo::run_gsea_gpu(
                &ranks,
                &pathway_db.pathways,
                1000,
                42,
                15,
                500,
                ScoreType::Std,
                1.0,
            )
            .expect("GPU run failed");

            // Fallback for non-GPU env
            #[cfg(not(feature = "gpu"))]
            let results = run_gsea(
                &ranks,
                &pathway_db.pathways,
                1000,
                42,
                15,
                500,
                1e-10, // eps
                ScoreType::Std,
                1.0,
            );

            // Compare
            if let Some(gene_refs) = ref_map.get(&gene) {
                for res in results {
                    if let Some(&(r_es, r_pval)) = gene_refs.get(&res.pathway_name) {
                        total_pathways_checked += 1;
                        let es_diff = (res.es - r_es).abs();
                        all_es_errors.push(es_diff);

                        let rel_denom = r_pval.max(1e-10);
                        let pval_rel_diff = (res.p_value - r_pval).abs() / rel_denom;
                        all_pval_diffs.push(pval_rel_diff);

                        // Strict check on ES
                        assert!(
                            es_diff < 1e-4,
                            "Major ES mismatch for {}:{} R={:.6} Rust={:.6}",
                            gene,
                            res.pathway_name,
                            r_es,
                            res.es
                        );
                    }
                }
            }
        }

        // 4. Report stats
        if total_pathways_checked > 0 {
            let mean_es_err: f64 =
                all_es_errors.iter().sum::<f64>() / total_pathways_checked as f64;
            let mean_pval_diff: f64 =
                all_pval_diffs.iter().sum::<f64>() / total_pathways_checked as f64;
            // Median pval diff
            all_pval_diffs.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let median_pval_diff = all_pval_diffs[all_pval_diffs.len() / 2];

            println!(
                "\n=== Real Data Validation Stats ({} pathways) ===",
                total_pathways_checked
            );
            println!("Mean ES Error: {:.8}", mean_es_err);
            println!("Mean Rel P-val Diff: {:.4}", mean_pval_diff);
            println!("Median Rel P-val Diff: {:.4}", median_pval_diff);

            // Allow some variance in p-values due to random sampling differences (1000 perms)
            assert!(mean_es_err < 1e-5, "Overall ES error too high");
            // P-values will differ more, just ensure median isn't catastrophic
            assert!(median_pval_diff < 1.0, "Median p-value disagreement > 100%");
        }
    }

    fn read_r_csv(path: &str) -> Vec<(String, String, f64, f64)> {
        let file = File::open(path).expect("Failed to open R reference");
        let reader = BufReader::new(file);
        let mut data = Vec::new();

        // Header: "pathway","pval","padj","ES","NES","nMoreExtreme","size","leadingEdge","TargetGene"

        // Simple CSV parser (assuming no commas in fields for now, R usually quotes)
        // Using crate `csv` would be better but trying to avoid adding deps for test
        let mut lines = reader.lines();
        let _header = lines.next(); // Skip header

        for line in lines {
            let line = line.unwrap();
            // Need robust splitting because of quoted leadingEdge with pipes
            // HACK: Split by `","` which is R's typical quote+comma+quote
            // But first/last elements only have one quote.

            // Let's assume standard R CSV output: "val","val",...
            let clean_line = line.trim_matches('"');
            let parts: Vec<&str> = clean_line.split("\",\"").collect();

            if parts.len() >= 9 {
                let pathway = parts[0].to_string();
                let pval: f64 = parts[1].parse().unwrap_or(1.0);
                let es: f64 = parts[3].parse().unwrap_or(0.0);
                let gene = parts[8].to_string();

                data.push((gene, pathway, es, pval));
            }
        }
        data
    }
}
