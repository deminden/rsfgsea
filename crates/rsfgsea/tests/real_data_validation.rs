#[cfg(test)]
mod tests {
    use csv::StringRecord;
    use rsfgsea::prelude::*;
    use std::collections::HashMap;
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

        assert!(
            !ref_map.is_empty(),
            "R muscle comparison reference is empty; regenerate crates/rsfgsea/tests/data/muscle_comparison/r_fgsea_results.csv"
        );
        println!("Loaded R results for {} genes.", ref_map.len());

        // 2. Load Pathways
        let pathway_db =
            rsfgsea::io::read_gmt(gmt_path.to_str().unwrap()).expect("Failed to load GMT");

        // 3. Process each gene
        let mut all_es_errors = Vec::new();
        let mut all_pval_diffs = Vec::new();
        let mut total_pathways_checked = 0;
        let mut outliers_gt_0_1pct = 0usize;
        let mut outliers_gt_1pct = 0usize;

        let mut sorted_genes: Vec<_> = ref_map.keys().cloned().collect();
        sorted_genes.sort();

        for gene in sorted_genes {
            let rnk_path = data_dir.join(format!("{}.rnk", gene));
            if !rnk_path.exists() {
                continue;
            }

            let ranks = rsfgsea::io::read_ranked_list(rnk_path.to_str().unwrap())
                .expect("Failed to read ranks");

            // Match the upstream-R reference generation exactly: CPU multilevel
            // with fixed seed and the same size filters.
            let results = fgsea_multilevel_with_sample_size(
                &ranks,
                &pathway_db.pathways,
                1000,
                42,
                15,
                500,
                1e-10, // eps
                ScoreType::Std,
                1.0,
                101,
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
                        if pval_rel_diff > 0.001 {
                            outliers_gt_0_1pct += 1;
                        }
                        if pval_rel_diff > 0.01 {
                            outliers_gt_1pct += 1;
                        }

                        // Strict check on ES
                        assert!(
                            es_diff < 1e-6,
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

        assert!(
            total_pathways_checked > 0,
            "No pathway rows were compared; reference or test inputs are inconsistent."
        );

        // 4. Report stats
        let mean_es_err: f64 = all_es_errors.iter().sum::<f64>() / total_pathways_checked as f64;
        let mean_pval_diff: f64 =
            all_pval_diffs.iter().sum::<f64>() / total_pathways_checked as f64;
        all_pval_diffs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_pval_diff = all_pval_diffs[all_pval_diffs.len() / 2];

        println!(
            "\n=== Real Data Validation Stats ({} pathways) ===",
            total_pathways_checked
        );
        println!("Mean ES Error: {:.8}", mean_es_err);
        println!("Mean Rel P-val Diff: {:.4}", mean_pval_diff);
        println!("Median Rel P-val Diff: {:.4}", median_pval_diff);
        println!("Outliers >0.1%: {}", outliers_gt_0_1pct);
        println!("Outliers >1%: {}", outliers_gt_1pct);

        assert!(mean_es_err < 1e-12, "Overall ES error too high");
        assert!(
            mean_pval_diff < 1e-12,
            "Mean relative p-value disagreement too high: {:.4}",
            mean_pval_diff
        );
        assert!(
            median_pval_diff < 1e-12,
            "Median relative p-value disagreement too high: {:.4}",
            median_pval_diff
        );
        assert!(
            outliers_gt_0_1pct == 0,
            "Too many pathways with >0.1% relative p-value disagreement: {}",
            outliers_gt_0_1pct
        );
        assert!(
            outliers_gt_1pct == 0,
            "Too many pathways with >1% relative p-value disagreement: {}",
            outliers_gt_1pct
        );
    }

    fn read_r_csv(path: &str) -> Vec<(String, String, f64, f64)> {
        let mut reader = csv::Reader::from_path(path).expect("Failed to open R reference");
        let mut data = Vec::new();
        for record in reader.records() {
            let record = record.expect("Failed to parse R reference row");
            data.push(parse_real_data_record(&record));
        }
        data
    }

    fn parse_real_data_record(record: &StringRecord) -> (String, String, f64, f64) {
        let pathway = record.get(0).expect("Missing pathway column").to_string();
        let pval = record
            .get(1)
            .expect("Missing pval column")
            .parse()
            .expect("Failed to parse pval");
        let es = record
            .get(3)
            .expect("Missing ES column")
            .parse()
            .expect("Failed to parse ES");
        let gene = record
            .get(6)
            .expect("Missing TargetGene column")
            .to_string();

        (gene, pathway, es, pval)
    }
}
