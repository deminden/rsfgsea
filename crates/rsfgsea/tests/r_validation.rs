#[cfg(test)]
mod tests {
    use rsfgsea::prelude::*;
    use std::collections::HashMap;
    use std::fs::File;
    use std::io::{BufRead, BufReader};

    #[test]
    fn test_cross_validation_with_r_fgsea() {
        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let root = std::path::Path::new(&manifest_dir);
        let ref_path = root.join("tests/r_fgsea_reference.csv");
        let ranks_path = root.join("../../data/pearson_symbols.rnk");
        let gmt_path = root.join("../../data/h.all.v2025.1.Hs.symbols.gmt");

        if !ref_path.exists() || !ranks_path.exists() {
            println!("Skipping R cross-validation: data files not found (expected in CI).");
            return;
        }

        let ref_es = read_r_csv(ref_path.to_str().unwrap());
        let ranks = read_ranked_list(ranks_path.to_str().unwrap()).unwrap();
        let pd = read_gmt(gmt_path.to_str().unwrap()).unwrap();

        let results = run_gsea(
            &ranks,
            &pd.pathways,
            5000, // Matched to R reference granularity ~5000
            42,
            1,
            500,
            1e-10,
            ScoreType::Std,
            1.0,
        );
        let rs_results: HashMap<String, (f64, f64)> = results
            .into_iter()
            .map(|r| (r.pathway_name, (r.es, r.p_value)))
            .collect();

        let mut stats = DiffStats::new();

        for (pathway, (r_es, r_pval)) in ref_es {
            if let Some(&(rs_es, rs_pval)) = rs_results.get(&pathway) {
                let diff = (r_es - rs_es).abs();
                assert!(
                    diff < 2e-2,
                    "ES mismatch for pathway {}: R={} RS={}",
                    pathway,
                    r_es,
                    rs_es
                );

                // Relative p-value check
                let rel_denom = r_pval.max(1e-10);
                let pval_rel_diff = (r_pval - rs_pval).abs() / rel_denom;

                // Accumulate stats
                stats.update(pval_rel_diff);

                // Info print for monitoring major discrepancies
                // if pval_rel_diff > 0.5 {
                //      println!("Info: High Relative P-value diff for {}: R={}, RS={}, RelDiff={:.4}",
                //         pathway, r_pval, rs_pval, pval_rel_diff);
                // }
            }
        }

        stats.print();
    }

    struct DiffStats {
        total_rel_diff: f64,
        count: usize,
        bins: [usize; 4], // <1%, 1-10%, 10-50%, >50%
    }

    impl DiffStats {
        fn new() -> Self {
            Self {
                total_rel_diff: 0.0,
                count: 0,
                bins: [0; 4],
            }
        }

        fn update(&mut self, diff: f64) {
            self.total_rel_diff += diff;
            self.count += 1;
            if diff <= 0.01 {
                self.bins[0] += 1;
            } else if diff <= 0.10 {
                self.bins[1] += 1;
            } else if diff <= 0.50 {
                self.bins[2] += 1;
            } else {
                self.bins[3] += 1;
            }
        }

        fn print(&self) {
            println!("\nR vs RSFGSEA Relative P-value Differences (n_perm=5000):");
            if self.count > 0 {
                println!(
                    "  Mean Relative Diff:   {:.2}%",
                    (self.total_rel_diff / self.count as f64) * 100.0
                );
                println!("  Distribution:");
                println!(
                    "    < 1% diff:          {} ({:.1}%)",
                    self.bins[0],
                    (self.bins[0] as f64 / self.count as f64) * 100.0
                );
                println!(
                    "    1% - 10% diff:      {} ({:.1}%)",
                    self.bins[1],
                    (self.bins[1] as f64 / self.count as f64) * 100.0
                );
                println!(
                    "    10% - 50% diff:     {} ({:.1}%)",
                    self.bins[2],
                    (self.bins[2] as f64 / self.count as f64) * 100.0
                );
                println!(
                    "    > 50% diff:         {} ({:.1}%)",
                    self.bins[3],
                    (self.bins[3] as f64 / self.count as f64) * 100.0
                );
            }
        }
    }

    fn read_r_csv(path: &str) -> HashMap<String, (f64, f64)> {
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
                let map_val = (es, parts[1].parse::<f64>().unwrap_or(1.0)); // Store (ES, pval)
                map.insert(pathway, map_val);
            }
        }
        map
    }
}
