#[cfg(test)]
mod tests {
    use csv::StringRecord;
    use rsfgsea::prelude::*;
    use std::collections::HashMap;

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

        let results = fgsea_multilevel_with_sample_size(
            &ranks,
            &pd.pathways,
            1000, // Matches upstream fgseaMultilevel default nPermSimple
            42,
            1,
            500,
            1e-10,
            ScoreType::Std,
            1.0,
            101,
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
                    diff < 1e-6,
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

        assert!(stats.count > 0, "Expected at least one matched pathway.");
        let mean_rel_diff = stats.total_rel_diff / stats.count as f64;
        let median_rel_diff = stats.median();
        assert!(
            mean_rel_diff < 1e-12,
            "Mean relative p-value diff too high: {:.16}",
            mean_rel_diff
        );
        assert!(
            median_rel_diff < 1e-12,
            "Median relative p-value diff too high: {:.16}",
            median_rel_diff
        );
        assert_eq!(
            stats.bins[0], stats.count,
            "Expected all pathways to fall within 0.1% relative p-value diff."
        );
        assert_eq!(stats.bins[1], 0, "Unexpected pathways in 0.1%-1% diff bin.");
        assert_eq!(stats.bins[2], 0, "Unexpected pathways in 1%-5% diff bin.");
        assert_eq!(
            stats.bins[3], 0,
            "Unexpected pathways with >5% relative p-value diff."
        );

        stats.print();
    }

    struct DiffStats {
        total_rel_diff: f64,
        count: usize,
        bins: [usize; 4], // <0.1%, 0.1-1%, 1-5%, >5%
        values: Vec<f64>,
    }

    impl DiffStats {
        fn new() -> Self {
            Self {
                total_rel_diff: 0.0,
                count: 0,
                bins: [0; 4],
                values: Vec::new(),
            }
        }

        fn update(&mut self, diff: f64) {
            self.total_rel_diff += diff;
            self.count += 1;
            self.values.push(diff);
            if diff <= 0.001 {
                self.bins[0] += 1;
            } else if diff <= 0.01 {
                self.bins[1] += 1;
            } else if diff <= 0.05 {
                self.bins[2] += 1;
            } else {
                self.bins[3] += 1;
            }
        }

        fn median(&self) -> f64 {
            let mut values = self.values.clone();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap());
            values[values.len() / 2]
        }

        fn print(&self) {
            println!("\nR vs RSFGSEA Relative P-value Differences (n_perm=5000):");
            if self.count > 0 {
                println!("  Median Relative Diff: {:.2}%", self.median() * 100.0);
                println!(
                    "  Mean Relative Diff:   {:.2}%",
                    (self.total_rel_diff / self.count as f64) * 100.0
                );
                println!("  Distribution:");
                println!(
                    "    < 0.1% diff:        {} ({:.1}%)",
                    self.bins[0],
                    (self.bins[0] as f64 / self.count as f64) * 100.0
                );
                println!(
                    "    0.1% - 1% diff:     {} ({:.1}%)",
                    self.bins[1],
                    (self.bins[1] as f64 / self.count as f64) * 100.0
                );
                println!(
                    "    1% - 5% diff:       {} ({:.1}%)",
                    self.bins[2],
                    (self.bins[2] as f64 / self.count as f64) * 100.0
                );
                println!(
                    "    > 5% diff:          {} ({:.1}%)",
                    self.bins[3],
                    (self.bins[3] as f64 / self.count as f64) * 100.0
                );
            }
        }
    }

    fn read_r_csv(path: &str) -> HashMap<String, (f64, f64)> {
        let mut reader = csv::Reader::from_path(path).expect("Failed to open R reference");
        let mut map = HashMap::new();
        for record in reader.records() {
            let record = record.expect("Failed to parse R reference row");
            let (pathway, es, pval) = parse_r_reference_row(&record);
            map.insert(pathway, (es, pval));
        }
        map
    }

    fn parse_r_reference_row(record: &StringRecord) -> (String, f64, f64) {
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
        (pathway, es, pval)
    }
}
