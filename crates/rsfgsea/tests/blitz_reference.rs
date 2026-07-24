use rsfgsea::prelude::*;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};

#[derive(Debug, Deserialize)]
struct BlitzExpected {
    #[serde(rename = "Term")]
    term: String,
    es: f64,
    nes: f64,
    pval: f64,
    fdr: f64,
    geneset_size: usize,
    leading_edge: String,
}

#[test]
fn blitz_reference_fixture_tracks_locked_stack() {
    assert_blitz_fixture(
        "synthetic",
        read_ranked_list("tests/data/blitz_reference/synthetic.rnk").unwrap(),
    );
}

#[test]
fn blitz_edgecase_fixture_tracks_locked_stack() {
    assert_blitz_fixture(
        "edgecases",
        read_ranked_list_allowing_duplicates("tests/data/blitz_reference/edgecases.rnk"),
    );
}

#[test]
fn blitz_publication_fixture_tracks_locked_stack() {
    assert_blitz_fixture(
        "publication_fgsea",
        read_ranked_list("tests/data/blitz_reference/publication_fgsea.rnk").unwrap(),
    );
}

fn assert_blitz_fixture(prefix: &str, ranks: RankedList) {
    let pathways = read_gmt(format!("tests/data/blitz_reference/{prefix}.gmt"))
        .unwrap()
        .pathways;
    let observed = fgsea_blitz_with_options(&ranks, &pathways, &BlitzOptions::default()).unwrap();
    let observed_by_name = observed
        .iter()
        .map(|row| (row.pathway_name.as_str(), row))
        .collect::<HashMap<_, _>>();

    let mut reader = csv::ReaderBuilder::new()
        .delimiter(b'\t')
        .from_path(format!("tests/data/blitz_reference/{prefix}.expected.tsv"))
        .unwrap();
    let expected = reader
        .deserialize::<BlitzExpected>()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();

    assert_eq!(observed.len(), expected.len());
    assert_eq!(
        observed
            .iter()
            .map(|row| row.pathway_name.as_str())
            .collect::<Vec<_>>(),
        expected
            .iter()
            .map(|row| row.term.as_str())
            .collect::<Vec<_>>(),
        "{prefix} result ordering differs"
    );
    for expected_row in expected {
        let observed_row = observed_by_name
            .get(expected_row.term.as_str())
            .unwrap_or_else(|| panic!("missing blitz result {}", expected_row.term));
        assert_eq!(observed_row.size, expected_row.geneset_size);
        assert_eq!(observed_row.leading_edge_csv(), expected_row.leading_edge);
        assert!(observed_row.log2err.is_none());
        assert_exact_bits(
            &format!("{} ES", expected_row.term),
            observed_row.es,
            expected_row.es,
        );
        assert_exact_bits(
            &format!("{} pval", expected_row.term),
            observed_row.p_value,
            expected_row.pval,
        );
        assert_exact_bits(
            &format!("{} padj", expected_row.term),
            observed_row.padj.unwrap(),
            expected_row.fdr,
        );
        assert_exact_bits(
            &format!("{} NES", expected_row.term),
            observed_row.nes.unwrap(),
            expected_row.nes,
        );
    }
}

fn assert_exact_bits(label: &str, observed: f64, expected: f64) {
    assert_eq!(
        observed.to_bits(),
        expected.to_bits(),
        "{label} differs: observed {observed}, expected {expected}"
    );
}

fn read_ranked_list_allowing_duplicates(path: &str) -> RankedList {
    let file = File::open(path).unwrap();
    let reader = BufReader::new(file);
    let mut genes = Vec::new();
    let mut scores = Vec::new();
    for (line_idx, line) in reader.lines().enumerate() {
        let line = line.unwrap();
        if line.trim().is_empty() {
            continue;
        }
        let parts = line.split_whitespace().collect::<Vec<_>>();
        assert!(
            parts.len() >= 2,
            "malformed ranked-list line {} in {path}",
            line_idx + 1
        );
        genes.push(parts[0].to_string());
        scores.push(parts[1].parse::<f64>().unwrap());
    }
    RankedList::new(genes, scores)
}
