use crate::core::{Pathway, PathwayDb, RankedList};
use anyhow::{Context, Result};
use std::collections::HashSet;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

pub fn read_ranked_list<P: AsRef<Path>>(path: P) -> Result<RankedList> {
    let file = File::open(path.as_ref())?;
    let reader = BufReader::new(file);

    let mut genes = Vec::new();
    let mut scores = Vec::new();
    let mut seen_genes = HashSet::new();

    for (line_idx, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }

        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 2 {
            anyhow::bail!(
                "Malformed ranked-list line {}: expected at least 2 whitespace-separated columns.",
                line_idx + 1
            );
        }

        let gene = parts[0].to_string();
        let score: f64 = parts[1]
            .parse()
            .with_context(|| format!("Failed to parse score on line {}", line_idx + 1))?;
        if !score.is_finite() {
            anyhow::bail!(
                "Non-finite score '{}' found on line {}. Ranked lists must contain only finite numeric scores.",
                parts[1],
                line_idx + 1
            );
        }

        if !seen_genes.insert(gene.clone()) {
            anyhow::bail!(
                "Duplicate gene '{}' found on line {}. Ranked lists must contain unique gene IDs.",
                gene,
                line_idx + 1
            );
        }

        genes.push(gene);
        scores.push(score);
    }

    Ok(RankedList::new(genes, scores))
}

pub fn read_gmt<P: AsRef<Path>>(path: P) -> Result<PathwayDb> {
    let file = File::open(path.as_ref())?;
    let reader = BufReader::new(file);

    let mut pathways = Vec::new();

    for (line_idx, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }

        let parts: Vec<&str> = line.split('\t').collect();
        if parts.len() < 3 {
            anyhow::bail!(
                "Malformed GMT line {}: expected at least 3 tab-separated columns.",
                line_idx + 1
            );
        }

        let name = parts[0].to_string();
        let description = if parts[1].is_empty() {
            None
        } else {
            Some(parts[1].to_string())
        };
        let genes = parts[2..]
            .iter()
            .map(|&s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();

        pathways.push(Pathway {
            name,
            description,
            genes,
        });
    }

    Ok(PathwayDb { pathways })
}
