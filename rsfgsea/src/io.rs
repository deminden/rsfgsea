use crate::core::{Pathway, PathwayDb, RankedList};
use anyhow::{Context, Result};
use std::fs::File;
use std::io::{BufRead, BufReader};

pub fn read_ranked_list(path: &str) -> Result<RankedList> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut genes = Vec::new();
    let mut scores = Vec::new();

    for (line_idx, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }

        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 2 {
            continue; // Or error?
        }

        let gene = parts[0].to_string();
        let score: f64 = parts[1]
            .parse()
            .with_context(|| format!("Failed to parse score on line {}", line_idx + 1))?;

        genes.push(gene);
        scores.push(score);
    }

    Ok(RankedList::new(genes, scores))
}

pub fn read_gmt(path: &str) -> Result<PathwayDb> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut pathways = Vec::new();

    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }

        let parts: Vec<&str> = line.split('\t').collect();
        if parts.len() < 3 {
            continue;
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
