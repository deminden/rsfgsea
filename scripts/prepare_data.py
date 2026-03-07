# Builds sample ranked-list `.rnk` files from precomputed bladder correlation data for local fgsea comparisons.
import pandas as pd
import numpy as np
import random
import os
import json

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Load Ensembl to Symbol mapping
with open('data/ensembl_to_symbol.json', 'r') as f:
    ensembl_to_symbol = json.load(f)

# Load Spearman correlations
print("Loading Spearman correlations...")
df = pd.read_csv('data/bladder_small_590genes.tsv_spearman_correlations.tsv', sep='\t', index_col=0)

# Select 50 random genes (columns)
all_genes = df.columns.tolist()
if len(all_genes) < 50:
    selected_genes = all_genes
else:
    selected_genes = random.sample(all_genes, 50)

print(f"Selected {len(selected_genes)} genes.")

# Constants
n_samples = 590 

def r_to_t(r, n):
    # Cap r to avoid infinity
    r_capped = r
    if r > 0.999:
        r_capped = 0.999
    elif r < -0.999:
        r_capped = -0.999
        
    return r_capped * np.sqrt(n - 2) / np.sqrt(1 - r_capped**2)

results_dir = 'tests/data/muscle_comparison'
os.makedirs(results_dir, exist_ok=True)

for gene in selected_genes:
    # Get correlations for this gene
    corrs = df[gene].copy()
    
    # Take top 500 by absolute correlation
    corrs_abs = corrs.abs().sort_values(ascending=False)
    top_500_genes = corrs_abs.head(500).index
    
    # Get the actual signed correlations for these 500
    top_500_corrs = corrs.loc[top_500_genes]
    
    # Convert to t-statistics
    t_stats = top_500_corrs.apply(lambda r: r_to_t(r, n_samples))
    
    # Convert Ensembl IDs to symbols
    t_stats_symbols = {}
    for ensembl_id, t_val in t_stats.items():
        symbol = ensembl_to_symbol.get(ensembl_id, ensembl_id)  # Use Ensembl ID if no symbol found
        t_stats_symbols[symbol] = t_val
    
    # Create series with symbols
    t_stats_final = pd.Series(t_stats_symbols)
    
    # Remove any infinite values
    t_stats_final = t_stats_final.replace([np.inf, -np.inf], np.nan).dropna()
    
    # Save to .rnk file
    # Use the symbol for the gene name if available
    gene_symbol = ensembl_to_symbol.get(gene, gene)
    rnk_filename = os.path.join(results_dir, f"{gene_symbol}.rnk")
    t_stats_final.to_csv(rnk_filename, sep='\t', header=False)

print(f"Saved {len(selected_genes)} ranked lists to {results_dir}")
