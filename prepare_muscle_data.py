import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import random
import os
import json
import warnings

# Suppress constant input warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

print("Loading gene mapping...")
with open('data/gene_to_id_mapping_full.json', 'r') as f:
    ensembl_to_symbol = json.load(f)

print("Loading Muscle expression data...")
muscle_file = "/home/den/bio/min-rust-corr/data/Muscle - Skeletal_normalized_counts.tsv.gz"
df = pd.read_csv(muscle_file, sep='\t', index_col=0, compression='gzip')

print(f"Loaded expression data: {df.shape[0]} genes x {df.shape[1]} samples")

# Filter out genes with zero variance (constant expression)
print("Filtering genes with variance...")
gene_vars = df.var(axis=1)
df_filtered = df[gene_vars > 0]
print(f"After filtering: {df_filtered.shape[0]} genes with non-zero variance")

# Select 50 random genes from filtered set
all_genes = df_filtered.index.tolist()
selected_genes = random.sample(all_genes, min(50, len(all_genes)))

print(f"Selected {len(selected_genes)} random genes for analysis")

n_samples = df_filtered.shape[1]

def r_to_t(r, n):
    """Convert correlation to t-statistic"""
    r_capped = np.clip(r, -0.999, 0.999)
    return r_capped * np.sqrt(n - 2) / np.sqrt(1 - r_capped**2)

results_dir = 'tests/data/muscle_comparison'
os.makedirs(results_dir, exist_ok=True)

for idx, gene_id in enumerate(selected_genes, 1):
    print(f"[{idx}/{len(selected_genes)}] Processing {gene_id}...")
    
    # Get expression values for this gene
    gene_expr = df_filtered.loc[gene_id].values
    
    # Vectorized correlation calculation
    # This is much faster than looping
    correlations = []
    for other_gene_id in df_filtered.index:
        other_expr = df_filtered.loc[other_gene_id].values
        try:
            corr, _ = spearmanr(gene_expr, other_expr)
            if np.isfinite(corr):
                correlations.append((other_gene_id, corr))
        except:
            pass  # Skip genes that cause errors
    
    # Sort by absolute correlation and take top 500
    correlations_df = pd.DataFrame(correlations, columns=['GeneID', 'Correlation'])
    correlations_df['AbsCorr'] = correlations_df['Correlation'].abs()
    top_500 = correlations_df.nlargest(min(500, len(correlations_df)), 'AbsCorr')
    
    # Convert to t-statistics
    top_500['t_stat'] = top_500['Correlation'].apply(lambda r: r_to_t(r, n_samples))
    
    # Convert Ensembl IDs to symbols
    top_500['Symbol'] = top_500['GeneID'].map(lambda x: ensembl_to_symbol.get(x, x))
    
    # Remove any infinite values
    top_500 = top_500[np.isfinite(top_500['t_stat'])]
    
    # Save to .rnk file with symbols
    gene_symbol = ensembl_to_symbol.get(gene_id, gene_id)
    rnk_filename = os.path.join(results_dir, f"{gene_symbol}.rnk")
    
    # Write ranked list (Symbol, t-statistic)
    with open(rnk_filename, 'w') as f:
        for _, row in top_500.iterrows():
            f.write(f"{row['Symbol']}\t{row['t_stat']}\n")
    
    print(f"  Saved {len(top_500)} genes")

print(f"\nCompleted! Saved {len(selected_genes)} ranked lists to {results_dir}")
