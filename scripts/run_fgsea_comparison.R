
library(fgsea)
library(data.table)

# Load pathways
pathways <- gmtPathways('data/h.all.v2025.1.Hs.symbols.gmt')

# Directory containing .rnk files
rnk_dir <- 'tests/data/muscle_comparison'
files <- list.files(rnk_dir, pattern = "\\.rnk$", full.names = TRUE)

print(paste("Found", length(files), "rank files"))

results_list <- list()

for (f in files) {
  gene_name <- tools::file_path_sans_ext(basename(f))
  
  # Read ranked list
  ranks <- read.table(f, header = FALSE, col.names = c("Gene", "Score"), sep = "\t")
  stats <- setNames(ranks$Score, ranks$Gene)
  
  # Run FGSEA with relaxed parameters (we only have 500 genes per list)
  set.seed(42)
  tryCatch({
    fgseaRes <- fgsea(pathways = pathways, 
                      stats = stats,
                      minSize=1,      # Relaxed from 15
                      maxSize=5000)   # Increased from 500
    
    # Convert leadingEdge list to string
    fgseaRes$leadingEdge <- sapply(fgseaRes$leadingEdge, function(x) paste(x, collapse="|"))
    
    # Add gene name column
    fgseaRes$TargetGene <- gene_name
    
    results_list[[gene_name]] <- fgseaRes
    print(paste("Processed", gene_name, "- found", nrow(fgseaRes), "pathways"))
  }, error = function(e) {
    print(paste("ERROR processing", gene_name, ":", e$message))
  })
}

# Combine all results
all_results <- rbindlist(results_list)

print(paste("Total results:", nrow(all_results)))

# Save to CSV
write.csv(all_results, 'tests/data/muscle_comparison/r_fgsea_results.csv', row.names = FALSE)
print(paste("Saved R results to", 'tests/data/muscle_comparison/r_fgsea_results.csv'))
