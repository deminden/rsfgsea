# Runs upstream R fgseaMultilevel over the committed muscle comparison ranked
# lists and writes a reference result table for Rust parity tests.

library(fgsea)
library(data.table)

script_path <- tryCatch(normalizePath(sys.frame(1)$ofile, winslash = "/", mustWork = TRUE),
                        error = function(e) NA_character_)
if (is.na(script_path)) {
  script_path <- normalizePath("scripts/run_fgsea_comparison.R", winslash = "/", mustWork = TRUE)
}
repo_root <- normalizePath(file.path(dirname(script_path), ".."), winslash = "/", mustWork = TRUE)
pathways_path <- file.path(repo_root, "data", "h.all.v2025.1.Hs.symbols.gmt")
pathways <- gmtPathways(pathways_path)

# Directory containing the committed symbol-based .rnk files used for
# R/Rust parity.
rnk_dir <- file.path(repo_root, "crates", "rsfgsea", "tests", "data", "muscle_comparison")
files <- list.files(rnk_dir, pattern = "\\.rnk$", full.names = TRUE)

print(paste("Found", length(files), "rank files"))

results_list <- list()

for (f in files) {
  gene_name <- tools::file_path_sans_ext(basename(f))
  
  # Read ranked list
  ranks <- read.table(f, header = FALSE, col.names = c("Gene", "Score"), sep = "\t")
  if (anyDuplicated(ranks$Gene)) {
    print(paste("Skipping", gene_name, "- duplicate gene IDs in ranked list"))
    next
  }
  stats <- setNames(ranks$Score, ranks$Gene)
  
  # Match the Rust parity test exactly: multilevel mode, min/max size filters,
  # and fixed seed for reproducible p-values.
  set.seed(42)
  tryCatch({
    fgseaRes <- fgseaMultilevel(
      pathways = pathways,
      stats = stats,
      minSize = 15,
      maxSize = 500,
      nPermSimple = 1000,
      eps = 1e-10
    )

    # Save only the columns used by Rust validation to keep the reference
    # simple and deterministic.
    fgseaRes <- fgseaRes[, .(
      pathway = pathway,
      pval = pval,
      padj = padj,
      ES = ES,
      NES = NES,
      size = size
    )]
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

# Save to the committed reference file used by the Rust validation test.
out_path <- file.path(rnk_dir, "r_fgsea_results.csv")
write.csv(all_results, out_path, row.names = FALSE)
print(paste("Saved R results to", out_path))
