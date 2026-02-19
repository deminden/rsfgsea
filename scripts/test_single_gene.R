library(fgsea)
library(data.table)

# Load pathways
pathways <- gmtPathways('data/h.all.v2025.1.Hs.symbols.gmt')

# Test with one file
ranks <- read.table('tests/data/muscle_comparison/AKAP3.rnk', header = FALSE, col.names = c("Gene", "Score"), sep = "\t")
stats <- setNames(ranks$Score, ranks$Gene)

print(paste("Loaded", length(stats), "genes"))
print(paste("Loaded", length(pathways), "pathways"))

# Check overlap
all_pathway_genes <- unique(unlist(pathways))
overlap <- sum(names(stats) %in% all_pathway_genes)
print(paste("Overlap:", overlap, "genes"))

# Try with relaxed parameters
set.seed(42)
fgseaRes <- fgsea(pathways = pathways, 
                  stats = stats,
                  minSize=1,
                  maxSize=5000)

print(paste("Found", nrow(fgseaRes), "pathway results"))
print(head(fgseaRes))
