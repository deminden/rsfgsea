
args <- commandArgs(trailingOnly = TRUE)
rnk_path <- args[[1]]
gmt_path <- args[[2]]
seed <- as.integer(args[[3]])
out_path <- args[[4]]
min_size <- as.integer(args[[5]])
max_size <- as.integer(args[[6]])
nperm <- as.integer(args[[7]])

suppressPackageStartupMessages(library(fgsea))

tbl <- read.table(rnk_path, header = FALSE, sep = "	", stringsAsFactors = FALSE, quote = "", comment.char = "")
colnames(tbl) <- c("gene", "score")
st <- tbl$score
names(st) <- tbl$gene
st <- sort(st, decreasing = TRUE)
pathways <- gmtPathways(gmt_path)

set.seed(seed)
res <- suppressWarnings(fgseaSimple(
  pathways = pathways,
  stats = st,
  nperm = nperm,
  minSize = min_size,
  maxSize = max_size,
  scoreType = "std",
  gseaParam = 1.0,
  nproc = 1
))

if ("leadingEdge" %in% colnames(res)) {
  res$leadingEdge <- vapply(res$leadingEdge, function(x) paste(x, collapse = ","), character(1))
}
res <- as.data.frame(res)
write.table(res, file = out_path, sep = "	", quote = FALSE, row.names = FALSE)
