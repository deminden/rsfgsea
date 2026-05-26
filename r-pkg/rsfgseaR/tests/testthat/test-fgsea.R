test_that("fgsea wrapper returns fgsea-style columns", {
  stats <- c(g1 = 2, g2 = 1, g3 = -1, g4 = -2)
  pathways <- list(PW_A = c("g1", "g2"), PW_B = c("g3", "g4"))

  res <- rsfgseaR::fgsea(pathways = pathways, stats = stats, mode = "simple", nPermSimple = 100L)

  expect_s3_class(res, "data.frame")
  expect_true(all(c("pathway", "size", "es", "nes", "pval", "padj", "log2err", "leadingEdge") %in% names(res)))
  expect_equal(nrow(res), 2L)
  expect_type(res$leadingEdge[[1]], "character")
})

test_that("fgsea supports CLI-style file inputs and output", {
  stats_path <- tempfile(fileext = ".rnk")
  gmt_path <- tempfile(fileext = ".gmt")
  out_path <- tempfile(fileext = ".tsv")

  writeLines(c("g1 2", "g2 1", "g3 -1", "g4 -2"), stats_path)
  writeLines(
    c("PW_A\tna\tg1\tg2", "PW_B\tna\tg3\tg4"),
    gmt_path
  )

  res <- rsfgseaR::fgsea(
    pathways = gmt_path,
    stats = stats_path,
    mode = "simple",
    nPermSimple = 100L,
    output = out_path
  )

  expect_equal(nrow(res), 2L)
  expect_true(file.exists(out_path))
  written <- read.delim(out_path, sep = "\t", check.names = FALSE)
  expect_true(all(c("pathway", "size", "es", "nes", "pval", "padj", "log2err", "leading_edge") %in% names(written)))
})

test_that("fgseaSimple and fgseaMultilevel run", {
  stats <- c(g1 = 2, g2 = 1, g3 = -1, g4 = -2)
  pathways <- list(PW_A = c("g1", "g2"), PW_B = c("g3", "g4"))

  simple_res <- rsfgseaR::fgseaSimple(pathways = pathways, stats = stats, nperm = 100L)
  multi_res <- rsfgseaR::fgseaMultilevel(pathways = pathways, stats = stats, nPermSimple = 100L)

  expect_equal(nrow(simple_res), 2L)
  expect_equal(nrow(multi_res), 2L)
})

test_that("fgsea blitz mode runs and rejects incompatible options", {
  stats <- c(g1 = 2, g2 = 1, g3 = -1, g4 = -2)
  pathways <- list(PW_A = c("g1", "g2"), PW_B = c("g3", "g4"))

  blitz_res <- rsfgseaR::fgsea(
    pathways = pathways,
    stats = stats,
    mode = "blitz",
    nPermSimple = 64L,
    minSize = 1L,
    maxSize = 4L,
    blitz.anchors = 4L
  )

  expect_equal(nrow(blitz_res), 2L)
  expect_true(all(is.na(blitz_res$log2err)))
  expect_error(
    rsfgseaR::fgsea(pathways = pathways, stats = stats, mode = "blitz", scoreType = "pos"),
    "mode = 'blitz' supports only scoreType = 'std'"
  )
})

test_that("fgseaSimple respects the active R sample.kind", {
  skip_if_not_installed("fgsea")

  old_kind <- RNGkind()
  on.exit(do.call(RNGkind, as.list(old_kind)), add = TRUE)

  stats <- stats::setNames(seq(30, 1), paste0("g", seq_len(30)))
  pathways <- list(
    PW_A = paste0("g", c(1, 2, 3, 8, 10)),
    PW_B = paste0("g", c(20, 22, 24, 26, 28)),
    PW_C = paste0("g", c(5, 7, 11, 13, 17))
  )

  RNGkind("Mersenne-Twister", "Inversion", "Rounding")
  set.seed(20260322)
  fg_res <- fgsea::fgseaSimple(
    pathways = pathways,
    stats = stats,
    nperm = 1000L,
    minSize = 1L,
    maxSize = length(stats) - 1L,
    nproc = 1L,
    scoreType = "std",
    gseaParam = 1
  )

  rs_res <- rsfgseaR::fgseaSimple(
    pathways = pathways,
    stats = stats,
    nperm = 1000L,
    seed = 20260322L,
    nproc = 1L,
    minSize = 1L,
    maxSize = length(stats) - 1L,
    scoreType = "std",
    gseaParam = 1
  )

  merged <- merge(as.data.frame(fg_res), as.data.frame(rs_res), by = "pathway", suffixes = c("_fg", "_rs"))
  expect_equal(merged$pval_fg, merged$pval_rs, tolerance = 1e-12)
  expect_equal(merged$padj_fg, merged$padj_rs, tolerance = 1e-12)
  expect_equal(merged$NES, merged$nes, tolerance = 1e-12)
})

test_that("fgsea enforces CLI-equivalent option rules", {
  stats <- c(g1 = 2, g2 = 1, g3 = -1, g4 = -2)
  pathways <- list(PW_A = c("g1", "g2"), PW_B = c("g3", "g4"))

  expect_error(
    rsfgseaR::fgsea(pathways = pathways, stats = stats, mode = "multilevel", nperm = 100L),
    "nperm is only valid"
  )
  expect_error(
    rsfgseaR::fgsea(pathways = pathways, stats = stats, nPermSimple = 0L),
    "nPermSimple must be >= 1"
  )
  expect_error(
    rsfgseaR::fgsea(pathways = pathways, stats = stats, nproc = -1L),
    "nproc must be >= 0"
  )
  expect_error(
    rsfgseaR::fgsea(pathways = pathways, stats = stats, scoreType = "bad"),
    "scoreType must be one of"
  )
  expect_error(
    rsfgseaR::fgsea(pathways = pathways, stats = stats, mode = "simple", gpu = TRUE),
    "gpu currently supports only mode = 'fgsea'"
  )
})

test_that("fgsea validates decor arguments", {
  stats <- c(g1 = 2, g2 = 1, g3 = -1, g4 = -2)
  pathways <- list(PW_A = c("g1", "g2"), PW_B = c("g3", "g4"))

  expect_error(
    rsfgseaR::fgsea(pathways = pathways, stats = stats, method = "decor", mode = "simple", nperm = 50L),
    "requires decor.cache"
  )
  expect_error(
    rsfgseaR::fgsea(pathways = pathways, stats = stats, method = "decor", mode = "multilevel", nperm = 50L, decor.cache = tempfile()),
    "nperm is only valid"
  )
  expect_error(
    rsfgseaR::fgsea(pathways = pathways, stats = stats, method = "decor", mode = "simple", nperm = 50L, decor.cache = tempfile(), decor.preset = "bogus"),
    "decor.preset must be one of"
  )
})

test_that("decor balanced preset builds cache and runs", {
  stats <- c(g1 = 2, g2 = 1, g3 = -1, g4 = -2)
  pathways <- list(PW_A = c("g1", "g2"), PW_B = c("g3", "g4"))
  expr_path <- tempfile(fileext = ".tsv")
  cache_path <- tempfile(fileext = ".decor.tsv")
  writeLines(
    c(
      "gene\ts1\ts2\ts3\ts4",
      "g1\t1\t2\t3\t4",
      "g2\t1.1\t2.1\t3.1\t4.1",
      "g3\t4\t3\t2\t1",
      "g4\t2\t1\t2\t1"
    ),
    expr_path
  )

  decor <- rsfgseaR::fgsea(
    pathways = pathways,
    stats = stats,
    method = "decor",
    mode = "simple",
    nperm = 50L,
    seed = 42L,
    decor.cache = cache_path,
    decor.expression = expr_path,
    decor.preset = "balanced"
  )

  expect_true(file.exists(cache_path))
  expect_equal(nrow(decor), 2L)

  decor_multilevel <- rsfgseaR::fgsea(
    pathways = pathways,
    stats = stats,
    method = "decor",
    mode = "multilevel",
    nPermSimple = 25L,
    sampleSize = 11L,
    seed = 42L,
    decor.cache = cache_path,
    decor.expression = expr_path,
    decor.preset = "balanced"
  )

  expect_equal(nrow(decor_multilevel), 2L)
})

test_that("decor accepts named presets", {
  stats <- c(g1 = 2, g2 = 1, g3 = -1, g4 = -2)
  pathways <- list(PW_A = c("g1", "g2"), PW_B = c("g3", "g4"))
  expr_path <- tempfile(fileext = ".tsv")
  writeLines(
    c(
      "gene\ts1\ts2\ts3\ts4",
      "g1\t1\t2\t3\t4",
      "g2\t1.1\t2.1\t3.1\t4.1",
      "g3\t4\t3\t2\t1",
      "g4\t2\t1\t2\t1"
    ),
    expr_path
  )

  for (preset in c("sensitive", "balanced", "specific", "strict")) {
    decor <- rsfgseaR::fgsea(
      pathways = pathways,
      stats = stats,
      method = "decor",
      mode = "simple",
      nperm = 25L,
      seed = 42L,
      decor.cache = tempfile(fileext = ".decor.tsv"),
      decor.expression = expr_path,
      decor.preset = preset
    )
    expect_equal(nrow(decor), 2L)
  }
})

test_that("decor explicit formula matches balanced preset", {
  stats <- c(g1 = 2, g2 = 1, g3 = -1, g4 = -2)
  pathways <- list(PW_A = c("g1", "g2"), PW_B = c("g3", "g4"))
  expr_path <- tempfile(fileext = ".tsv")
  cache_path <- tempfile(fileext = ".decor.tsv")
  writeLines(
    c(
      "gene\ts1\ts2\ts3\ts4",
      "g1\t1\t2\t3\t4",
      "g2\t1.1\t2.1\t3.1\t4.1",
      "g3\t4\t3\t2\t1",
      "g4\t2\t1\t2\t1"
    ),
    expr_path
  )

  preset <- rsfgseaR::fgsea(
    pathways = pathways,
    stats = stats,
    method = "decor",
    mode = "simple",
    nperm = 50L,
    seed = 42L,
    decor.cache = cache_path,
    decor.expression = expr_path,
    decor.preset = "balanced"
  )
  explicit <- rsfgseaR::fgsea(
    pathways = pathways,
    stats = stats,
    method = "decor",
    mode = "simple",
    nperm = 50L,
    seed = 42L,
    decor.cache = cache_path,
    decor.weight.formula = "threshold-rational",
    decor.alpha = 60,
    decor.threshold = 0.04
  )

  expect_equal(preset$pathway, explicit$pathway)
  expect_equal(preset$size, explicit$size)
  expect_equal(preset$es, explicit$es)
  expect_equal(preset$nes, explicit$nes)
  expect_equal(preset$pval, explicit$pval)
  expect_equal(preset$padj, explicit$padj)
})

test_that("decor accepts stringency ladder", {
  stats <- c(g1 = 2, g2 = 1, g3 = -1, g4 = -2)
  pathways <- list(PW_A = c("g1", "g2"), PW_B = c("g3", "g4"))
  expr_path <- tempfile(fileext = ".tsv")
  writeLines(
    c(
      "gene\ts1\ts2\ts3\ts4",
      "g1\t1\t2\t3\t4",
      "g2\t1.1\t2.1\t3.1\t4.1",
      "g3\t4\t3\t2\t1",
      "g4\t2\t1\t2\t1"
    ),
    expr_path
  )

  for (stringency in c(10, 50, 75, 95)) {
    decor <- rsfgseaR::fgsea(
      pathways = pathways,
      stats = stats,
      method = "decor",
      mode = "simple",
      nperm = 25L,
      seed = 42L,
      decor.cache = tempfile(fileext = ".decor.tsv"),
      decor.expression = expr_path,
      decor.stringency = stringency
    )
    expect_equal(nrow(decor), 2L)
  }
})

test_that("decor rejects preset and stringency together", {
  stats <- c(g1 = 2, g2 = 1, g3 = -1, g4 = -2)
  pathways <- list(PW_A = c("g1", "g2"), PW_B = c("g3", "g4"))

  expect_error(
    rsfgseaR::fgsea(
      pathways = pathways,
      stats = stats,
      method = "decor",
      mode = "simple",
      nperm = 25L,
      decor.cache = tempfile(fileext = ".decor.tsv"),
      decor.preset = "specific",
      decor.stringency = 50
    ),
    "decor.preset or decor.stringency"
  )
})

test_that("decor does not expose null selection", {
  stats <- c(g1 = 2, g2 = 1, g3 = -1, g4 = -2)
  pathways <- list(PW_A = c("g1", "g2"), PW_B = c("g3", "g4"))
  expr_path <- tempfile(fileext = ".tsv")
  cache_path <- tempfile(fileext = ".decor.tsv")
  writeLines(
    c(
      "gene\ts1\ts2\ts3\ts4",
      "g1\t1\t2\t3\t4",
      "g2\t1.1\t2.1\t3.1\t4.1",
      "g3\t4\t3\t2\t1",
      "g4\t2\t1\t2\t1"
    ),
    expr_path
  )

  expect_error(rsfgseaR::fgsea(
    pathways = pathways,
    stats = stats,
    method = "decor",
    mode = "simple",
    nperm = 25L,
    seed = 42L,
    decor.cache = cache_path,
    decor.expression = expr_path,
    decor.null = "profile"
  ), "unused argument")
})

test_that("fgsea validates named numeric stats", {
  expect_error(
    rsfgseaR::fgsea(pathways = list(PW = c("g1")), stats = c(1, 2)),
    "named numeric vector"
  )
})

test_that("readRanks parses ranked-list files strictly", {
  ranks_path <- tempfile(fileext = ".rnk")
  writeLines(c("g1 2", "g2 1", "g3 -1"), ranks_path)

  ranks <- rsfgseaR::readRanks(ranks_path)

  expect_equal(unname(ranks), c(2, 1, -1))
  expect_equal(names(ranks), c("g1", "g2", "g3"))
})

test_that("gmtPathways and writeGmtPathways round-trip pathway lists", {
  pathways <- list(PW_A = c("g1", "g2"), PW_B = c("g3", "g4"))
  gmt_path <- tempfile(fileext = ".gmt")

  rsfgseaR::writeGmtPathways(pathways, gmt_path)
  loaded <- rsfgseaR::gmtPathways(gmt_path)

  expect_equal(loaded, pathways)
})

test_that("R-style aliases mirror bridge helpers", {
  expect_equal(rsfgseaR::rsfgseaVersion(), rsfgseaR::rsfgsea_version())
  expect_equal(rsfgseaR::supportedModes(), rsfgseaR::supported_modes())
})

test_that("plotEnrichment writes a PNG file", {
  stats <- c(g1 = 2, g2 = 1, g3 = -1, g4 = -2)
  out_path <- tempfile(fileext = ".png")

  rsfgseaR::plotEnrichment(
    pathway = c("g1", "g2"),
    stats = stats,
    output = out_path,
    pathwayName = "PW_A",
    dpi = 300L
  )

  expect_true(file.exists(out_path))
  expect_gt(file.info(out_path)$size, 0)
})

test_that("plotGseaTable writes a PNG file", {
  stats <- c(g1 = 2, g2 = 1, g3 = -1, g4 = -2)
  pathways <- list(PW_A = c("g1", "g2"), PW_B = c("g3", "g4"))
  fgsea_res <- data.frame(
    pathway = c("PW_A", "PW_B"),
    nes = c(1.5, -1.4),
    pval = c(0.01, 0.02),
    padj = c(0.02, 0.03)
  )
  out_path <- tempfile(fileext = ".png")

  rsfgseaR::plotGseaTable(
    pathways = pathways,
    stats = stats,
    fgseaRes = fgsea_res,
    output = out_path,
    dpi = 300L
  )

  expect_true(file.exists(out_path))
  expect_gt(file.info(out_path)$size, 0)
})
