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
