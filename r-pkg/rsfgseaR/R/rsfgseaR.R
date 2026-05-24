#' rsfgseaR
#'
#' R interface for the `rsfgsea` Rust backend.
#' @name rsfgseaR-package
NULL

.write_pathways_gmt <- function(pathways, path) {
  stopifnot(is.list(pathways), !is.null(names(pathways)))
  lines <- vapply(
    names(pathways),
    function(name) {
      genes <- pathways[[name]]
      if (!is.character(genes)) {
        stop("Each pathways entry must be a character vector of genes.", call. = FALSE)
      }
      paste(c(name, "na", genes), collapse = "\t")
    },
    character(1)
  )
  writeLines(lines, path, useBytes = TRUE)
}

.validate_stats <- function(stats) {
  if (!is.numeric(stats) || is.null(names(stats)) || any(names(stats) == "")) {
    stop("stats must be a named numeric vector.", call. = FALSE)
  }
  if (anyDuplicated(names(stats))) {
    stop("stats names must be unique.", call. = FALSE)
  }
  if (any(!is.finite(stats))) {
    stop("stats values must be finite.", call. = FALSE)
  }
}

.validate_integerish_scalar <- function(value, name, min_value = NULL, allow_null = FALSE) {
  if (is.null(value)) {
    if (allow_null) {
      return(invisible(NULL))
    }
    stop(name, " must not be NULL.", call. = FALSE)
  }

  if (length(value) != 1L || !is.numeric(value) || !is.finite(value) || value != as.integer(value)) {
    stop(name, " must be a single integer value.", call. = FALSE)
  }

  if (!is.null(min_value) && value < min_value) {
    stop(name, " must be >= ", min_value, ".", call. = FALSE)
  }
}

.validate_positive_scalar <- function(value, name) {
  if (length(value) != 1L || !is.numeric(value) || !is.finite(value) || value <= 0) {
    stop(name, " must be a single finite numeric value > 0.", call. = FALSE)
  }
}

.validate_choice <- function(value, name, choices) {
  if (!is.character(value) || length(value) != 1L || !(tolower(value) %in% choices)) {
    stop(name, " must be one of: ", paste(choices, collapse = ", "), ".", call. = FALSE)
  }
}

.resolve_sample_kind <- function() {
  kinds <- RNGkind()
  if (length(kinds) >= 3L && is.character(kinds[[3L]]) && nzchar(kinds[[3L]])) {
    return(kinds[[3L]])
  }
  "Rounding"
}

.resolve_decor_preset <- function(preset) {
  switch(tolower(preset),
    sensitive = list(
      alpha = 22.0,
      weight.formula = "raw-rational",
      threshold = 0.0,
      gamma = 1.0,
      penalty.floor = 0.0
    ),
    balanced = list(
      alpha = 60.0,
      weight.formula = "threshold-rational",
      threshold = 0.04,
      gamma = 1.0,
      penalty.floor = 0.0
    ),
    specific = list(
      alpha = 65.0,
      weight.formula = "threshold-rational",
      threshold = 0.05,
      gamma = 1.0,
      penalty.floor = 0.0
    ),
    strict = list(
      alpha = -log(0.10),
      weight.formula = "exp-scaled",
      threshold = 0.0,
      gamma = 1.0,
      penalty.floor = 0.0
    )
  )
}

.resolve_decor_stringency <- function(stringency) {
  if (length(stringency) != 1L || !is.numeric(stringency) || !is.finite(stringency) || stringency < 0 || stringency > 100) {
    stop("decor.stringency must be a single finite numeric value from 0 to 100.", call. = FALSE)
  }

  preset <- if (stringency < 35) {
    "sensitive"
  } else if (stringency < 65) {
    "balanced"
  } else if (stringency < 85) {
    "specific"
  } else {
    "strict"
  }
  resolved <- .resolve_decor_preset(preset)
  resolved$preset <- preset
  resolved$stringency <- stringency
  resolved
}

.normalize_stats <- function(stats) {
  if (is.character(stats) && length(stats) == 1L) {
    if (!file.exists(stats)) {
      stop("Ranks file does not exist: ", stats, call. = FALSE)
    }
    return(readRanks(stats))
  }

  .validate_stats(stats)
  stats
}

.normalize_pathways <- function(pathways) {
  if (is.character(pathways) && length(pathways) == 1L) {
    if (!file.exists(pathways)) {
      stop("GMT file does not exist: ", pathways, call. = FALSE)
    }
    return(list(path = pathways, cleanup = FALSE))
  }

  if (!is.list(pathways) || is.null(names(pathways)) || any(names(pathways) == "")) {
    stop("pathways must be a GMT path or a named list of character vectors.", call. = FALSE)
  }

  path <- tempfile(fileext = ".gmt")
  .write_pathways_gmt(pathways, path)
  list(path = path, cleanup = TRUE)
}

.write_results_tsv <- function(result, path) {
  export <- result
  export$leading_edge <- vapply(
    export$leadingEdge,
    function(x) paste(x, collapse = ","),
    character(1)
  )
  export$leadingEdge <- NULL
  utils::write.table(
    export,
    file = path,
    sep = "\t",
    row.names = FALSE,
    quote = FALSE
  )
}

.as_fgsea_df <- function(result) {
  leading_edge <- strsplit(result$leadingEdge, ",", fixed = TRUE)
  result$leadingEdge <- I(lapply(leading_edge, function(x) {
    if (length(x) == 1L && identical(x[[1]], "")) character() else x
  }))
  as.data.frame(result, stringsAsFactors = FALSE)
}

#' Read a GMT file into a named pathway list
#'
#' @param path Path to a GMT file.
#'
#' @return A named list of character vectors.
#' @export
gmtPathways <- function(path) {
  if (!is.character(path) || length(path) != 1L || !file.exists(path)) {
    stop("path must point to an existing GMT file.", call. = FALSE)
  }

  lines <- readLines(path, warn = FALSE)
  parsed <- lapply(seq_along(lines), function(i) {
    line <- trimws(lines[[i]])
    if (identical(line, "")) {
      return(NULL)
    }

    fields <- strsplit(line, "\t", fixed = TRUE)[[1]]
    if (length(fields) < 3L) {
      stop("Malformed GMT line ", i, ": expected pathway, description, and at least one gene.", call. = FALSE)
    }

    genes <- fields[-c(1L, 2L)]
    if (any(genes == "")) {
      stop("Malformed GMT line ", i, ": gene names must be non-empty.", call. = FALSE)
    }

    list(name = fields[[1]], genes = genes)
  })

  parsed <- Filter(Negate(is.null), parsed)
  names_vec <- vapply(parsed, `[[`, character(1), "name")
  if (anyDuplicated(names_vec)) {
    stop("GMT pathway names must be unique.", call. = FALSE)
  }

  stats::setNames(lapply(parsed, `[[`, "genes"), names_vec)
}

#' Write a named pathway list to GMT
#'
#' @param pathways Named list of character vectors.
#' @param path Output GMT file path.
#'
#' @return Invisibly returns `path`.
#' @export
writeGmtPathways <- function(pathways, path) {
  if (!is.character(path) || length(path) != 1L) {
    stop("path must be a single output file path.", call. = FALSE)
  }
  .normalize_pathways(pathways)
  .write_pathways_gmt(pathways, path)
  invisible(path)
}

#' Read a ranked list file
#'
#' Reads a whitespace- or tab-separated two-column ranked list file with gene
#' names in the first column and numeric statistics in the second column.
#'
#' @param path Path to the ranked list file.
#'
#' @return A named numeric vector.
#' @export
readRanks <- function(path) {
  if (!is.character(path) || length(path) != 1L || !file.exists(path)) {
    stop("path must point to an existing ranked list file.", call. = FALSE)
  }

  lines <- readLines(path, warn = FALSE)
  parsed <- lapply(seq_along(lines), function(i) {
    line <- trimws(lines[[i]])
    if (identical(line, "")) {
      return(NULL)
    }

    fields <- strsplit(line, "[[:space:]]+", perl = TRUE)[[1]]
    if (length(fields) < 2L) {
      stop("Malformed ranked-list line ", i, ": expected at least 2 whitespace-separated columns.", call. = FALSE)
    }

    score <- suppressWarnings(as.numeric(fields[[2]]))
    if (!is.finite(score)) {
      stop("Malformed ranked-list line ", i, ": invalid numeric score '", fields[[2]], "'.", call. = FALSE)
    }

    list(gene = fields[[1]], score = score)
  })

  parsed <- Filter(Negate(is.null), parsed)
  genes <- vapply(parsed, `[[`, character(1), "gene")
  if (anyDuplicated(genes)) {
    stop("Ranked list gene names must be unique.", call. = FALSE)
  }

  scores <- vapply(parsed, `[[`, numeric(1), "score")
  stats <- stats::setNames(scores, genes)
  .validate_stats(stats)
  stats
}

#' Return the rsfgsea backend version
#'
#' @return A version string.
#' @export
rsfgseaVersion <- function() {
  rsfgsea_version()
}

#' Return supported execution modes
#'
#' @return A character vector of supported modes.
#' @export
supportedModes <- function() {
  supported_modes()
}

#' Write a single-pathway enrichment plot as PNG
#'
#' @param pathway Character vector of genes in the pathway.
#' @param stats Named numeric vector of preranked statistics, or a path to a ranked-list file.
#' @param output Output PNG file path.
#' @param pathwayName Pathway label used internally for the Rust plotting call.
#' @param scoreType One of `"std"`, `"pos"`, `"neg"`.
#' @param gseaParam Weighting exponent.
#' @param width_inches Output width in inches.
#' @param height_inches Output height in inches.
#' @param dpi Output DPI metadata for the PNG file.
#' @param transparent_background Logical flag; if `TRUE`, write the PNG with a transparent background.
#' @param title Optional plot title. If `NULL`, no title is drawn.
#'
#' @return Invisibly returns `output`.
#' @export
plotEnrichment <- function(
  pathway,
  stats,
  output,
  pathwayName = "pathway",
  scoreType = "std",
  gseaParam = 1.0,
  width_inches = 3.0,
  height_inches = 2.2,
  dpi = 300L,
  transparent_background = FALSE,
  title = NULL
) {
  stats <- .normalize_stats(stats)
  if (!is.character(pathway) || length(pathway) == 0L || anyNA(pathway) || any(pathway == "")) {
    stop("pathway must be a non-empty character vector of genes.", call. = FALSE)
  }
  if (!is.character(output) || length(output) != 1L || identical(output, "")) {
    stop("output must be a single file path.", call. = FALSE)
  }
  if (!is.character(pathwayName) || length(pathwayName) != 1L) {
    stop("pathwayName must be a single string.", call. = FALSE)
  }
  .validate_choice(scoreType, "scoreType", c("std", "pos", "neg"))
  .validate_positive_scalar(width_inches, "width_inches")
  .validate_positive_scalar(height_inches, "height_inches")
  .validate_integerish_scalar(dpi, "dpi", min_value = 1L)
  if (!is.logical(transparent_background) || length(transparent_background) != 1L || is.na(transparent_background)) {
    stop("transparent_background must be TRUE or FALSE.", call. = FALSE)
  }
  if (!is.null(title) && (!is.character(title) || length(title) != 1L)) {
    stop("title must be NULL or a single string.", call. = FALSE)
  }
  if (!is.numeric(gseaParam) || length(gseaParam) != 1L || !is.finite(gseaParam) || gseaParam < 0) {
    stop("gseaParam must be a single finite numeric value >= 0.", call. = FALSE)
  }

  write_enrichment_plot(
    unname(as.numeric(stats)),
    names(stats),
    pathway,
    output,
    pathwayName,
    scoreType,
    gseaParam,
    as.numeric(width_inches),
    as.numeric(height_inches),
    as.integer(dpi),
    isTRUE(transparent_background),
    if (is.null(title)) "" else title
  )

  invisible(output)
}

#' Write a multi-pathway GSEA table plot as PNG
#'
#' @param pathways Named list of character vectors of pathway genes.
#' @param stats Named numeric vector of preranked statistics, or a path to a ranked-list file.
#' @param fgseaRes Data frame with at least `pathway`, `nes`, `pval`, and `padj` columns.
#' @param output Output PNG file path.
#' @param gseaParam Weighting exponent.
#' @param width_inches Output width in inches.
#' @param height_inches Optional output height in inches. `NULL` derives height from row count.
#' @param dpi Output DPI metadata for the PNG file.
#' @param transparent_background Logical flag; if `TRUE`, write the PNG with a transparent background.
#'
#' @return Invisibly returns `output`.
#' @export
plotGseaTable <- function(
  pathways,
  stats,
  fgseaRes,
  output,
  gseaParam = 1.0,
  width_inches = 5.6,
  height_inches = NULL,
  dpi = 300L,
  transparent_background = FALSE
) {
  stats <- .normalize_stats(stats)
  if (!is.list(pathways) || length(pathways) == 0L || is.null(names(pathways)) || any(names(pathways) == "")) {
    stop("pathways must be a non-empty named list of character vectors.", call. = FALSE)
  }
  for (i in seq_along(pathways)) {
    genes <- pathways[[i]]
    if (!is.character(genes) || length(genes) == 0L || anyNA(genes) || any(genes == "")) {
      stop(sprintf("pathways[['%s']] must be a non-empty character vector.", names(pathways)[[i]]), call. = FALSE)
    }
  }
  if (!is.data.frame(fgseaRes)) {
    stop("fgseaRes must be a data frame.", call. = FALSE)
  }
  required_cols <- c("pathway", "nes", "pval", "padj")
  missing_cols <- setdiff(required_cols, colnames(fgseaRes))
  if (length(missing_cols) > 0L) {
    stop(sprintf("fgseaRes is missing required columns: %s", paste(missing_cols, collapse = ", ")), call. = FALSE)
  }
  if (!is.character(output) || length(output) != 1L || identical(output, "")) {
    stop("output must be a single file path.", call. = FALSE)
  }
  .validate_positive_scalar(width_inches, "width_inches")
  if (!is.null(height_inches)) {
    .validate_positive_scalar(height_inches, "height_inches")
  }
  .validate_integerish_scalar(dpi, "dpi", min_value = 1L)
  if (!is.logical(transparent_background) || length(transparent_background) != 1L || is.na(transparent_background)) {
    stop("transparent_background must be TRUE or FALSE.", call. = FALSE)
  }
  if (!is.numeric(gseaParam) || length(gseaParam) != 1L || !is.finite(gseaParam) || gseaParam < 0) {
    stop("gseaParam must be a single finite numeric value >= 0.", call. = FALSE)
  }

  pathway_names <- names(pathways)
  write_gsea_table_plot(
    unname(as.numeric(stats)),
    names(stats),
    pathway_names,
    unname(pathways),
    as.character(fgseaRes$pathway),
    as.numeric(fgseaRes$nes),
    as.numeric(fgseaRes$pval),
    as.numeric(fgseaRes$padj),
    output,
    gseaParam,
    as.numeric(width_inches),
    if (is.null(height_inches)) NULL else as.numeric(height_inches),
    as.integer(dpi),
    isTRUE(transparent_background)
  )

  invisible(output)
}

#' Wrapper-style fgsea interface
#'
#' Closest to the standard fgsea-style interface. Uses simple screening first and
#' multilevel refinement unless `nperm` is set.
#'
#' Run fgsea-compatible preranked enrichment
#'
#' @param pathways Either a named list of character vectors or a path to a GMT file.
#' @param stats Named numeric vector of preranked statistics, or a path to a ranked-list file.
#' @param nPermSimple Integer permutation count for the simple screening stage.
#'   For CPU/GPU or R/GPU comparisons, prefer `100000L` as a practical baseline;
#'   use `10000L` only as a smoke tier and `1000000L` for final tail/stress
#'   checks when runtime allows.
#' @param seed Optional integer RNG seed. `NULL` uses a fresh random seed.
#' @param nproc Number of worker threads. `0` keeps the default Rayon behavior.
#' @param minSize Minimum pathway size.
#' @param maxSize Maximum pathway size. Defaults to `length(stats) - 1`.
#' @param eps Multilevel epsilon parameter.
#' @param scoreType One of `"std"`, `"pos"`, `"neg"`.
#' @param gseaParam Weighting exponent.
#' @param mode One of `"fgsea"`, `"simple"`, `"multilevel"`, `"blitz"`.
#' @param nperm Optional fixed-permutation override for wrapper mode.
#' @param sampleSize Multilevel sample size.
#' @param method One of `"classic"` or `"decor"`. The default preserves the
#'   fgsea-compatible classic method.
#' @param decor.cache Path to a decor redundancy cache.
#' @param decor.expression Optional normalized expression matrix used to build
#'   or rebuild the decor cache.
#' @param decor.preset Decor redundancy preset. `"balanced"` is the default
#'   held-out-validated threshold preset. Other choices are `"sensitive"`,
#'   `"specific"`, and `"strict"`.
#' @param decor.stringency Optional numeric 0-100 convenience control. When set,
#'   it autoswitches between the calibrated decor presets instead of exposing
#'   formula-level controls.
#' @param decor.cache.mode One of `"auto"`, `"reuse"`, `"rebuild"`.
#' @param decor.correlation Correlation method for decor cache building. Only
#'   `"pearson"` is currently implemented.
#' @param decor.redundancy Redundancy score definition, `"positive_mean"` or
#'   `"abs_mean"`.
#' @param output Optional TSV output path. When set, results are also written in
#'   the same column shape as the CLI.
#' @param gpu Logical flag mirroring the CLI `--gpu` switch. Uses the same
#'   hybrid GPU path as the Rust CLI and currently supports only `mode = "fgsea"`.
#'   On WSL2, if CUDA is visible but WebGPU selects `llvmpipe`, start R with
#'   `GALLIUM_DRIVER=d3d12` and `MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA`.
#' @param blitz.anchors Number of blitz calibration anchors.
#' @param blitz.symmetric Use one symmetric positive/negative blitz null fit.
#' @param blitz.center Center signature values before blitz scoring.
#' @param blitz.accuracy Blitz normal-tail accuracy setting for parity metadata.
#' @param blitz.deep.accuracy Blitz deep-tail accuracy setting for parity metadata.
#'
#' @return A data frame with fgsea-style result columns.
#' @export
fgsea <- function(
  pathways,
  stats,
  nPermSimple = 1000L,
  seed = NULL,
  nproc = 0L,
  minSize = NULL,
  maxSize = NULL,
  eps = 1e-50,
  scoreType = "std",
  gseaParam = 1.0,
  mode = "fgsea",
  nperm = NULL,
  sampleSize = 101L,
  method = "classic",
  decor.cache = NULL,
  decor.expression = NULL,
  decor.preset = "balanced",
  decor.stringency = NULL,
  decor.cache.mode = "auto",
  decor.correlation = "pearson",
  decor.redundancy = "positive_mean",
  output = NULL,
  gpu = FALSE,
  blitz.anchors = 40L,
  blitz.symmetric = FALSE,
  blitz.center = TRUE,
  blitz.accuracy = 40L,
  blitz.deep.accuracy = 50L
) {
  stats <- .normalize_stats(stats)
  .validate_integerish_scalar(nPermSimple, "nPermSimple", min_value = 1L)
  if (!is.null(seed)) {
    .validate_integerish_scalar(seed, "seed", min_value = 0L)
  }
  .validate_integerish_scalar(nproc, "nproc", min_value = 0L)
  .validate_integerish_scalar(minSize, "minSize", min_value = 1L, allow_null = TRUE)
  .validate_integerish_scalar(sampleSize, "sampleSize", min_value = 1L)
  .validate_integerish_scalar(blitz.anchors, "blitz.anchors", min_value = 1L)
  .validate_integerish_scalar(blitz.accuracy, "blitz.accuracy", min_value = 1L)
  .validate_integerish_scalar(blitz.deep.accuracy, "blitz.deep.accuracy", min_value = 1L)
  if (!is.null(maxSize)) {
    .validate_integerish_scalar(maxSize, "maxSize", min_value = 1L)
  }
  if (!is.null(nperm)) {
    .validate_integerish_scalar(nperm, "nperm", min_value = 1L)
  }
  if (!is.numeric(eps) || length(eps) != 1L || !is.finite(eps)) {
    stop("eps must be a single finite numeric value.", call. = FALSE)
  }
  if (!is.numeric(gseaParam) || length(gseaParam) != 1L || !is.finite(gseaParam)) {
    stop("gseaParam must be a single finite numeric value.", call. = FALSE)
  }
  .validate_choice(mode, "mode", c("fgsea", "simple", "multilevel", "blitz"))
  .validate_choice(scoreType, "scoreType", c("std", "pos", "neg"))
  .validate_choice(method, "method", c("classic", "decor"))
  .validate_choice(decor.preset, "decor.preset", c("sensitive", "balanced", "specific", "strict"))
  if (!is.null(decor.stringency) && tolower(decor.preset) != "balanced") {
    stop("Use either decor.preset or decor.stringency, not both.", call. = FALSE)
  }
  .validate_choice(decor.cache.mode, "decor.cache.mode", c("auto", "reuse", "rebuild"))
  .validate_choice(decor.correlation, "decor.correlation", c("pearson", "spearman"))
  .validate_choice(decor.redundancy, "decor.redundancy", c("positive_mean", "abs_mean"))
  if (!is.null(decor.cache) && (!is.character(decor.cache) || length(decor.cache) != 1L || identical(decor.cache, ""))) {
    stop("decor.cache must be NULL or a single file path.", call. = FALSE)
  }
  if (!is.null(decor.expression) && (!is.character(decor.expression) || length(decor.expression) != 1L || identical(decor.expression, ""))) {
    stop("decor.expression must be NULL or a single file path.", call. = FALSE)
  }
  if (!is.logical(gpu) || length(gpu) != 1L || is.na(gpu)) {
    stop("gpu must be TRUE or FALSE.", call. = FALSE)
  }
  if (!is.logical(blitz.symmetric) || length(blitz.symmetric) != 1L || is.na(blitz.symmetric)) {
    stop("blitz.symmetric must be TRUE or FALSE.", call. = FALSE)
  }
  if (!is.logical(blitz.center) || length(blitz.center) != 1L || is.na(blitz.center)) {
    stop("blitz.center must be TRUE or FALSE.", call. = FALSE)
  }
  if (!is.null(output) && (!is.character(output) || length(output) != 1L)) {
    stop("output must be NULL or a single file path.", call. = FALSE)
  }
  if (tolower(mode) == "multilevel" && !is.null(nperm)) {
    stop("nperm is only valid with mode = 'fgsea' or mode = 'simple'.", call. = FALSE)
  }
  if (gpu && tolower(mode) != "fgsea") {
    stop("gpu currently supports only mode = 'fgsea'.", call. = FALSE)
  }
  if (tolower(mode) == "blitz") {
    if (gpu) {
      stop("gpu is not supported with mode = 'blitz'.", call. = FALSE)
    }
    if (tolower(method) != "classic") {
      stop("mode = 'blitz' supports only method = 'classic'.", call. = FALSE)
    }
    if (!is.null(nperm)) {
      stop("nperm is not supported with mode = 'blitz'.", call. = FALSE)
    }
    if (tolower(scoreType) != "std") {
      stop("mode = 'blitz' supports only scoreType = 'std'.", call. = FALSE)
    }
    if (!identical(as.numeric(gseaParam), 1.0)) {
      stop("mode = 'blitz' supports only gseaParam = 1.", call. = FALSE)
    }
  }
  if (tolower(method) == "decor") {
    if (is.null(decor.cache)) {
      stop("method = 'decor' requires decor.cache.", call. = FALSE)
    }
    if (tolower(decor.correlation) == "spearman") {
      stop("spearman decor correlation is not implemented yet.", call. = FALSE)
    }
    if (gpu || tolower(mode) == "multilevel" || (tolower(mode) == "fgsea" && is.null(nperm))) {
      stop("decor supports CPU fixed-permutation simple runs; use mode = 'simple' or provide nperm without gpu.", call. = FALSE)
    }
  } else if (!is.null(decor.cache) || !is.null(decor.expression) || !is.null(decor.stringency) || tolower(decor.preset) != "balanced") {
    stop("decor arguments require method = 'decor'.", call. = FALSE)
  }
  decor.resolved <- if (is.null(decor.stringency)) {
    .resolve_decor_preset(decor.preset)
  } else {
    .resolve_decor_stringency(decor.stringency)
  }

  pathways_info <- .normalize_pathways(pathways)
  if (pathways_info$cleanup) {
    on.exit(unlink(pathways_info$path), add = TRUE)
  }

  result <- fgsea_rust(
    unname(as.numeric(stats)),
    names(stats),
    pathways_info$path,
    as.integer(nPermSimple),
    if (is.null(seed)) NULL else as.integer(seed),
    as.integer(nproc),
    if (is.null(minSize)) -1L else as.integer(minSize),
    if (is.null(maxSize)) -1L else as.integer(maxSize),
    eps,
    scoreType,
    gseaParam,
    mode,
    if (is.null(nperm)) -1L else as.integer(nperm),
    as.integer(sampleSize),
    .resolve_sample_kind(),
    gpu,
    method,
    if (is.null(decor.cache)) NULL else decor.cache,
    if (is.null(decor.expression)) NULL else decor.expression,
    as.numeric(decor.resolved$alpha),
    decor.cache.mode,
    decor.correlation,
    decor.redundancy,
    decor.resolved$weight.formula,
    as.numeric(decor.resolved$threshold),
    as.numeric(decor.resolved$gamma),
    as.numeric(decor.resolved$penalty.floor),
    1e-12,
    as.integer(blitz.anchors),
    isTRUE(blitz.symmetric),
    isTRUE(blitz.center),
    as.integer(blitz.accuracy),
    as.integer(blitz.deep.accuracy)
  )

  result_df <- .as_fgsea_df(result)
  if (!is.null(output)) {
    .write_results_tsv(result_df, output)
  }
  result_df
}

#' Fixed-permutation simple fgsea interface
#'
#' @param pathways Either a named list of character vectors or a path to a GMT file.
#' @param stats Named numeric vector of preranked statistics, or a path to a ranked-list file.
#' @param nperm Number of permutations. For CPU/GPU or R/GPU comparisons,
#'   prefer `100000L` as a practical baseline; use `10000L` only as a smoke tier.
#' @param seed Optional integer RNG seed. `NULL` uses a fresh random seed.
#' @param nproc Number of worker threads. `0` keeps the default Rayon behavior.
#' @param minSize Minimum pathway size.
#' @param maxSize Maximum pathway size. Defaults to `length(stats) - 1`.
#' @param eps Multilevel epsilon parameter, kept for interface parity.
#' @param scoreType One of `"std"`, `"pos"`, `"neg"`.
#' @param gseaParam Weighting exponent.
#' @param sampleSize Multilevel sample size, kept for interface parity.
#' @param output Optional TSV output path in CLI-style tabular format.
#' @param gpu Logical flag mirroring the CLI `--gpu` switch. GPU execution
#'   currently supports only `mode = "fgsea"`, so `fgseaSimple()` will reject it.
#'   On WSL2, if CUDA is visible but WebGPU selects `llvmpipe`, start R with
#'   `GALLIUM_DRIVER=d3d12` and `MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA`.
#'
#' @return A data frame with fgsea-style result columns.
#' @export
fgseaSimple <- function(
  pathways,
  stats,
  nperm = 1000L,
  seed = NULL,
  nproc = 0L,
  minSize = 1L,
  maxSize = NULL,
  eps = 1e-50,
  scoreType = "std",
  gseaParam = 1.0,
  sampleSize = 101L,
  output = NULL,
  gpu = FALSE
) {
  fgsea(
    pathways = pathways,
    stats = stats,
    nPermSimple = as.integer(nperm),
    seed = seed,
    nproc = nproc,
    minSize = minSize,
    maxSize = maxSize,
    eps = eps,
    scoreType = scoreType,
    gseaParam = gseaParam,
    mode = "simple",
    nperm = as.integer(nperm),
    sampleSize = sampleSize,
    method = "classic",
    output = output,
    gpu = gpu
  )
}

#' Multilevel fgsea interface
#'
#' @param pathways Either a named list of character vectors or a path to a GMT file.
#' @param stats Named numeric vector of preranked statistics, or a path to a ranked-list file.
#' @param nPermSimple Simple-stage permutation count used before multilevel refinement.
#'   For CPU/GPU or R/GPU comparisons, prefer `100000L` as a practical baseline;
#'   use `10000L` only as a smoke tier and `1000000L` for final tail/stress
#'   checks when runtime allows.
#' @param seed Optional integer RNG seed. `NULL` uses a fresh random seed.
#' @param nproc Number of worker threads. `0` keeps the default Rayon behavior.
#' @param minSize Minimum pathway size.
#' @param maxSize Maximum pathway size. Defaults to `length(stats) - 1`.
#' @param eps Multilevel epsilon parameter.
#' @param scoreType One of `"std"`, `"pos"`, `"neg"`.
#' @param gseaParam Weighting exponent.
#' @param sampleSize Multilevel sample size.
#' @param output Optional TSV output path in CLI-style tabular format.
#' @param gpu Logical flag mirroring the CLI `--gpu` switch. GPU execution
#'   currently supports only `mode = "fgsea"`, so `fgseaMultilevel()` will reject it.
#'   On WSL2, if CUDA is visible but WebGPU selects `llvmpipe`, start R with
#'   `GALLIUM_DRIVER=d3d12` and `MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA`.
#'
#' @return A data frame with fgsea-style result columns.
#' @export
fgseaMultilevel <- function(
  pathways,
  stats,
  nPermSimple = 1000L,
  seed = NULL,
  nproc = 0L,
  minSize = 1L,
  maxSize = NULL,
  eps = 1e-50,
  scoreType = "std",
  gseaParam = 1.0,
  sampleSize = 101L,
  output = NULL,
  gpu = FALSE
) {
  fgsea(
    pathways = pathways,
    stats = stats,
    nPermSimple = nPermSimple,
    seed = seed,
    nproc = nproc,
    minSize = minSize,
    maxSize = maxSize,
    eps = eps,
    scoreType = scoreType,
    gseaParam = gseaParam,
    mode = "multilevel",
    nperm = NULL,
    sampleSize = sampleSize,
    output = output,
    gpu = gpu
  )
}
