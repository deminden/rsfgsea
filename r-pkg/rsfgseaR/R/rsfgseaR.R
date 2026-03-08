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

.validate_choice <- function(value, name, choices) {
  if (!is.character(value) || length(value) != 1L || !(tolower(value) %in% choices)) {
    stop(name, " must be one of: ", paste(choices, collapse = ", "), ".", call. = FALSE)
  }
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
#' @param seed Optional integer RNG seed. `NULL` uses a fresh random seed.
#' @param nproc Number of worker threads. `0` keeps the default Rayon behavior.
#' @param minSize Minimum pathway size.
#' @param maxSize Maximum pathway size. Defaults to `length(stats) - 1`.
#' @param eps Multilevel epsilon parameter.
#' @param scoreType One of `"std"`, `"pos"`, `"neg"`.
#' @param gseaParam Weighting exponent.
#' @param mode One of `"fgsea"`, `"simple"`, `"multilevel"`.
#' @param nperm Optional fixed-permutation override for wrapper mode.
#' @param sampleSize Multilevel sample size.
#' @param output Optional TSV output path. When set, results are also written in
#'   the same column shape as the CLI.
#' @param gpu Logical flag mirroring the CLI `--gpu` switch. Uses the same
#'   hybrid GPU path as the Rust CLI and currently supports only `mode = "fgsea"`.
#'
#' @return A data frame with fgsea-style result columns.
#' @export
fgsea <- function(
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
  mode = "fgsea",
  nperm = NULL,
  sampleSize = 101L,
  output = NULL,
  gpu = FALSE
) {
  stats <- .normalize_stats(stats)
  .validate_integerish_scalar(nPermSimple, "nPermSimple", min_value = 1L)
  if (!is.null(seed)) {
    .validate_integerish_scalar(seed, "seed", min_value = 0L)
  }
  .validate_integerish_scalar(nproc, "nproc", min_value = 0L)
  .validate_integerish_scalar(minSize, "minSize", min_value = 1L)
  .validate_integerish_scalar(sampleSize, "sampleSize", min_value = 1L)
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
  .validate_choice(mode, "mode", c("fgsea", "simple", "multilevel"))
  .validate_choice(scoreType, "scoreType", c("std", "pos", "neg"))
  if (!is.logical(gpu) || length(gpu) != 1L || is.na(gpu)) {
    stop("gpu must be TRUE or FALSE.", call. = FALSE)
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
    as.integer(minSize),
    if (is.null(maxSize)) -1L else as.integer(maxSize),
    eps,
    scoreType,
    gseaParam,
    mode,
    if (is.null(nperm)) -1L else as.integer(nperm),
    as.integer(sampleSize),
    gpu
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
#' @param nperm Number of permutations.
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
    output = output,
    gpu = gpu
  )
}

#' Multilevel fgsea interface
#'
#' @param pathways Either a named list of character vectors or a path to a GMT file.
#' @param stats Named numeric vector of preranked statistics, or a path to a ranked-list file.
#' @param nPermSimple Simple-stage permutation count used before multilevel refinement.
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
