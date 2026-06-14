# Pure feature-engineering helpers for CpGMAP.
# Mirrors python/cpgmap_analysis.py:add_features so both languages share one definition.

METH_THRESHOLD <- 0.5   # a segment is "methylated" if methRatio >= this

add_features <- function(df) {
  keep <- !is.na(df$methRatio) & !is.na(df$cytosineCount) &
          !is.na(df$start) & !is.na(df$end)
  df <- df[keep, , drop = FALSE]
  df$region_length <- pmax(df$end - df$start, 1)        # clip to >= 1 (no div-by-zero)
  df$cpg_density   <- df$cytosineCount / df$region_length
  df$methylated    <- as.integer(df$methRatio >= METH_THRESHOLD)
  df
}
