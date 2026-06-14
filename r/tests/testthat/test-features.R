library(testthat)
source(testthat::test_path("..", "..", "features.R"))   # robust to working directory

raw <- data.frame(
  start         = c(100, 200),
  end           = c(1100, 250),
  methRatio     = c(0.80, 0.42),
  cytosineCount = c(20, 5)
)

test_that("region_length is end - start, clipped to at least 1", {
  out <- add_features(raw)
  expect_equal(out$region_length, c(1000, 50))
})

test_that("cpg_density is cytosineCount / region_length", {
  out <- add_features(raw)
  expect_equal(out$cpg_density, c(20 / 1000, 5 / 50))
})

test_that("methylation label uses the 0.5 threshold", {
  out <- add_features(raw)
  expect_equal(out$methylated, c(1L, 0L))
})

test_that("methylation label is methylated exactly at 0.5", {
  d <- data.frame(start = 0, end = 10, methRatio = 0.5, cytosineCount = 1)
  expect_equal(add_features(d)$methylated, 1L)
})

test_that("rows with NA methRatio are dropped", {
  d <- data.frame(start = c(0, 0), end = c(10, 10),
                  methRatio = c(0.8, NA), cytosineCount = c(5, 5))
  expect_equal(nrow(add_features(d)), 1)
})
