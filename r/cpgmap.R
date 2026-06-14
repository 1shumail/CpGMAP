# =============================================================================
# CpGMAP — methylation-status classifier (R)  [reconstructed for the record]
#
# Final Year Project, BS Bioinformatics (2018-19). This script reconstructs the
# original R analysis: predict the binary methylation status of tissue-specific
# CpG methylation segments using a neural network. The 2018 proposal targeted an
# SVM ("MethFinder"); the surviving variant used a neural net, so both are shown.
#
# Data: NGSmethDB methylation segments (WGBS, Roadmap Epigenomics human tissues).
# Columns: chrom, start, end, id, score, strand, methRatio, cytosineCount
#
# Run from the repository root:  Rscript r/cpgmap.R
# =============================================================================

pkgs <- c("readr", "dplyr", "neuralnet", "e1071")
for (p in pkgs) if (!require(p, character.only = TRUE)) {
  install.packages(p, repos = "https://cloud.r-project.org"); library(p, character.only = TRUE)
}

set.seed(42)

# ---- 1. Load one tissue (decompress a BED dump from data/raw) ----------------
# The original FYP used a per-tissue Excel export; here we read the BED directly.
bed_path <- "data/raw/spleen_99.bed.bz2"
cols <- c("chrom", "start", "end", "id", "score", "strand", "methRatio", "cytosineCount")
seg <- readr::read_tsv(bed_path, comment = "", skip = 1, col_names = cols,
                       show_col_types = FALSE)

# ---- 2. Features + binary label ---------------------------------------------
seg <- seg %>%
  filter(!is.na(methRatio), !is.na(cytosineCount)) %>%
  mutate(region_length = pmax(end - start, 1),
         cpg_density   = cytosineCount / region_length,
         # methylated (1) vs unmethylated (0) at a 0.5 ratio threshold
         label         = as.integer(methRatio >= 0.5))

# ---- 3. Train / test split ---------------------------------------------------
n   <- nrow(seg)
idx <- sample(seq_len(n), size = floor(0.75 * n))
train <- seg[idx, ]
test  <- seg[-idx, ]

# ---- 4a. Neural network (the surviving 2018 model) ---------------------------
# NOTE: methRatio/score directly encode the label, so this framing is partly
# circular and scores near-perfectly. It is kept to reproduce the original work.
nn <- neuralnet(label ~ score + methRatio + cytosineCount,
                data = train, hidden = 3, act.fct = "logistic",
                linear.output = FALSE, stepmax = 1e6)
nn_prob <- neuralnet::compute(nn, test[, c("score", "methRatio", "cytosineCount")])$net.result
nn_pred <- ifelse(nn_prob > 0.5, 1, 0)
cat("\nNeural net (leaky reference) accuracy:",
    round(mean(nn_pred == test$label), 3), "\n")

# ---- 4b. SVM on non-leaky structural features (the honest task) --------------
# CpG-dense regions (CpG islands) trend unmethylated, so structure carries real
# signal without using the methylation value itself.
svm_fit <- e1071::svm(as.factor(label) ~ region_length + cytosineCount + cpg_density,
                      data = train, kernel = "radial", scale = TRUE)
svm_pred <- predict(svm_fit, test[, c("region_length", "cytosineCount", "cpg_density")])
cat("SVM (non-leaky structure) accuracy:",
    round(mean(as.integer(as.character(svm_pred)) == test$label), 3), "\n")
print(table(predicted = svm_pred, actual = test$label))

# ---- 5. Save the network diagram ---------------------------------------------
png("results/r_neuralnet.png", width = 900, height = 600)
plot(nn, rep = "best")
dev.off()

cat("\nDone. See results/ for the network diagram.\n")
