# Data — tissue-specific methylation segments

All data in `data/raw/` are **methylation segments** (genome regions of homogeneous
methylation) downloaded from **NGSmethDB**, the whole-genome methylation database
maintained by the bioinfoUGR group at the University of Granada.

- **Source:** NGSmethDB — methylation-segment dumps
  (`bioinfo2.ugr.es` → Database Content → methylation segments)
- **Assay:** whole-genome bisulfite sequencing (WGBS), single-cytosine resolution,
  CpG context. Underlying methylomes are human tissue samples from the
  Roadmap Epigenomics project, re-served by NGSmethDB.
- **Segment threshold:** files suffixed `_99` / `_95` are the NGSmethDB
  segmentation levels (90 / 95 / 99).
- **Tissues (7):** adipose, adrenal gland, esophagus, heart aorta, pancreas,
  small intestine, spleen.

### File format (BED)

| column | meaning |
|---|---|
| `chrom` | chromosome |
| `start`, `end` | segment coordinates |
| `id` | `chrom_start` identifier |
| `score` | methylation score (0–1000, = methRatio × 1000) |
| `strand` | strand (`.`) |
| `methRatio` | methylation ratio of the segment (0–1) |
| `cytosineCount` | number of cytosines in the segment |

### How to cite the data
- Hackenberg M, Barturen G, Oliver JL. **NGSmethDB: a database for next-generation
  sequencing single-cytosine-resolution DNA methylation data.** *Nucleic Acids
  Research* 2011；39(suppl_1):D75–D79. doi:10.1093/nar/gkq942
- Lebrón R, Gómez-Martín C, Carpena P, Bernaola-Galván P, Barturen G, Hackenberg M,
  Oliver JL. **NGSmethDB 2017: enhanced methylomes and differential methylation.**
  *Nucleic Acids Research* 2016. doi:10.1093/nar/gkw996

> Files are kept bzip2-compressed (~6 MB total). The analysis scripts read them
> directly without manual decompression.
