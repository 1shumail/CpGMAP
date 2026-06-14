# CpGMAP — original final-year project overview

This summarises the original 2018–19 Bioinformatics final-year project that CpGMAP
grew out of. It is a sanitised overview (no personal data); the modern, reproducible
analysis lives in [`../python/`](../python/) and [`../r/`](../r/).

## Goal

Build a tissue-specific machine-learning tool to predict the **methylation status**
of CpG sites in human DNA. The gap it addressed: existing tools either *detected* CpG
islands **or** *predicted* methylation, and were not tissue-specific.

## Scope (from the SRS)

- Predict whether a CpG region is **methylated** (gene likely repressed) or
  **unmethylated** (gene likely expressed).
- Do this **per tissue**, using whole-genome bisulfite-sequencing data.
- Intended users: bioinformatics students, teachers, and researchers (e.g. locating
  housekeeping genes).

## System architecture

A three-tier Windows desktop application:

```
┌─────────────────────────────┐
│  C# WinForms UI              │   Login · Registration · Sequence Input · Output
│  (Login / Input / Output)    │
└──────────────┬──────────────┘
               │  Entity Framework
┌──────────────▼──────────────┐
│  SQL Server database         │   users + tissue methylation segments
│  (CpGMAP.mdf)                │
└──────────────┬──────────────┘
               │  data export
┌──────────────▼──────────────┐
│  R / ML model                │   methylation-status classifier
│  (neural network → SVM)      │
└─────────────────────────────┘
```

## Method

Tissue methylation segments (NGSmethDB) were used to train a classifier on
region features. The proposal targeted a **support-vector machine** ("MethFinder");
the surviving implementation used a **neural network**. Both are reconstructed in
[`../r/cpgmap.R`](../r/cpgmap.R) and modernised in
[`../python/cpgmap_analysis.py`](../python/cpgmap_analysis.py).

## My role

A 3-person group project that I **led** as project leader, covering the full software
lifecycle: requirements (SRS), use-case and UML/ER design, implementation, and the
data-science model. The literature review drew on MethCGI, CpGcluster, CpGIMethPred
and related methylation-prediction work.

## Software lifecycle artefacts produced

Project proposal & literature review · Software Requirements Specification (SRS) ·
project analysis · use-case and UML/ER diagrams · project-management plan · poster ·
presentations. (Original documents are retained privately; this overview omits all
personal data.)
