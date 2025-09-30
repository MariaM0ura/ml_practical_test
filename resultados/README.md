# KNN Pipeline Project

This repository contains a single script `knn_pipeline_project.py` that performs the full pipeline described in the project deliverables:

- load/clean the pickle embeddings dataset
- exploratory data analysis
- t-SNE visualization
- KNN (Euclidean and Cosine) with 10-fold CV and k search (1..15)
- metrics and ROC averaging
- saves CSV summaries, plots, and two PDF reports

Usage:

```
python knn_pipeline_project.py --pickle PATH_TO_PICKLE --outdir output
```

Outputs will be written into `output/`.

Requirements are in `requirements.txt`.
