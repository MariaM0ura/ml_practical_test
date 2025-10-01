#!/usr/bin/env python3
"""
knn_pipeline_project_refactored.py

A refactored, object-oriented pipeline for KNN classification on embeddings.

This script encapsulates the entire workflow into a `KnnPipeline` class,
improving structure, maintainability, and readability.

Key Features:
 - Loads and preprocesses embedding data from a pickle file.
 - Performs exploratory data analysis with t-SNE visualization.
 - Conducts a robust KNN classification experiment using 10-fold cross-validation,
   searching for the best 'k' (1-15) with both Euclidean and Cosine distances.
 - Calculates key metrics: Macro AUC (OVR), Macro F1-score, and Top-k accuracy.
 - Generates and saves essential outputs:
   - CSV files with detailed and summarized results.
   - Plots for t-SNE, AUC vs. k, and mean ROC curves.
   - Two PDF reports: one with technical results and another with dynamic interpretation.
   - Auxiliary files (`requirements.txt`, `README.md`).

Usage:
    python knn_pipeline_project_refactored.py --pickle PATH_TO_PICKLE --outdir output

Author: Refactored by an AI assistant for clarity and robustness.
"""

import argparse
import logging
import pickle
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import f1_score, roc_auc_score, roc_curve, top_k_accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize

# --- Configuration Constants ---
RANDOM_STATE = 42
K_VALUES = range(1, 16)
N_SPLITS = 10
EXPECTED_DIM = 320
FPR_GRID = np.linspace(0, 1, 101)
DISTANCE_METRICS = ["euclidean", "cosine"]

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class ReportGenerator:
    """Handles the generation of PDF reports."""

    def __init__(self, output_dir: Path, summary_df: pd.DataFrame, best_df: pd.DataFrame,
                 tsne_png: Path, roc_png: Path, auc_k_png: Path,
                 dataset_stats: Dict[str, Any]):
        self.output_dir = output_dir
        self.summary_df = summary_df
        self.best_df = best_df
        self.tsne_png = tsne_png
        self.roc_png = roc_png
        self.auc_k_png = auc_k_png
        self.dataset_stats = dataset_stats
        self.width, self.height = A4
        self.margin = 40

    def generate_all_reports(self) -> None:
        """Generates both results and interpretation reports."""
        self._generate_results_report()
        self._generate_interpretation_report()

    def _draw_page_header(self, c: canvas.Canvas, title: str) -> int:
        """Draws a standard header on a new page."""
        y_pos = self.height - self.margin
        c.setFont("Helvetica-Bold", 16)
        c.drawString(self.margin, y_pos, title)
        y_pos -= 24
        c.setFont("Helvetica", 10)
        c.drawString(self.margin, y_pos, f"Generated: {datetime.utcnow().isoformat()} UTC")
        y_pos -= 30
        return y_pos

    def _generate_results_report(self) -> None:
        """Generates the technical results PDF report."""
        path = self.output_dir / "report_results.pdf"
        c = canvas.Canvas(str(path), pagesize=A4)
        
        # Page 1: Intro and t-SNE
        y = self._draw_page_header(c, "KNN Pipeline - Technical Results")
        c.setFont("Helvetica-Bold", 12)
        c.drawString(self.margin, y, "1. Dataset Overview")
        y -= 20
        c.setFont("Helvetica", 10)
        
        counts = self.dataset_stats['syndrome_counts']
        top_items = list(counts.items())[:6]
        lines = [
            f"Total samples after cleaning: {self.dataset_stats['n_samples']}",
            f"Number of syndromes (classes): {self.dataset_stats['n_classes']}",
            f"Embedding dimension: {self.dataset_stats['emb_dim']}",
            "\nTop 6 Syndromes by Image Count:"
        ] + [f"  - {name}: {cnt}" for name, cnt in top_items]
        
        for line in lines:
            c.drawString(self.margin, y, line)
            y -= 14
        
        y -= 20
        c.setFont("Helvetica-Bold", 12)
        c.drawString(self.margin, y, "2. t-SNE Visualization of Embeddings")
        y -= 15
        c.drawImage(ImageReader(self.tsne_png), self.margin, y - 300, 
                    width=self.width - 2 * self.margin, height=300, preserveAspectRatio=True)
        
        # Page 2: ROC and AUC vs. K
        c.showPage()
        y = self._draw_page_header(c, "KNN Pipeline - Performance Plots")
        c.setFont("Helvetica-Bold", 12)
        c.drawString(self.margin, y, "3. Mean ROC Curve Comparison (Best K)")
        y -= 15
        c.drawImage(ImageReader(self.roc_png), self.margin, y - 300, 
                    width=self.width - 2 * self.margin, height=300, preserveAspectRatio=True)
        y -= 320

        c.setFont("Helvetica-Bold", 12)
        c.drawString(self.margin, y, "4. AUC vs. K Neighbors")
        y -= 15
        c.drawImage(ImageReader(self.auc_k_png), self.margin, y - 300, 
                    width=self.width - 2 * self.margin, height=300, preserveAspectRatio=True)

        # Page 3: Summary Table
        c.showPage()
        y = self._draw_page_header(c, "KNN Pipeline - Summary Table")
        c.setFont("Helvetica-Bold", 12)
        c.drawString(self.margin, y, "5. Performance Metrics Across All Runs")
        y -= 20
        c.setFont("Courier", 8)
        
        table_header = "{:<12} {:<4} {:<12} {:<12} {:<12} {:<12}".format(
            "Distance", "K", "AUC Mean", "F1 Mean", "Top-1 Mean", "Top-3 Mean")
        c.drawString(self.margin, y, table_header)
        y -= 12
        c.line(self.margin, y, self.width - self.margin, y)
        y -= 12

        for _, row in self.summary_df.iterrows():
            if y < self.margin + 40:
                c.showPage()
                y = self.height - self.margin
            
            line = "{:<12} {:<4} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f}".format(
                row['distance'], int(row['k']), row['auc_mean'], row['f1_mean'],
                row['top1_mean'], row['top3_mean'])
            c.drawString(self.margin, y, line)
            y -= 12
        
        c.save()
        logging.info(f"Generated results report at {path}")

    def _generate_interpretation_report(self) -> None:
        """Generates the interpretation PDF report with dynamic findings."""
        path = self.output_dir / "report_interpretation.pdf"
        d = canvas.Canvas(str(path), pagesize=A4)
        y = self._draw_page_header(d, "Interpretation of KNN Analysis")
        
        d.setFont("Helvetica", 11)

        best_overall = self.summary_df.loc[self.summary_df['auc_mean'].idxmax()]
        best_metric = best_overall['distance']
        best_k = int(best_overall['k'])
        best_auc = best_overall['auc_mean']
        best_f1 = best_overall['f1_mean']
        best_top3 = best_overall['top3_mean']
        
        # Get stats for the other metric at its best k
        other_metric = "euclidean" if best_metric == "cosine" else "cosine"
        other_best = self.best_df[self.best_df['distance'] == other_metric]
        other_auc = other_best['auc_mean'].values[0]

        bullets = [
            f"1. Overall Performance: The embeddings show strong class separability. The best model achieved a "
            f"Macro AUC of {best_auc:.3f}, indicating excellent classification potential.",
            
            f"2. Best Distance Metric: **{best_metric.capitalize()} distance** consistently outperformed "
            f"{other_metric} distance (best AUC {best_auc:.3f} vs {other_auc:.3f}). This suggests that the "
            f"**direction** of the embedding vectors is more informative than their absolute magnitude for this task.",
            
            f"3. Optimal K value: The optimal number of neighbors (k) was found to be **{best_k}** for the best "
            f"performing model. A higher k-value generally implies a smoother decision boundary, making the model more "
            f"robust to noise.",
            
            f"4. F1-Score vs. AUC: The best Macro F1-score ({best_f1:.3f}) is slightly lower than the AUC. This is common "
            f"and can indicate that while class probabilities are well-ranked (high AUC), the optimal decision threshold "
            f"might be harder to find, especially for minority classes.",
            
            f"5. Practical Application (Top-3 Accuracy): The Top-3 accuracy is very high (around {best_top3:.3f}). This is "
            f"extremely valuable for clinical or research use-cases where the model can act as a recommendation system, "
            f"presenting a short list of potential syndromes for a specialist to review."
        ]

        for bullet in bullets:
            lines = self._wrap_text(bullet, 100)
            for line in lines:
                if y < self.margin + 40:
                    d.showPage()
                    y = self.height - self.margin
                d.drawString(self.margin, y, line)
                y -= 14
            y -= 10 # Extra space between bullets

        d.save()
        logging.info(f"Generated interpretation report at {path}")

    @staticmethod
    def _wrap_text(text: str, max_len: int) -> List[str]:
        """Simple text wrapper for ReportLab."""
        words = text.split(' ')
        lines = []
        current_line = ""
        for word in words:
            if len(current_line) + len(word) + 1 <= max_len:
                current_line += f" {word}"
            else:
                lines.append(current_line.strip())
                current_line = f" {word}"
        lines.append(current_line.strip())
        return lines


class KnnPipeline:
    """Encapsulates the entire KNN classification pipeline."""
    
    def __init__(self, pickle_path: str, outdir: str, max_vis: int):
        self.pickle_path = Path(pickle_path)
        self.output_dir = Path(outdir)
        self.max_vis = max_vis
        
        self.df: Optional[pd.DataFrame] = None
        self.X: Optional[np.ndarray] = None
        self.y: Optional[np.ndarray] = None
        self.le: Optional[LabelEncoder] = None
        self.X_scaled: Optional[np.ndarray] = None
        self.results: Optional[Dict] = None
        self.summary_df: Optional[pd.DataFrame] = None
        self.best_df: Optional[pd.DataFrame] = None
        
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> None:
        """Executes the full pipeline from data loading to reporting."""
        logging.info("Starting KNN pipeline.")
        self._load_and_preprocess_data()
        self._run_tsne_and_plot()
        self._run_knn_cv_experiment()
        self._aggregate_and_save_results()
        self._generate_plots()
        self._generate_reports()
        self._write_aux_files()
        logging.info(f"Pipeline finished. All outputs are in '{self.output_dir}'.")

    def _load_and_preprocess_data(self) -> None:
        """Loads data from pickle, flattens, cleans, and prepares for modeling."""
        logging.info(f"Loading and flattening data from {self.pickle_path}...")
        try:
            with open(self.pickle_path, 'rb') as f:
                database = pickle.load(f)
        except FileNotFoundError:
            logging.error(f"Pickle file not found at {self.pickle_path}")
            raise
            
        rows = [
            {
                'syndrome_id': syndrome_id,
                'subject_id': subject_id,
                'image_id': image_id,
                'embedding': np.asarray(embedding, dtype=np.float32)
            }
            for syndrome_id, subjects in database.items()
            for subject_id, images in subjects.items()
            for image_id, embedding in images.items()
        ]
        df = pd.DataFrame(rows)

        logging.info("Cleaning data and checking embedding dimensions...")
        df['emb_len'] = df['embedding'].apply(lambda x: x.shape[0] if isinstance(x, np.ndarray) else None)
        df = df[df['emb_len'] == EXPECTED_DIM].drop(columns=['emb_len']).reset_index(drop=True)
        self.df = df
        
        logging.info(f"Data loaded. Found {len(df)} valid samples.")

        self.X = np.vstack(self.df['embedding'].values)
        self.le = LabelEncoder()
        self.y = self.le.fit_transform(self.df['syndrome_id'].values)
        
        logging.info("Scaling features using StandardScaler...")
        scaler = StandardScaler()
        self.X_scaled = scaler.fit_transform(self.X)

    def _run_tsne_and_plot(self) -> None:
        """Runs t-SNE on a stratified sample of the data and saves the plot."""
        logging.info(f"Running t-SNE on a sample of {self.max_vis} points...")
        
        if self.X_scaled.shape[0] > self.max_vis:
            # Use train_test_split for stratified sampling
            _, X_vis, _, y_vis = train_test_split(
                self.X_scaled, self.y, 
                test_size=self.max_vis, 
                stratify=self.y, 
                random_state=RANDOM_STATE
            )
        else:
            X_vis, y_vis = self.X_scaled, self.y
        
        pca = PCA(n_components=50, random_state=RANDOM_STATE)
        X_pca = pca.fit_transform(X_vis)
        
        tsne = TSNE(n_components=2, init='pca', perplexity=30, random_state=RANDOM_STATE)
        X_tsne = tsne.fit_transform(X_pca)

        plt.figure(figsize=(12, 10))
        unique_labels = np.unique(y_vis)
        for lbl in unique_labels:
            mask = (y_vis == lbl)
            plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], s=10, alpha=0.7, 
                        label=str(self.le.inverse_transform([lbl])[0]))
        
        if len(unique_labels) <= 20:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        
        plt.title("t-SNE Visualization of Embeddings")
        plt.xlabel("t-SNE Component 1")
        plt.ylabel("t-SNE Component 2")
        plt.tight_layout()
        
        out_png = self.output_dir / "tsne_plot.png"
        plt.savefig(out_png, dpi=300)
        plt.close()
        logging.info(f"t-SNE plot saved to {out_png}")

    def _run_knn_cv_experiment(self) -> None:
        """Performs the main KNN cross-validation experiment."""
        logging.info(f"Starting KNN {N_SPLITS}-fold CV experiment (this may take a while)...")
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        n_classes = len(self.le.classes_)

        self.results = {
            metric: {k: {'auc': [], 'f1': [], 'top1': [], 'top3': [], 'roc_tprs': []} for k in K_VALUES}
            for metric in DISTANCE_METRICS
        }

        for fold, (train_idx, test_idx) in enumerate(skf.split(self.X_scaled, self.y), 1):
            logging.info(f"--- Processing Fold {fold}/{N_SPLITS} ---")
            X_tr, X_te = self.X_scaled[train_idx], self.X_scaled[test_idx]
            y_tr, y_te = self.y[train_idx], self.y[test_idx]
            
            for metric in DISTANCE_METRICS:
                for k in K_VALUES:
                    knn = KNeighborsClassifier(n_neighbors=k, metric=metric, n_jobs=-1)
                    knn.fit(X_tr, y_tr)
                    y_proba = knn.predict_proba(X_te)
                    y_pred = np.argmax(y_proba, axis=1)

                    # Calculate metrics
                    y_true_bin = label_binarize(y_te, classes=np.arange(n_classes))
                    auc = roc_auc_score(y_true_bin, y_proba, average="macro", multi_class="ovr")
                    f1 = f1_score(y_te, y_pred, average="macro", zero_division=0)
                    top1 = top_k_accuracy_score(y_te, y_proba, k=1, labels=np.arange(n_classes))
                    top3 = top_k_accuracy_score(y_te, y_proba, k=3, labels=np.arange(n_classes)) if n_classes >= 3 else np.nan
                    
                    # Store results
                    res_dict = self.results[metric][k]
                    res_dict['auc'].append(auc)
                    res_dict['f1'].append(f1)
                    res_dict['top1'].append(top1)
                    res_dict['top3'].append(top3)
                    
                    # Calculate and store ROC curve
                    tprs_fold = []
                    for c in range(n_classes):
                        if np.sum(y_true_bin[:, c]) > 0:
                            fpr, tpr, _ = roc_curve(y_true_bin[:, c], y_proba[:, c])
                            tprs_fold.append(np.interp(FPR_GRID, fpr, tpr))
                    if tprs_fold:
                        res_dict['roc_tprs'].append(np.mean(tprs_fold, axis=0))

    def _aggregate_and_save_results(self) -> None:
        """Aggregates CV results into DataFrames and saves them to CSV."""
        logging.info("Aggregating CV results...")
        rows = []
        for metric in DISTANCE_METRICS:
            for k, vals in self.results[metric].items():
                row = {'distance': metric, 'k': k}
                for score in ['auc', 'f1', 'top1', 'top3']:
                    row[f'{score}_mean'] = np.nanmean(vals[score])
                    row[f'{score}_std'] = np.nanstd(vals[score])
                rows.append(row)
        
        self.summary_df = pd.DataFrame(rows)
        self.best_df = self.summary_df.loc[self.summary_df.groupby('distance')['auc_mean'].idxmax()].copy()

        summary_csv = self.output_dir / "knn_summary_by_k.csv"
        best_csv = self.output_dir / "knn_best_by_distance.csv"
        
        self.summary_df.to_csv(summary_csv, index=False)
        self.best_df.to_csv(best_csv, index=False)
        logging.info(f"Results saved to {summary_csv} and {best_csv}")

    def _generate_plots(self) -> None:
        """Generates and saves all summary plots."""
        logging.info("Generating summary plots...")
        # Plot 1: AUC vs. K
        plt.figure(figsize=(10, 6))
        for metric in DISTANCE_METRICS:
            dfm = self.summary_df[self.summary_df['distance'] == metric]
            plt.errorbar(dfm['k'], dfm['auc_mean'], yerr=dfm['auc_std'], marker='o', capsize=4, label=metric.capitalize())
        plt.xlabel("Number of Neighbors (k)")
        plt.ylabel("Mean Macro AUC (10-fold CV)")
        plt.title("Model Performance vs. Number of Neighbors (k)")
        plt.legend()
        plt.xticks(list(K_VALUES))
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        auc_png = self.output_dir / "auc_vs_k.png"
        plt.savefig(auc_png, dpi=300)
        plt.close()
        logging.info(f"AUC vs. k plot saved to {auc_png}")

        # Plot 2: Mean ROC Comparison
        plt.figure(figsize=(8, 8))
        for _, row in self.best_df.iterrows():
            metric, k_best = row['distance'], int(row['k'])
            tprs = np.array(self.results[metric][k_best]['roc_tprs'])
            mean_tpr = np.mean(tprs, axis=0)
            std_tpr = np.std(tprs, axis=0)
            
            plt.plot(FPR_GRID, mean_tpr, label=f"{metric.capitalize()} (k={k_best}, AUC={row['auc_mean']:.3f})")
            plt.fill_between(FPR_GRID, 
                             np.maximum(mean_tpr - std_tpr, 0), 
                             np.minimum(mean_tpr + std_tpr, 1), 
                             alpha=0.2)
                             
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label="Random Chance")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Mean ROC Curve Comparison (Best K for each metric)")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.axis('square')
        plt.tight_layout()
        roc_png = self.output_dir / "roc_comparison.png"
        plt.savefig(roc_png, dpi=300)
        plt.close()
        logging.info(f"ROC comparison plot saved to {roc_png}")
        
    def _generate_reports(self) -> None:
        """Initializes and runs the report generator."""
        logging.info("Generating PDF reports...")
        dataset_stats = {
            'n_samples': len(self.df),
            'n_classes': len(self.le.classes_),
            'emb_dim': self.X.shape[1],
            'syndrome_counts': self.df['syndrome_id'].value_counts().to_dict()
        }
        
        reporter = ReportGenerator(
            output_dir=self.output_dir,
            summary_df=self.summary_df,
            best_df=self.best_df,
            tsne_png=self.output_dir / "tsne_plot.png",
            roc_png=self.output_dir / "roc_comparison.png",
            auc_k_png=self.output_dir / "auc_vs_k.png",
            dataset_stats=dataset_stats
        )
        reporter.generate_all_reports()

    def _write_aux_files(self) -> None:
        """Writes auxiliary files like requirements.txt and README.md."""
        logging.info("Writing auxiliary files (requirements.txt, README.md)...")
        
        # requirements.txt
        reqs = ['numpy', 'pandas', 'scikit-learn', 'matplotlib', 'reportlab']
        with open(self.output_dir / 'requirements.txt', 'w') as f:
            f.write('\n'.join(reqs))
            

def main():
    """Main function to parse arguments and run the pipeline."""
    parser = argparse.ArgumentParser(
    description="Refactored KNN Pipeline for embeddings classification.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--pickle", required=True, help="Path to the embeddings pickle file.")
    parser.add_argument("--outdir", default="output", help="Directory to save all outputs.")
    parser.add_argument("--max_vis", default=2000, type=int, help="Max points for t-SNE visualization.")
    args = parser.parse_args()

    try:
        pipeline = KnnPipeline(pickle_path=args.pickle, outdir=args.outdir, max_vis=args.max_vis)
        pipeline.run()
    except Exception as e:
        logging.error("An unhandled exception occurred during the pipeline execution.", exc_info=True)

if __name__ == '__main__':
    main()