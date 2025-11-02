# SVCCA-Based Representation Reuse Discovery on CIFAR-10

## Two-Stage Workflow

This experiment is organized into two clear stages to separate model training from analysis.

### Stage 1 — Katib Training & Artifact Collection
- **Goal:** Train ResNet-18 under different hyperparameters and collect checkpoints and activations.  
- **Files:** `cifar10-reuse-discovery.yaml`, `Dockerfile`, `train.py`  
- **Output:** Each trial uploads its checkpoints and layer activations to MinIO under `exp1/<trial>/…`.

### Stage 2 — Representation Analysis & Reporting
- **Goal:** Analyze stored activations across trials to measure similarity using SVCCA and CKA.  
- **Files:** `build_analysis_dataset.py`, `svcca_cka_report.ipynb`  
- **Output:** CSV matrices (`cka_epoch*.csv`, `svcca_epoch*.csv`, `pairwise_scores.csv`) and visual reports (heatmaps, scatter plots, summaries).

---
## 1. Experiment Overview
This experiment investigates how representations learned by neural networks converge across trials during hyperparameter tuning. Using **Singular Vector Canonical Correlation Analysis (SVCCA)** and **Linear CKA**, we quantify similarity between internal representations of networks trained with different hyperparameters.

### Research Basis
Based on *Raghu et al., NIPS 2017*, SVCCA compares neural representations as subspaces, invariant to affine transformations. It enables us to measure whether different networks learn similar features—crucial for identifying **computation reuse opportunities** in large-scale hyperparameter sweeps.

### Objectives
- Detect representational convergence across Katib trials.
- Quantify layer-wise redundancy for potential compute reuse.
- Visualize layer learning dynamics over epochs.

---

## 2. Katib Experiment Configuration (`cifar10-reuse-discovery.yaml`)

**Algorithm:** Grid Search  
**Parallel Trials:** 3  
**Max Trials:** 24  
**Max Failures:** 3  

**Objective:** Maximize `val_accuracy` ≥ 0.90  
**Metrics Logged:** `train_loss`, `val_accuracy`, `wall_time_seconds`

**Hyperparameters Tuned:**
| Name | Type | Values |
|------|------|---------|
| lr | categorical | 0.001, 0.0005, 0.0001 |
| weight_decay | categorical | 0.0001, 0.0005 |
| dropout | categorical | 0.0, 0.3 |
| optimizer | categorical | adam, sgd |

Each trial runs in a container image:
```
10.249.190.44:5000/cifar10-katib:base-2.1.0-cu118
```
with MinIO environment variables injected for artifact upload.

**Artifact Structure:**
```
exp1/
 └── <trial-id>/
     ├── checkpoints/epoch_{E}.pt
     ├── checkpoints/best.pt
     └── activations/epoch_{E}/{layer}.npz
```

---

## 3. Containerized Training Environment (`Dockerfile`)

**Base Image:** `pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel`  
**Installed Packages:** `torchvision`, `torchaudio`, `numpy`, `tqdm`, `boto3`  
**Purpose:** Ensure reproducibility and GPU acceleration for both Katib jobs and SVCCA analysis.

This lightweight image ensures artifact compatibility and allows both training and post-analysis to be executed within the same environment.

---

## 4. Training Pipeline & Activation Logging (`train.py`)

### Data Pipeline
- Dataset: CIFAR-10 with FakeData fallback (offline mode).
- Augmentations: RandomCrop(32) + HorizontalFlip().
- Probe Subset: 512 test samples used for layer activation capture.

### Model
ResNet-18 (`torchvision.models.resnet18`) with optional dropout head.

### Optimizers
Supports `adam` or `sgd` (momentum=0.9, nesterov=True).

### Hook-Based Activation Capture
`HookKeeper` registers forward hooks on:
```
conv1, layer1.1.relu, layer2.1.relu, layer3.1.relu, layer4.1.relu, fc
```
Captures flattened per-sample activations and stores as compressed `.npz`.

### Artifact Management
| Artifact | Location | Description |
|-----------|-----------|-------------|
| Checkpoints | `/checkpoints/epoch_{E}.pt`, `/checkpoints/best.pt` | Model states |
| Activations | `/activations/epoch_{E}/{layer}.npz` | Layerwise activation vectors |
| Metrics | stdout (`name=value`) | Parsed by Katib collector |

---

## 5. Post-Training Analysis (`build_analysis_dataset.py`)

### Purpose
Aggregate MinIO-stored activations and compute **pairwise representational similarities** between all Katib trials.

### Core Steps
1. **Trial Discovery** — Enumerates trials under `PREFIX/*/activations/`.
2. **Feature Normalization** — Zero-mean activations per trial.
3. **Metrics Computed:**
   - **Linear CKA** — via centered Gram matrices.
   - **SVCCA** — via SVD truncation → whitening → canonical correlation.
4. **Outputs:**
   - `analysis/cka_epoch{E}_{layer}.csv`
   - `analysis/svcca_epoch{E}_{layer}.csv`
   - `pairwise_scores.csv`
   - `trial_summary.csv`
5. **Optional Uploads** — Back to MinIO (`PREFIX/analysis/`).

### Mathematical Summary
For each pair of trials (X, Y):

\`\`\`math

ho_{SVCCA} = \frac{1}{k} \sum_{i=1}^{k} \rho_i
\qquad
CKA(X, Y) = \frac{HSIC(X, Y)}{\sqrt{HSIC(X, X) HSIC(Y, Y)}}
\`\`\`

### Interpretation
- High CKA/SVCCA → strong subspace similarity → potential **reusable checkpoints**.
- Early layers stabilize faster → supports **Freeze Training** and **bottom-up convergence**.

---

## 6. Visualization & Reporting (`svcca_cka_report.ipynb`)

This notebook reads the CSV outputs and visualizes similarity matrices and summary statistics.

### Visualizations
- **Heatmaps:** pairwise CKA & SVCCA across trials.
- **Scatter Plots:** correlation between CKA and SVCCA.
- **Summaries:** mean/median/std grouped by layer and epoch.

### Example Workflow
```python
df = pd.read_csv("pairwise_scores.csv")
epochs = sorted(df['epoch'].unique())
layers = sorted(df['layer'].unique())
# Heatmap for epoch=20, layer=layer3.1.relu
```

### Insights
- Early convolutional layers: consistently high similarity (≈0.9–1.0).  
- Deeper layers: more variance, trial-specific specialization.  
- Confirms *“lower layers learn first”* trend (Raghu et al., 2017).

---

## 7. System-Level Reflection

| Aspect | Mechanism | Impact on Compute Reuse |
|--------|------------|-------------------------|
| **Activation Capture** | Hook-based extraction | Enables representation comparison without retraining |
| **SVCCA/CKA Analysis** | Subspace correlation | Identifies reusable checkpoints & converged layers |
| **Freeze Training** | Sequential layer freezing | Saves gradient computation in stabilized layers |
| **MinIO Artifact Store** | Structured hierarchy | Supports automated cross-trial reuse discovery |

---

## 8. Conclusion

This pipeline reproduces SVCCA-based representation analysis entirely within **Kubeflow Katib**.

- **Bridges theory and practice:** from representation similarity to computation reuse.  
- **Findings:** Early layers exhibit strong representational convergence → viable reuse candidates.  
- **Outcome:** Provides a foundation for *Cross-Trial Checkpoint Graph (CTCG)* and *Delta-Semantic Reuse* research.

> The integration of SVCCA and Katib demonstrates that large-scale hyperparameter tuning can benefit from representational analysis to reduce redundant computation and accelerate training.

---
