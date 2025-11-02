#!/usr/bin/env python3
"""
Builds a consolidated analysis dataset from Katib/MinIO artifacts:
- Loads per-trial activation snapshots (npz) for selected epochs & layers
- Computes pairwise Linear CKA and SVCCA between trials per (epoch, layer)
- Optionally parses best validation accuracy from best.pt
- Saves:
  * analysis/cka_epoch{E}_{layer}.csv (matrix)
  * analysis/svcca_epoch{E}_{layer}.csv (matrix)
  * analysis/pairwise_scores.csv (long format)
  * analysis/trial_summary.csv    (per-trial metadata including best acc if present)

Environment variables:
  MINIO_ENDPOINT   (e.g., "10.249.190.44:9000" or "http://10.249.190.44:9000")
  MINIO_BUCKET     (default: "katib-artifacts")
  MINIO_ACCESS_KEY
  MINIO_SECRET_KEY
  MINIO_PREFIX     (default: "exp1")
  ANALYSIS_EPOCHS  (comma-separated, default: "1,2,4,8,16,20")
  ANALYSIS_LAYERS  (comma-separated resnet18 names, default provided below)
  OUTDIR_LOCAL     (default: "/workspace/out")
  UPLOAD_RESULTS   ("1" to push CSVs back to MinIO under {PREFIX}/analysis/)

Run example (inside your training image):
  python build_analysis_dataset.py
"""

import os, io, csv, itertools, json, sys
from pathlib import Path  # Add this import
from typing import Dict, List, Tuple
import numpy as np

# --- Default MinIO Environment (auto-set if not defined externally) ---
os.environ.setdefault("MINIO_ENDPOINT", "http://10.249.190.44:9000")
os.environ.setdefault("MINIO_BUCKET", "katib-artifacts")
os.environ.setdefault("MINIO_ACCESS_KEY", "minioadmin")
os.environ.setdefault("MINIO_SECRET_KEY", "minioadmin")
os.environ.setdefault("MINIO_PREFIX", "exp1")

# Optional: torch used only to read "best.pt" for val_acc metadata (if present)
try:
    import torch
except Exception:
    torch = None

try:
    import boto3
except Exception as e:
    print("boto3 is required (pip install boto3).", file=sys.stderr)
    raise

def env_get(name, default=None):
    v = os.getenv(name, default)
    return v

# ------------------ Config ------------------
PREFIX = env_get("MINIO_PREFIX", "exp1")
ENDPOINT = env_get("MINIO_ENDPOINT", "")
BUCKET   = env_get("MINIO_BUCKET", "katib-artifacts")
AK       = env_get("MINIO_ACCESS_KEY", "minioadmin")
SK       = env_get("MINIO_SECRET_KEY", "minioadmin")

EPOCHS = [int(x) for x in env_get("ANALYSIS_EPOCHS", "1,2,4,8,16,20").split(",")]
LAYERS = env_get("ANALYSIS_LAYERS",
                 "conv1,layer1.1.relu,layer2.1.relu,layer3.1.relu,layer4.1.relu,fc").split(",")

OUTDIR_LOCAL = env_get("OUTDIR_LOCAL", "/workspace/out")
UPLOAD_RESULTS = env_get("UPLOAD_RESULTS", "1") == "1"

# -------------- MinIO client ---------------
def s3():
    if not ENDPOINT or not AK or not SK:
        raise SystemExit("Missing MINIO_* env vars")
    ep = ENDPOINT if ENDPOINT.startswith("http") else "http://" + ENDPOINT
    return boto3.client(
        "s3",
        endpoint_url=ep,
        aws_access_key_id=AK,
        aws_secret_access_key=SK,
    )

# -------------- Helpers --------------------
def list_trials(s3c) -> List[str]:
    trials = set()
    base = f"{PREFIX}/"
    resp = s3c.list_objects_v2(Bucket=BUCKET, Prefix=base)
    while True:
        for obj in resp.get("Contents", []):
            parts = obj["Key"].split("/")
            # exp1/<trial>/activations/epoch_E/<layer>.npz
            if len(parts) >= 4 and parts[2] == "activations":
                trials.add(parts[1])
        if resp.get("IsTruncated"):
            resp = s3c.list_objects_v2(Bucket=BUCKET, Prefix=base,
                                       ContinuationToken=resp["NextContinuationToken"])
        else:
            break
    return sorted(trials)

def fetch_npz_array(s3c, key: str, arr_key: str = "activations"):
    obj = s3c.get_object(Bucket=BUCKET, Key=key)
    data = obj["Body"].read()
    with np.load(io.BytesIO(data)) as npz:
        return npz[arr_key]

def fetch_acts(s3c, trial: str, epoch: int, layer: str):
    key = f"{PREFIX}/{trial}/activations/epoch_{epoch}/{layer}.npz"
    try:
        A = fetch_npz_array(s3c, key, "activations")
    except Exception:
        return None
    # zero-mean features
    A = A - A.mean(axis=0, keepdims=True)
    return A

def fetch_best_acc(s3c, trial: str):
    if torch is None:
        return None, None
    key = f"{PREFIX}/{trial}/checkpoints/best.pt"
    try:
        obj = s3c.get_object(Bucket=BUCKET, Key=key)
        buf = obj["Body"].read()
        state = torch.load(io.BytesIO(buf), map_location="cpu")
        val_acc = float(state.get("val_acc")) if isinstance(state, dict) and "val_acc" in state else None
        epoch = int(state.get("epoch")) if isinstance(state, dict) and "epoch" in state else None
        return val_acc, epoch
    except Exception:
        return None, None

# -------------- CKA ------------------------
def center_gram(X):
    G = X @ X.T
    n = G.shape[0]
    H = np.eye(n) - np.ones((n, n))/n
    return H @ G @ H

def cka_linear(X, Y):
    n = min(len(X), len(Y))
    if len(X) != len(Y):
        X = X[:n]; Y = Y[:n]
    Kx = center_gram(X)
    Ky = center_gram(Y)
    hsic = np.sum(Kx * Ky)
    denom = np.sqrt(np.sum(Kx*Kx) * np.sum(Ky*Ky)) + 1e-12
    return float(hsic / denom)

# -------------- SVCCA ----------------------
def svd_keep(X, var_keep=0.99):
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    var = S**2
    cum = np.cumsum(var) / (var.sum() + 1e-12)
    k = int(np.searchsorted(cum, 0.99) + 1)
    Xr = U[:, :k] * S[:k]
    return Xr

def invsqrt(mat):
    eigvals, eigvecs = np.linalg.eigh(mat)
    eigvals = np.clip(eigvals, 1e-12, None)
    Dm12 = np.diag(1.0/np.sqrt(eigvals))
    return eigvecs @ Dm12 @ eigvecs.T

def svcca_score(X, Y, var_keep=0.99):
    n = min(len(X), len(Y))
    if len(X) != len(Y):
        X = X[:n]; Y = Y[:n]
    Xr = svd_keep(X, var_keep)
    Yr = svd_keep(Y, var_keep)
    Cxx = Xr.T @ Xr
    Cyy = Yr.T @ Yr
    Cxy = Xr.T @ Yr
    Cxx_invh = invsqrt(Cxx)
    Cyy_invh = invsqrt(Cyy)
    T = Cxx_invh @ Cxy @ Cyy_invh
    s = np.linalg.svd(T, compute_uv=False)
    return float(np.mean(s))

# -------------- Writers --------------------
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def write_csv(path, header, mat):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["trial"] + header)
        for i, row in enumerate(mat):
            w.writerow([header[i]] + [f"{v:.6f}" for v in row])

def upload_file(s3c, local_path, s3_key, content_type="text/csv"):
    with open(local_path, "rb") as fh:
        s3c.put_object(Bucket=BUCKET, Key=s3_key, Body=fh, ContentType=content_type)

def object_exists(s3c, bucket, key):
    try:
        s3c.head_object(Bucket=bucket, Key=key)
        return True
    except s3c.exceptions.ClientError as e:
        if e.response['Error']['Code'] == '404':
            return False
        raise
    
# -------------- Main -----------------------
def main():
    out_root = OUTDIR_LOCAL
    ensure_dir(out_root)
    s3c = s3()

    trials = list_trials(s3c)
    print(f"Found {len(trials)} trials under {PREFIX}")
    if len(trials) < 2:
        print("Not enough trials for pairwise comparisons.")
        return

    # Trial-level summary (best acc, best epoch if available)
    summary_rows = []
    for t in trials:
        best_acc, best_epoch = fetch_best_acc(s3c, t)
        summary_rows.append({"trial": t, "best_val_accuracy": best_acc, "best_epoch": best_epoch})
    # write summary
    summary_csv = os.path.join(out_root, "trial_summary.csv")
    with open(summary_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["trial","best_val_accuracy","best_epoch"])
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)
    print(f"Wrote {summary_csv}")

    # Long-form table for all pairwise scores
    long_csv = os.path.join(out_root, "pairwise_scores.csv")
    with open(long_csv, "w", newline="") as f_long:
        wlong = csv.writer(f_long)
        wlong.writerow(["epoch","layer","metric","trial_i","trial_j","score"])

        for epoch in EPOCHS:
            for layer in LAYERS:
                # load all activations for this (epoch, layer)
                acts = {}
                for t in trials:
                    A = fetch_acts(s3c, t, epoch, layer)
                    if A is not None and A.ndim == 2 and A.shape[0] >= 2:
                        acts[t] = A
                kept = sorted(acts.keys())
                n = len(kept)
                if n < 2:
                    print(f"[epoch={epoch} layer={layer}] insufficient trials with activations")
                    continue

                # CKA matrix
                M_cka = np.zeros((n, n), dtype=np.float64)
                # SVCCA matrix
                M_sv  = np.zeros((n, n), dtype=np.float64)

                for i, j in itertools.product(range(n), range(n)):
                    if i == j:
                        M_cka[i, j] = 1.0
                        M_sv[i, j]  = 1.0
                    elif i < j:
                        c = cka_linear(acts[kept[i]], acts[kept[j]])
                        s = svcca_score(acts[kept[i]], acts[kept[j]], var_keep=0.99)
                        M_cka[i, j] = M_cka[j, i] = c
                        M_sv[i, j]  = M_sv[j, i]  = s
                        # long rows (both metrics)
                        wlong.writerow([epoch, layer, "CKA", kept[i], kept[j], f"{c:.6f}"])
                        wlong.writerow([epoch, layer, "SVCCA", kept[i], kept[j], f"{s:.6f}"])

                # write square CSVs too
                mat_dir = os.path.join(out_root, "analysis")
                ensure_dir(mat_dir)
                cka_csv = os.path.join(mat_dir, f"cka_epoch{epoch}_{layer.replace('.','_')}.csv")
                sv_csv  = os.path.join(mat_dir, f"svcca_epoch{epoch}_{layer.replace('.','_')}.csv")
                write_csv(cka_csv, kept, M_cka)
                write_csv(sv_csv, kept, M_sv)
                print(f"Wrote {cka_csv} and {sv_csv}")

                if UPLOAD_RESULTS:
                    s3_key_cka = f"{PREFIX}/analysis/{Path(cka_csv).name}"
                    s3_key_sv  = f"{PREFIX}/analysis/{Path(sv_csv).name}"
                    # upload_file(s3c, cka_csv, s3_key_cka)
                    if not object_exists(s3c, BUCKET, s3_key_cka):
                        upload_file(s3c, cka_csv, s3_key_cka)
                    else:
                        print(f"Skipping upload (already exists): {s3_key_cka}")
                    upload_file(s3c, sv_csv,  s3_key_sv)

        # upload the long csv and summary too
        if UPLOAD_RESULTS:
            upload_file(s3c, long_csv,   f"{PREFIX}/analysis/pairwise_scores.csv")
            upload_file(s3c, summary_csv,f"{PREFIX}/analysis/trial_summary.csv")

    print(f"Wrote {long_csv}")
    print("Done.")

if __name__ == "__main__":
    main()