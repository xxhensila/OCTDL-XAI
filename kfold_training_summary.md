# OCTDL k-Fold Cross-Validation Integration Summary

This document summarizes the full process of using **patient-wise k-Fold cross-validation**. 
The goal is to robustly evaluate model generalization across patients.

---

## Original Setup (from GitHub)

* Dataset split: hard-coded split-% on train,validation,test sets (on **patient-level**) in `data/builder.py`.
* `main.py` → `train.py` uses these datasets once.
* Model configuration handled through `configs/OCTDL.yaml`.
* Estimator and logging done using custom `Estimator` class and TensorBoard.
* Evaluation done once on the held-out test set after training.

---

## Goal: Replace with Patient-wise k-Fold Cross-Validation

We aimed to:

1. Reuse the dataset (from `csv_path`) but split it on \`\` into k folds.
2. For each fold:

   * Train on (k−1) folds
   * Validate on the 1 held-out fold
   * Save model and metrics per fold

---

## Steps Performed

### 1. Created `kfold_train.py`

* Custom script using `GroupKFold` from scikit-learn.
* Loads the CSV (`cfg.data.csv_path`) and groups by `patient_id`.
* For each fold:

  * Creates new `train_df` and `val_df`
  * Constructs `OCTDataset` using these dataframes
  * Builds model using `generate_model(cfg)`
  * Logs metrics and saves model in `outputs/fold_{i}`

### 2. Modified YAML Config (`OCTDL.yaml`)

* Ensured `csv_path` and `image_root` are explicitly defined under `cfg.data`
* Example:

```yaml
base:
  save_path: outputs
  overwrite: True

train:
  batch_size: 32
  num_epochs: 100
  num_workers: 4
  pin_memory: True
  criterion: cross_entropy
  metrics: [acc, f1]

data:
  csv_path: OCTDL_dataset/metadata.csv
  image_root: OCTDL_dataset/
  num_classes: 4
  classes: [CNV, DME, DRUSEN, NORMAL]
```

### 3. Defined Custom Dataset

* Created `OCTDataset(Dataset)` that:

  * Takes a DataFrame and loads images from `image_root/{disease}/{filename}.jpg`
  * Applies transform if given
  * Returns `(image_tensor, label)`

### 4. Adjusted Train Loop

* Reused `train()` and `evaluate()` logic
* Estimator obtained using `get_estimator(cfg)`
* Logger per fold with `get_logger(cfg, fold=i)`

---

## Status

* Successfully implemented and debugged all errors.
* Final training now runs across `k=5` folds with patient-wise separation.
* Results stored cleanly and reproducibly per fold.

