# Inductor_transformer_genetic
# Inductor Design and Optimization with Transformer Surrogate Model

This repository provides an end‑to‑end framework for on‑chip spiral inductor design and optimization. The workflow consists of:

1. **Layout generation** – Create inductor geometries in Keysight ADS.
2. **Simulation & dataset creation** – Obtain S‑parameters via EM simulation (user‑performed).
3. **Transformer surrogate modeling** – Train a deep learning model that maps geometric parameters directly to full‑band S‑parameters.
4. **Robustness analysis** – Evaluate model stability under input perturbations.
5. **Multi‑objective optimization** – Use the trained Transformer as a fast evaluator inside an NSGA‑II genetic algorithm to optimize bandwidth, Q factor, and insertion loss.

---

## 📁 Repository Files

| File | Role in Workflow |
|------|------------------|
| `ADS_inductor_layout.py` | **Step 1** – Generate spiral inductor layouts in ADS. Outputs a design with multiple inductor instances having varied geometric parameters. |
| `datatrain_transformer.py` | **Step 3** – Train the Transformer model on the simulated dataset. Performs data loading, preprocessing, augmentation, cross‑validation, and final training. Saves the best model as `best_inductor_transformer.pth`. |
| `noise.py` | **Step 4** – Robustness test. Loads the trained model and measures R² degradation when Gaussian noise is added to the input features. |
| `optimizer_with_Trans.py` | **Step 5** – Multi‑objective optimization using NSGA‑II. Uses the Transformer model to quickly evaluate candidate designs and produces a Pareto front of optimal trade‑offs. |

---

## 🔁 Full Workflow

### Step 1: Generate Inductor Layouts (ADS)
Run `ADS_inductor_layout.py` inside a Keysight ADS environment to create a layout containing multiple spiral inductors. By default, it generates 9 inductor pairs with randomly varied parameters (turns, linewidth, center gap, etc.).

> **Note:** You must have ADS installed and the `keysight.ads.de` Python API accessible. Update the `workspace_path` and `library_path` variables to point to your ADS workspace/library.

### Step 2: EM Simulation & Dataset Preparation (User‑performed)
The generated layout must be simulated using ADS Momentum or another EM solver to obtain S‑parameters over frequency. After simulation, export the data into the following format:

- **`merged_data.txt`** – Contains the geometric parameters for each inductor. The script `datatrain_transformer.py` supports several common formats (e.g., `turns_top=2, turns_bot=1, linewidth_top=5.2, ...`).
- **`s_parameters.npz`** – NumPy archive with keys:
  - `freq` : 1D array of frequency points (GHz).
  - `S11_mag`, `S11_phase`, `S12_mag`, …, `S33_phase` : 2D arrays of shape `(n_samples, n_freq)`.

Place both files inside a folder named `inductor_s_params_dataset/`.

### Step 3: Train the Transformer Surrogate Model
```bash
python datatrain_transformer.py
