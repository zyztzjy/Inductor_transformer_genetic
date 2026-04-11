# Inductor Design and Optimization with Transformer Surrogate Model

This repository provides an end‑to‑end framework for on‑chip spiral inductor design and optimization. The workflow consists of:

1. Layout generation – Create inductor geometries in Keysight ADS.
2. Simulation & dataset creation – Obtain S‑parameters via EM simulation (user‑performed).
3. Transformer surrogate modeling – Train a deep learning model that maps geometric parameters to S‑parameters.
4. Robustness analysis – Evaluate model stability under input perturbations.
5. Multi‑objective optimization – Use the trained Transformer as a fast evaluator inside algorithm to optimize bandwidth, Q factor, and insertion loss.

---

Repository Files
----------------

| File                         | Role in Workflow                                                                                     |
|------------------------------|------------------------------------------------------------------------------------------------------|
| ADS_inductor_layout.py       | Step 1 – Generate spiral inductor layouts in ADS. Outputs a design with multiple inductor instances. |
| datatrain_transformer.py     | Step 3 – Train the Transformer model on the simulated dataset. Saves best_inductor_transformer.pth.  |
| noise.py                     | Step 4 – Robustness test. Measures R² degradation under Gaussian input noise.                        |
| optimizer_with_Trans.py      | Step 5 – Multi‑objective optimization using NSGA‑II. Produces a Pareto front of optimal designs.     |

---

Full Workflow
-------------

Step 1: Generate Inductor Layouts (ADS)
---------------------------------------
Run ADS_inductor_layout.py inside a Keysight ADS environment.

    python ADS_inductor_layout.py

By default, the script generates 9 inductor pairs with randomly varied geometric parameters:

| Parameter      | Description                         | Range      |
|----------------|-------------------------------------|------------|
| turns_top      | Number of turns (top layer)         | 1 – 5      |
| turns_bot      | Number of turns (bottom layer)      | 1 – 5      |
| linewidth_top  | Conductor width, top layer (µm)     | 3 – 10     |
| linewidth_bot  | Conductor width, bottom layer (µm)  | 3 – 10     |
| center_gap     | Spacing between turns (µm)          | 40 – 120   |
| inner_diam     | Inner diameter of the spiral (µm)   | 20 – 100   |

Configuration variables in ADS_inductor_layout.py:

| Variable         | Description                                                  |
|------------------|--------------------------------------------------------------|
| workspace_path   | Path to ADS workspace (e.g., C:\Users\...\MyWorkspace_wrk)   |
| library_path     | Path to ADS library (e.g., C:\Users\...\MyWorkspace_wrk\MyLibrary_lib) |
| num_pairs        | Number of inductor pairs to generate                          |
| pair_spacing_x   | Horizontal spacing between inductors (µm)                     |
| pair_spacing_y   | Vertical spacing between inductors (µm)                       |

Note: You must have Keysight ADS installed with the keysight.ads.de Python API accessible. Layer names (cond, cond2, via, etc.) should match your PDK definitions.

Output: An ADS layout design saved in the specified library, ready for EM simulation.


Step 2: EM Simulation & Dataset Preparation (User‑performed)
------------------------------------------------------------
The generated layout must be simulated using ADS Momentum or another EM solver to obtain S‑parameters over frequency.

Data Format Requirements:
-------------------------
After simulation, export the data into the following structure:

1. merged_data.txt – Geometric parameters for each inductor.
   Supported formats:
   - turns_top=2, turns_bot=1, linewidth_top=5.2, linewidth_bot=5.8, center_gap=68, inner_diam=39
   - N_t=2, N_b=1, W_t=5.2, W_b=5.8, G_c=68, D_i=39
   - 2, 1, 5.2, 5.8, 68, 39
   - 
2. s_parameters.npz – NumPy archive
 https://pan.baidu.com/s/16bQFLr7LbXajGpKOWINTZQ code: e8s2




Step 3: Train the Transformer Surrogate Model
---------------------------------------------
Run datatrain_transformer.py to train the model.

    python datatrain_transformer.py

Outputs:
- best_inductor_transformer.pth : Trained model state, feature/target scalers, and metadata.
- training_results.png : Training/validation loss curves, per‑dimension R² scores, and true‑vs‑predicted scatter plot.

Step 4: Evaluate Model Robustness
---------------------------------
Run noise.py to test the model's sensitivity to input noise.

    python noise.py

This script:
- Loads the trained model from best_inductor_transformer.pth.
- Splits the dataset to create a test set.
- Adds Gaussian noise to the test set features at levels: 0%, 1%, 3%, 5%, 7%, 10%, 12%, 15%.
- Computes average R² score for each noise level.
- Plots R² degradation versus noise level.

Output:
- noise_robustness_transformer.png : Plot showing performance degradation under increasing noise.


Step 5: Multi‑Objective Optimization
----------------------------------------------
Run optimizer_with_Trans.py to perform optimization.

    python optimizer_with_Trans.py

This script:
- Uses the trained Transformer model as a fast surrogate evaluator.
- Optimizes three objectives simultaneously:
  - Maximize Bandwidth (GHz)
  - Maximize Q Factor
  - Minimize Insertion Loss (dB)
- Population size: 100, Generations: 200.

Outputs:
- convergence_history.csv : Per‑generation best and average fitness values.
- pareto_optimal_solutions.csv : Table of Pareto‑optimal designs with their performance metrics.
- convergence_curves.png : Evolution plots for bandwidth, Q factor, insertion loss, and the final Pareto front.

Dependencies
------------
- Python 3.8+
- numpy
- pandas
- matplotlib
- scikit‑learn
- scipy
- torch (PyTorch)
- deap (for genetic algorithm)
- Keysight ADS (for layout generation) with Python API

Install Python packages:

    pip install numpy pandas torch scikit-learn matplotlib scipy deap


Notes
-----
- The Transformer model expects input features in the order: [N_t, N_b, W_t, W_b, G_c, D_i] (all in µm except turns).
- The number of frequency points is inferred from the dataset; ensure consistency between training and inference.
- For the optimizer, parameter bounds and objectives can be customized in optimizer_with_Trans.py (BOUNDS, POP_SIZE, NGEN, etc.).


Contact
-------
For questions or issues, please open an issue in the repository.
