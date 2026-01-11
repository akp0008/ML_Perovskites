# Data-driven Design of Mixed Halide Perovskites

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

End-to-end Materials Informatics framework for the high-throughput analysis of **1,044 halide perovskites** utilizing **CHGNet-derived defect formation energies (DFEs)**. Integrates **unsupervised clustering (BGMM)**, **explainable machine learning (CatBoost + SHAP)**, and **inferential statistics (ANOVA + Tukey HSD)** to navigate complex stability-bandgap trade-offs. The modular architecture serves as the high-throughput surrogate engine for an autonomous Multi-Objective Genetic Algorithm (MOGA) workflow, enabling the exploration of $\sim 10^{12}$ mixed compositions via the DAKOTA toolkit.

---

## Repository Structure

The project is organized into modular directories for data, analysis, and results:

```text
.
├── data/
│   └── dataset.csv                  # Master dataset of material features and targets
│
├── notebooks/
│   ├── EDA/
│   │   ├── correlations.ipynb       # Feature correlation heatmaps & analysis
│   │   └── clustering.ipynb         # t-SNE dimensionality reduction and Bayesian Gaussian Mixture Model clustering of compositions
│   │
│   ├── Models/                      # CatBoost Regressors for each target
│   │   ├── A-DFE.ipynb
│   │   ├── B-DFE.ipynb
│   │   ├── X-DFE.ipynb
│   │   └── Bandgap.ipynb
│   │
│   └── Statistics/
│       └── inferential.ipynb        # One-way ANOVA & Tukey HSD tests
├── src/                             # Production-grade modular source code
│   ├── __init__.py                  # Package initializer
│   └── perovskite_predictor.py      # Unified ML pipeline (CatBoost + Optuna + SHAP)
│
├── results/
│   └── tukey/                       # Statistical post-hoc analysis tables
│       ├── tukey_results_A_A-DFE.csv
│       ├── tukey_results_A_B-DFE.csv
│       └── ...
│
├── requirements.txt                 # Python dependencies
└── README.md                        # Project documentation
```

## Modular ML Pipeline

**Location:** `src/`

The core predictive logic is implemented as a **Modular Machine Learning Pipeline** designed for high-throughput materials discovery and autonomous property optimization. The architecture emphasizes scalability, interpretability, and automated diagnostics.

### Core Features

- **Automated Tuning**
  - Integrated **Optuna** engine for automated hyperparameter optimization.
  - Enables efficient exploration of model configurations with minimal manual intervention.

- **Explainability (XAI)**
  - Built-in **SHAP** and **feature importance** modules.
  - Supports extraction and analysis of physically meaningful descriptors.

- **Diagnostics & Monitoring**
  - Automated generation of **learning curves** to assess bias–variance trade-offs.
  - **Parity plots** for evaluating predictive accuracy and systematic errors.

This modular design allows seamless extension to new models, descriptors, and optimization strategies while maintaining reproducibility and transparency.

---

## Inverse Design Workflow (MOGA Integration)

```text
+---------------------+           (1) Generates          +----------------------+
|    DAKOTA / MOGA    | -------------------------------> |  Composition Vector  |
|  Optimization Engine|                                  |  (Design Variables)  |
+---------------------+                                  +----------+-----------+
          ^                                                         |
          |                                                         | (2) Input
          | (4) Feedback / Fitness                                  v
          |                                              +----------------------+
+---------+-----------+                                  |   CatBoost Models    |
| Objective Functions | <---------(3) Predicts---------  |  (Surrogate Engine)  |
| (Stability, Bandgap)|                                  +----------------------+
+---------------------+
```

This repository hosts the **predictive backend** for the inverse design campaign described in *Pandey et al. (2025)*.

To navigate the combinatorial space of $\sim 10^{12}$ mixed-halide compositions, this project integrates with the **DAKOTA** optimization toolkit:
*  **Surrogate Modeling:** The `CatBoost` models in `src/` act as rapid evaluators for thermodynamic stability and bandgap.
*  **Autonomous Optimization:** These surrogates are coupled with a Multi-Objective Genetic Algorithm (MOGA) to drive an evolutionary search towards the Pareto front.

---

## Notebooks Overview

### 1. Exploratory Data Analysis
Located in: `notebooks/EDA/`
* **Distribution Analysis:** Visualizing the spread of defect energies and bandgaps.
* **Correlation:** Spearman/MI/dCor/MIC heatmaps to identify relation between features.
* **Dimensionality Reduction and Clustering:** Unsupervised learning (t-SNE Dimensionality Reduction + Bayesian Gaussian Mixture Model clustering) to group compounds based on physiochemical descriptors.

### 2. Statistical Analysis
Located in: `notebooks/Statistics/`
* **Inferential Statistics:** One-way ANOVA to test the significance of site-specific substitutions.
* **Tukey HSD:** Pairwise comparisons exported to `results/tukey/` to identify distinct stability clusters (e.g., comparing MA-based vs. FA-based stability).

### 3. Machine Learning Models
Located in: `notebooks/Models/`
Each notebook trains a gradient-boosted decision tree (CatBoost) for a specific target property:
* **Inputs:** Compositional vectors and elemental properties.
* **Targets:** Bandgap ($E_g$) and Defect Formation Energies ($E_A, E_B, E_X$).
* **Workflow:** Preprocessing -> Hyperparameter Tuning (Optuna) -> Validation -> SHAP Analysis.


## Dataset

The project relies on:
* **`data/dataset.csv`**: A structured table containing stoichiometry, physiochemical features (radii, density, tolerance factor, etc.), and target variables derived from DFT/CHGNet calculations.


## Installation

To reproduce the environment, it is recommended to use a clean virtual environment (conda or venv).

### Option 1: Using pip
```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Using Conda
```bash
conda create -n material-ml python=3.10
conda activate material-ml
pip install -r requirements.txt
```

---

## Usage

1. **Launch Jupyter Lab:**
   ```bash
   jupyter lab
   ```
2. **Navigate** to the `notebooks/` directory to run the specific analysis.
3. **Note on Paths:** Ensure relative paths in the notebooks point to `../../data/dataset.csv` so the data loads correctly.

---

## Citation

If you use this code or dataset in your research, please cite the following paper:
```text
Pandey, A. K., Pandey, V., and Tewari, A. (2025).  
Machine learning aided bandgap and defect engineering of mixed halide perovskites for photovoltaic applications. 
Materials Today Physics, Article 102003.  
https://doi.org/10.1016/j.mtphys.2025.102003
```
### BibTeX

```text
@article{pandey2025machine,
  title={Machine learning aided bandgap and defect engineering of mixed halide perovskites for photovoltaic applications},
  author={Pandey, Ayush Kumar and Pandey, Vivek and Tewari, Abhishek},
  journal={Materials Today Physics},
  pages={102003},
  year={2025},
  publisher={Elsevier},
  doi={10.1016/j.mtphys.2025.102003}
}
```
## License

MIT License.

## Contributing

Pull requests and suggestions are welcome.
