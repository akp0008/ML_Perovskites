# Data-driven Design of Mixed Halide Perovskites

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

Complete computational workflow analyzing **1,044 halide perovskites** with **CHGNet-computed defect formation energies** (DFE). Integrates **EDA**, **unsupervised clustering**, **CatBoost prediction**, and **inferential statistics** (ANOVA + Tukey HSD).

**Goal:** Identify stability-bandgap trade-offs for tandem solar cell design (1.5 eV target).

---

## 📂 Repository Structure

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

---

## 🧪 Notebooks Overview

### 1. Exploratory Data Analysis
Located in: `notebooks/EDA/`
* **Distribution Analysis:** Visualizing the spread of defect energies and bandgaps.
* **Correlation:** Spearman/MI/dCor/MIC heatmaps to identify relation between features.
* **Diensionality Reduction and Clustering:** Unsupervised learning (t-SNE Dimensionality Reduction + Bayesian Gaussian Mixture Model clustering) to group compounds based on physiochemical descriptors.

### 2. Machine Learning Models
Located in: `notebooks/Models/`
Each notebook trains a gradient-boosted decision tree (CatBoost) for a specific target property:
* **Inputs:** Compositional vectors and elemental properties.
* **Targets:** Bandgap ($E_g$) and Defect Formation Energies ($E_A, E_B, E_X$).
* **Workflow:** Preprocessing -> Hyperparameter Tuning (Optuna) -> Validation -> SHAP Analysis.

### 3. Statistical Analysis
Located in: `notebooks/Statistics/`
* **Inferential Statistics:** One-way ANOVA to test the significance of site-specific substitutions.
* **Tukey HSD:** Pairwise comparisons exported to `results/tukey/` to identify distinct stability clusters (e.g., comparing MA-based vs. FA-based stability).

---

## 📊 Dataset

The project relies on:
* **`data/dataset.csv`**: A structured table containing stoichiometry, physiochemical features (radii, electronegativity), and target variables derived from DFT/CHGNet calculations.


## 📦 Installation

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

## ▶️ Usage

1. **Launch Jupyter Lab:**
   ```bash
   jupyter lab
   ```
2. **Navigate** to the `notebooks/` directory to run the specific analysis.
3. **Note on Paths:** Ensure relative paths in the notebooks point to `../../data/dataset.csv` so the data loads correctly.

---

## 📜 License

MIT License.

## 🤝 Contributing

Pull requests and suggestions are welcome.
