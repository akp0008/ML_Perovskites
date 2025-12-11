# Material Property Modeling: EDA, Clustering & CatBoost Prediction

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

This repository hosts a robust computational framework for the accelerated discovery of mixed halide perovskites (ABX3​) tailored for photovoltaic applications. Leveraging a synergistic approach of gradient-boosted machine learning (CatBoost), unsupervised clustering (t-SNE/BGMM), and rigorous inferential statistics, the workflow navigates the complex compositional space to identify candidates with optimal bandgaps and enhanced thermodynamic stability against defect formation.

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
│   │   └── clustering.ipynb         # t-SNE/BGMM clustering of compositions
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
* **Correlation Analysis:** Pearson and Spearman rank correlation heatmaps to diagnose multicollinearity and identify key structure-property relationships.
*  **Inferential Statistics:** One-way ANOVA and Tukey’s HSD post-hoc analysis to rigorously quantify the statistical significance of site-specific substitutions on thermodynamic stability.
* **Dimensionality Reduction and Clustering:** Unsupervised learning utilizing t-SNE for projection and Bayesian Gaussian Mixture Models (BGMM) to group compounds based on physiochemical descriptors.

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
* **`data/dataset.csv`**: A structured table containing stoichiometry, physiochemical features (ionic radii, tolerance factor, etc.), and target variables derived from DFT/CHGNet calculations.

---

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
