# 🧬 Multi-Modal Molecular Property Stacking Pipeline

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Geometric-EE4C2C?logo=pytorch&logoColor=white)
![RDKit](https://img.shields.io/badge/RDKit-2026.3.1-3B82F6)
![XGBoost](https://img.shields.io/badge/XGBoost-Balanced-yellow)
![Kaggle](https://img.shields.io/badge/Kaggle-Competition-20BEFF?logo=kaggle&logoColor=white)

> **OpenAImer 2026 GenAI Hackathon (Track A) — National Finalist (Top 15)** > 
> Developed by Data AImers

### [🏆 Leaderboard Performance](https://www.kaggle.com/competitions/openaimer26-1/leaderboard)
* **Private Leaderboard Score:** `0.6689`
* **Public Leaderboard Score:** `0.6974`

---

## 📌 Project Objective
To develop and train a highly robust Graph Neural Network (GNN) and Tree-based stacking ensemble for accurate multi-label binary property classification (5 properties) of molecular structures. Standard Graph Neural Networks often fail to capture global molecular properties (like weight or solubility), while traditional Tree models struggle to understand 2D structural topology. This pipeline resolves that limitation by separating local bond extraction from global property analysis, merging them via a rank-based meta-model.

---

## 🏗️ System Architecture

Our solution utilizes a two-level Multi-Modal Stacking Pipeline:

### Level 1: Deep Feature Extraction & Base Engines
1. **Local Graph Engines (Topology):** * **Models:** Graph Isomorphism Network (GIN) & Graph Attention Network (GATv2).
   * **Mechanism:** Message-passing networks that learn local bond structures and reactive zones utilizing 9 node features and 6 edge features.
   * **Innovation:** Integrates **Jumping Knowledge** to concatenate local atom representations across all message-passing layers, preventing deep GNN over-smoothing.
2. **Global Tabular Engines (Macro-Chemical):**
   * **Models:** Balanced XGBoost, Random Forest, and a Deep Multi-Layer Perceptron (MLP)
   * **Mechanism:** Processes dense, 2223-dimensional global chemical fingerprints (Morgan Fingerprints, MACCS Keys, explicit RDKit Descriptors).

### Level 2: The Meta-Orchestrator
* **Model:** Logistic Regression (Class-Balanced).
* **Mechanism:** Synthesizes the 5 Out-Of-Fold (OOF) Level-1 model predictions into a final multi-label consensus. 
* **Innovation:** Applies a **Percentile Rank Transformation** to bypass probability calibration clashes between GNNs and Tree models, immunizing the ensemble against leaderboard distribution shifts.

---

## 🛠️ Core Engineering Solutions (Edge Case Handling)

Working with a highly imbalanced, small dataset (2,307 molecules) required stringent architectural defenses to prevent runtime crashes and overfitting.

| Problem | System Component | Technical Outcome |
| :--- | :--- | :--- |
| **Invalid/Broken SMILES Strings** | Dummy Atom Fallback | Silently replaces broken dataset molecules with a single Carbon atom (`Chem.MolFromSmiles('C')`), ensuring tensor dimensions remain stable and preventing PyTorch DataLoader crashes. |
| **Extreme Target Imbalance** | Focal Loss ($\gamma = 2.0$) | Dynamically scales the penalty for missing rare active molecules, forcing the neural networks to prioritize the minority class and maximize the MCC metric. |
| **BatchNorm Crashing on Small Batches** | `drop_last=True` (DataLoader) | Prevents fatal PyTorch errors when the final cross-validation batch contains only 1 molecule, which mathematically breaks standard deviation calculations in 1D Batch Normalization. |
| **Optimizer False Negatives** | Clamped Threshold Search Space | Prevents the dynamic optimizer from aggressively chasing perfect precision (e.g., threshold > 0.85), which would tank the MCC score through excessive False Negatives. |
| **Data Leakage in CV** | 8-Fold Murcko Scaffold Split | Splits the dataset based on core 2D ring structures rather than random sampling, forcing models to generalize to unseen chemical backbones and perfectly simulating the Kaggle Private Leaderboard. |

---
# 🚀 Reproducibility & Execution

## 1. Clone and Install Dependencies

```bash
git clone 
cd molecular-stacking
pip install -r requirements.txt
```
## 2. Data Setup (Kaggle Integration)

This repository strictly separates **code** from **data**.
1. Download the competition data from the **OpenAImer 2026 Kaggle Competition Page**.
2. Place `train.csv` and `test.csv` directly into the `data/raw/` directory.

```text
data/raw/
├── train.csv
└── test.csv

```
## 3. Run the Training Pipeline

```bash
python train_pipeline.py
```
This script will:-
- Generate **8-fold cross-validation** features
- Train all **Level-1 base models**
- Train the **Level-2 Meta-Orchestrator**
- Save the optimized **MCC thresholds** to `data/processed/`

## 4. Generate Predictions

```bash
python inference.py
```
This will:
- Ingest `test.csv`
- Apply the trained ensemble pipeline
- Output the final `submission.csv` to the project root directory

---

## 🔗 Links & Resources
- **Kaggle Competition:** [OpenAImer 2026 Track A](https://www.kaggle.com/competitions/openaimer26-1/overview)
---

## 📂 Repository Structure

```text
molecular-stacking/
├── data/
│   ├── raw/                      # Download Kaggle CSVs here
│   └── processed/                # Serialized scalers, thresholds, and intermediate matrices
├── src/
│   ├── data/
│   │   ├── feature_extraction.py # RDKit Morgan/MACCS/Descriptor extraction
│   │   └── dataset.py            # PyTorch Geometric MoleculeDataset
│   ├── models/
│   │   ├── gnn_engines.py        # GIN & GATv2 architectures with Jumping Knowledge
│   │   └── tabular_engines.py    # MultiOutput wrapper for XGB, RF, MLP
│   ├── training/
│   │   ├── loss.py               # Imbalanced Focal Loss function
│   │   ├── splitters.py          # Murcko Scaffold CV Splitter
│   │   └── train_level1.py       # Pytorch Geometric training loops
│   └── utils/
│       ├── metrics.py            # Clamped MCC threshold optimization
│       └── rank_transforms.py    # Percentile rank calibration
├── notebooks/
│   └── EDA_and_Prototyping.ipynb # Original Hackathon notebook
├── train_pipeline.py             # Executes training and feature extraction
├── inference.py                  # Generates final Kaggle submission
├── requirements.txt
└── README.md
