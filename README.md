# EV Supply Chain Volatility Detector using Spatio-Temporal Graph Attention Networks

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A deep learning system that uses **Graph Attention Networks (GAT)** and **LSTM** to predict volatility spillovers across the Electric Vehicle supply chain by modeling real supplier relationships extracted from SEC filings.

## 🎯 Project Overview

Traditional risk models (GARCH, DCC-GARCH) rely on price correlations that break during structural shocks. This project models **real supply chain dependencies** and uses **attention mechanisms** to learn which relationships matter most at any given time.

**Key Innovation:** We predict downstream volatility by identifying risk transmission pathways days in advance (e.g., Albemarle lithium shortage → Panasonic → Tesla).

## 🏗️ Architecture

```
Input: Knowledge Graph (25 nodes, ~60 edges) + 20-day time series
    ↓
Multi-Head GAT Layer 1 (8 heads, hidden_dim=64)
    ↓
Multi-Head GAT Layer 2 (8 heads, hidden_dim=64)
    ↓
LSTM (hidden_dim=128)
    ↓
Linear Prediction Head
    ↓
Output: Next-day volatility for each node
```

### Supply Chain Structure
- **Tier 0 (OEMs):** Tesla, Ford, GM, Rivian, Lucid, Nio
- **Tier 1 (Battery):** Panasonic, LG Energy, CATL, Samsung SDI, BYD
- **Tier 2 (Components):** Magna, Aptiv, BorgWarner
- **Tier 3 (Raw Materials):** Albemarle, SQM, Livent, Lithium Americas, etc.

## 📊 Features (12 per node)

**Stock-Specific:**
- Log returns
- Realized volatility (20-day)
- GARCH(1,1) conditional volatility
- Bid-ask spread
- Volume shock
- RSI

**Macro (shared):**
- VIX index
- 10-Year Treasury yield
- Lithium carbonate price
- Copper price
- Industrial Production Index
- Dollar Index

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/ev-supply-chain-gat.git
cd ev-supply-chain-gat

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model (for NLP)
python -m spacy download en_core_web_sm
```

### Data Collection (Phase 1)

```bash
# Download SEC filings and market data
python scripts/download_data.py --start-date 2018-01-01 --end-date 2024-06-30

# Build knowledge graph
python scripts/build_graph.py
```

### Training (Phase 4)

```bash
# Train ST-GAT model
python scripts/train.py --config configs/default.yaml

# Hyperparameter search
python scripts/hyperparameter_search.py
```

### Evaluation (Phase 5)

```bash
# Evaluate on test set
python scripts/evaluate.py --checkpoint results/checkpoints/best_model.pt
```

## 📁 Project Structure

```
ev-supply-chain-gat/
├── data/
│   ├── raw/              # SEC filings, market data
│   ├── processed/        # Cleaned features, graphs
│   └── graphs/           # Knowledge graph snapshots
├── models/               # GAT + LSTM model definitions
├── baselines/            # GARCH, DCC-GARCH, VAR baselines
├── utils/                # Data loaders, graph builders, feature engineering
├── notebooks/            # Exploratory analysis
├── scripts/              # Training, evaluation, data download scripts
├── tests/                # Unit and integration tests
├── results/              # Figures, tables, model checkpoints
└── docs/                 # Architecture diagrams, API docs
```

## 🎓 Hypotheses Being Tested

1. **H1: Attention Entropy as Early Warning**
   - Low attention entropy at t-10 predicts high VIX at time t
   - Target: AUC > 0.70

2. **H2: Attention Pathway Correspondence**
   - Attention spikes on correct edges 3-10 days before downstream volatility
   - Validation: 2021 lithium surge, 2020-2022 chip shortage

3. **H3: Superior Predictive Performance**
   - ST-GAT outperforms DCC-GARCH by >15% RMSE
   - Statistical test: Diebold-Mariano (p < 0.05)

## 📈 Baseline Models

- Persistence Model (naive)
- GARCH(1,1) (univariate)
- DCC-GARCH (multivariate)
- VAR (Vector Autoregression)
- LSTM-only (no graph - ablation)
- GCN (static graph - ablation)

## 🔬 Current Status

- [x] **Phase 0:** Architecture & Design Complete
- [ ] **Phase 1:** Data Collection & Knowledge Graph (In Progress)
- [ ] **Phase 2:** Feature Engineering
- [ ] **Phase 3:** Model Design
- [ ] **Phase 4:** Implementation & Training
- [ ] **Phase 5:** Evaluation
- [ ] **Phase 6:** Visualization & Interpretability
- [ ] **Phase 7:** Documentation

## 📚 Key References

1. Veličković, P., et al. (2018). "Graph Attention Networks." ICLR.
2. Engle, R. (2002). "Dynamic Conditional Correlation." Journal of Business & Economic Statistics.
3. Diebold, F. X., & Yilmaz, K. (2014). "On the Network Topology of Variance Decompositions." Journal of Econometrics.
4. Acemoglu, D., et al. (2012). "The Network Origins of Aggregate Fluctuations." Econometrica.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

This is a research project. For major changes, please open an issue first to discuss what you would like to change.

## 📧 Contact

For questions or collaborations, please open an issue in this repository.

---

**Last Updated:** December 22, 2025  
**Current Phase:** Phase 1 - Data Collection
