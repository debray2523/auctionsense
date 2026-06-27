# AuctionSense 🏭

**Agentic AI for Seller-Side Pre-Auction Intelligence in B2B Industrial Commodity Auctions**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11+-green.svg)](https://python.org)
[![Paper](https://img.shields.io/badge/Paper-Zenodo-orange.svg)](https://zenodo.org/record/XXXXXXX)
[![Demo](https://img.shields.io/badge/Demo-Streamlit-red.svg)](https://auctionsense.streamlit.app)

> *The first open-source agentic AI framework that helps auction operators set smarter reserve prices, prevent thin-bidder disasters, and configure lots for maximum price realisation.*

---

## What Problem Does This Solve?

Every industrial auction operator faces three decisions before each event — decisions currently made by gut feel alone:

| Decision | The Problem | AuctionSense Solution |
|---|---|---|
| **Reserve Price** | Too high = unsold lot. Too low = lost value. Current practice: expert judgment | XGBoost model conditioned on live commodity prices + lot attributes → MAPE 4.7% |
| **Buyer Participation** | Thin participation (< 4 bidders) causes 15–25% price collapse. No warning system exists | LightGBM participation forecaster → AUC 0.84, 80% recall for HIGH risk lots |
| **Lot Configuration** | Split? Bundle? As-is? Zero AI tools exist for this decision | RAG + LLM reasoning over historical analogues → Expert rating 4.1/5.0 |

**Published paper:** [Zenodo DOI pending] | **LinkedIn article:** [link] | **Author:** Dr. Debendra Ray, DBA (ORCID: 0009-0002-5784-4442)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    LangGraph StateGraph                          │
│                                                                  │
│  Lot Input ──► Market Intel ──► Reserve Price ──► Participation │
│                   Agent           Agent              Agent       │
│                    │                │                  │         │
│                    └────────────────┴──────────────────┘        │
│                                     │                           │
│                              Lot Config Agent                    │
│                                     │                           │
│                           Auction Strategist ──► Pre-Auction    │
│                               (Orchestrator)      Briefing      │
└─────────────────────────────────────────────────────────────────┘

Data flows:
  - Commodity API (MCX/LME via yfinance)    → Market Intel Agent
  - Historical lot store (pgvector)          → Lot Config Agent
  - Buyer affinity store (CRM / mock JSON)   → Participation Agent
  - Compliance rules (configs/rules.yaml)    → Strategist Agent
```

---

## Quickstart

### Prerequisites
- Python 3.11+
- Docker & Docker Compose
- Azure OpenAI API key (GPT-4o) or OpenAI API key
- PostgreSQL with pgvector extension (included in Docker Compose)

### 1. Clone and configure
```bash
git clone https://github.com/debray2523/auctionsense.git
cd auctionsense
cp configs/.env.example configs/.env
# Edit configs/.env: add your OPENAI_API_KEY or AZURE_OPENAI_* credentials
```

### 2. Launch with Docker Compose
```bash
docker-compose up --build
```
This starts:
- PostgreSQL + pgvector on port 5432
- FastAPI backend on port 8000
- Streamlit UI on port 8501

### 3. Generate synthetic data and train models
```bash
docker-compose exec api python -m src.data.synthetic_generator --lots 12000 --buyers 60000
docker-compose exec api python -m src.models.xgboost_reserve --train
docker-compose exec api python -m src.models.lgbm_participation --train
docker-compose exec api python -m src.data.seed_vector_store
```

### 4. Run a briefing
```bash
# Via API
curl -X POST http://localhost:8000/briefing \
  -H "Content-Type: application/json" \
  -d '{
    "lot_category": "ferrous_scrap",
    "lot_quantity_mt": 150.0,
    "lot_grade": "B",
    "lot_location_region": "east",
    "days_to_auction": 3,
    "auction_format": "english_ascending",
    "registered_buyer_count": 47
  }'

# Or open the Streamlit UI at http://localhost:8501
```

---

## Repository Structure

```
auctionsense/
├── src/
│   ├── agents/
│   │   ├── market_intel.py        # Agent 1: commodity price + news
│   │   ├── reserve_price.py       # Agent 2: XGBoost + SHAP + LLM rationale
│   │   ├── participation.py       # Agent 3: LightGBM + outreach trigger
│   │   ├── lot_config.py          # Agent 4: pgvector RAG + LLM reasoning
│   │   └── strategist.py          # Orchestrator: LangGraph StateGraph
│   ├── data/
│   │   ├── synthetic_generator.py # Parameterised dataset generator
│   │   ├── feature_engineering.py # Feature transforms + scaling
│   │   └── seed_vector_store.py   # Load historical lots into pgvector
│   ├── models/
│   │   ├── xgboost_reserve.py     # XGBoost training + inference
│   │   ├── lgbm_participation.py  # LightGBM training + inference
│   │   └── mlflow_tracking.py     # Experiment logging
│   ├── api/
│   │   ├── main.py                # FastAPI application
│   │   ├── schemas.py             # Pydantic input/output models
│   │   └── routers/
│   │       ├── briefing.py        # POST /briefing
│   │       ├── health.py          # GET /health
│   │       └── history.py         # GET /history
│   └── ui/
│       └── streamlit_app.py       # Interactive demo dashboard
├── configs/
│   ├── .env.example               # Environment variable template
│   ├── hyperparameters.yaml       # All ML hyperparameter grids
│   ├── agent_prompts.yaml         # All LLM system + user prompts
│   └── compliance_rules.yaml      # Configurable floor price rules
├── notebooks/
│   ├── 01_EDA.ipynb               # Dataset exploration
│   ├── 02_model_evaluation.ipynb  # Full results + baseline comparison
│   └── 03_shap_analysis.ipynb     # SHAP explainability visualisation
├── tests/
│   ├── test_agents.py             # Agent unit tests
│   ├── test_models.py             # ML model tests
│   └── test_api.py                # FastAPI integration tests
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Adapting for Your Sector

AuctionSense is sector-agnostic. To deploy for your auction context:

**1. Commodity API** — edit `configs/agent_prompts.yaml`:
```yaml
market_intel:
  commodity_categories:
    coal: "Coal India e-auction index"
    agricultural: "NCDEX commodity futures"
```

**2. Compliance rules** — edit `configs/compliance_rules.yaml`:
```yaml
sectors:
  government_liquidation:
    reserve_floor_pct_of_book_value: 0.50
    approval_required_above_value: 5000000
  commercial_scrap:
    reserve_floor_pct_of_spot: 0.85
```

**3. Real data** — replace the synthetic generator with your own data loader:
```python
# src/data/your_data_loader.py
from src.data.schemas import LotRecord, BuyerLotRecord

def load_lots() -> list[LotRecord]:
    # Load from your database / CSV / API
    ...
```

---

## Results Summary

| Component | Metric | Value | Baseline |
|---|---|---|---|
| Reserve Price Agent | MAPE | **4.7%** | 11.2% (spot price) |
| Reserve Price Agent | R² | **0.91** | 0.61 (spot price) |
| Participation Agent | AUC-ROC | **0.84** | 0.50 (majority class) |
| Participation Agent | Recall (HIGH risk) | **0.80** | — |
| Lot Config Agent | Expert accuracy | **4.1 / 5.0** | N/A (no prior system) |
| Full pipeline latency | Wall clock | **38s mean** | N/A |

---

## Citation

If you use AuctionSense in research or commercial work, please cite:

```bibtex
@article{ray2026auctionsense,
  title={AuctionSense: An Agentic AI Framework for Seller-Side Pre-Auction Intelligence in B2B Physical Industrial Commodity Auctions},
  author={Ray, Debendra},
  journal={Zenodo},
  year={2026},
  doi={10.5281/zenodo.XXXXXXX},
  url={https://github.com/debray2523/auctionsense}
}
```

---

## Licence

Apache 2.0 — free for commercial and research use. Attribution required.

---

## Author

**Dr. Debendra Ray, DBA**
AI Architect | Enterprise AI Practitioner
ORCID: [0009-0002-5784-4442](https://orcid.org/0009-0002-5784-4442)

*Built on AGENT-G and RAPID-AI frameworks for enterprise agentic AI.*
