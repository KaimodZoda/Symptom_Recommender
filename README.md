# Medical Symptom Recommendation System

An association rule-based symptom recommendation system that suggests likely next symptoms based on patient demographics (gender, age) and current symptoms. Achieves **67.71% hit rate** with a 4-tier fallback strategy.

## System Architecture

1. `Standardize.py` → Extract & translate symptoms to create mapping
2. `Preprocessing.py` → Load data, standardize symptoms, encode demographics
3. `Rules_Mining.py` → Mine association rules using Apriori algorithm
4. `Test_Evaluation.py` → Evaluate system performance on test set
5. `engine.py` → Core recommendation engine (used by API)
6. `api.py` → FastAPI REST API for serving predictions

## Key Features

### 4-Method Fallback Strategy
- **Strict Rule** (Demographics + Symptoms) - Weight: 1.0
- **Age + Symptom Fallback** - Weight: 1.3
- **Gender + Symptom Fallback** - Weight: 1.1
- **Symptom-only Rule** - Weight: 0.9

### Technical Highlights
- Scoring Formula: `confidence × lift × (match_count²) × weight`
- Positive-only basket mining (Yes_* symptoms only)
- English→Thai translation with fuzzy matching (threshold: 80%)
- Fast pickle-based loading for production

## Requirements

Install all dependencies:
```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install pandas numpy mlxtend scikit-learn fastapi uvicorn deep-translator rapidfuzz openpyxl
```

## Input Data

**Excel file:** The Excel data file provided with the assignment  
*(Used only by Standardize.py and Preprocessing.py)*

**Format:** Contains `summary` column with JSON data including:
- `yes_symptoms`: Patient-confirmed symptoms
- `no_symptoms`: Patient-denied symptoms
- `idk_symptoms`: Patient-uncertain symptoms

---

## Pipeline Workflow

### File Dependencies

Before running each script, ensure these files are in the same directory:

| Script | Required Files |
|--------|---------------|
| `Standardize.py` | The Excel data file (assignment data) |
| `Preprocessing.py` | The Excel data file, `symptom_mapping.pkl` |
| `Rules_Mining.py` | `preprocessed_data.csv` |
| `Test_Evaluation.py` | `preprocessed_data.csv`, `symptom_mapping.csv` |
| `api.py` | `engine.py`, `symptom_mapping.pkl`, `association_rules.pkl` |
| `TestAPI.py` | API must be running |

### Step 1: Symptom Standardization

```bash
python Standardize.py
```

**What it does:**
- Extracts all unique symptoms from yes/no/idk columns
- Separates English and Thai symptoms
- Translates English→Thai using Google Translator
- Uses fuzzy matching (80% threshold) to consolidate similar terms

**Output Files:**
- `symptom_mapping.csv` (human-readable)
- `symptom_mapping.pkl` (fast loading)

### Step 2: Data Preprocessing

```bash
python Preprocessing.py
```

**What it does:**
- Loads Excel data and symptom mapping
- Standardizes all symptom names
- Encodes demographics:
  - Gender: M/F
  - Age groups: Adult (≤45), Middle (46-60), Senior (>60)
- Creates transaction baskets: `["G_M", "Age_Adult", "Yes_ไอ", "Yes_เจ็บคอ"]`
- Removes last symptom from yes_list (ground truth for evaluation)

**Output Files:**
- `preprocessed_data.csv`

**Basket Format:**
```python
["G_F", "Age_Middle", "Yes_ไอ", "Yes_มีไข้"]
```

### Step 3: Association Rule Mining

```bash
python Rules_Mining.py
```

**What it does:**
- Loads preprocessed data
- Applies Apriori algorithm:
  - `min_support = 0.003` (0.3% of transactions)
  - `min_lift = 1.2`
- Mines association rules from full dataset
- Exports rules to pickle

**Output Files:**
- `association_rules.pkl`

**Rule Format:**
```python
{
  "antecedents": {"G_M", "Age_Adult", "Yes_ไอ"},
  "consequents": {"Yes_เจ็บคอ"},
  "confidence": 0.75,
  "lift": 2.5
}
```

### Step 4: Evaluation (Optional)

```bash
python Test_Evaluation.py
```

**What it does:**
- Splits data 80/20 train/test
- Mines rules on training set only
- Evaluates on test set:
  - Input: All symptoms except last
  - Ground truth: Last symptom
  - Success: If ground truth appears in top 6 recommendations
- Tracks hits by method (Strict/Age/Gender/Symptom)
- Shows manual test with debug info

**Metrics:**
- Total Hit Rate: **67.71%** (65/96)
- Method Distribution:
  - Strict Rule: 56.9%
  - Age+Symptom: 29.2%
  - Gender+Symptom: 10.8%
  - Symptom-only: 3.1%

---

## API Usage

### Starting the API

```bash
python api.py
```

Or:
```bash
uvicorn api:app --reload
```

**API will run on:** http://127.0.0.1:8000  
**Interactive docs:** http://127.0.0.1:8000/docs

**Required Files (in same directory):**
- `symptom_mapping.pkl`
- `association_rules.pkl`
- `engine.py`

### API Endpoint

**POST** `/recommend`

**Request Body:**
```json
{
  "gender": "female",
  "age": 45,
  "current_symptoms": ["ไอ", "เจ็บคอ"],
  "top_n": 6
}
```

**Response:**
```json
{
  "status": "success",
  "count": 6,
  "recommendations": [
    {
      "symptom": "มีไข้",
      "score": 12.5,
      "confidence": 0.75,
      "lift": 2.8,
      "source": "Strict Rule"
    }
  ]
}
```

### Testing the API

```bash
python TestAPI.py
```
*(Make sure API is running first)*

---

## File Descriptions

### Core Pipeline Files

| File | Description |
|------|-------------|
| `Standardize.py` | Extracts symptoms and creates mapping file with translation |
| `Preprocessing.py` | Processes data: standardization + demographic encoding |
| `Rules_Mining.py` | Mines association rules and exports to pickle |
| `Test_Evaluation.py` | Evaluates system performance with train/test split |

### Production Files

| File | Description |
|------|-------------|
| `engine.py` | Core recommendation engine (SymptomRecommender class) |
| `api.py` | FastAPI REST API server |
| `TestAPI.py` | Client script to test API endpoints |

### Data Files

| File | Description |
|------|-------------|
| `symptom_mapping.csv/pkl` | Symptom standardization lookup table |
| `preprocessed_data.csv` | Processed data with baskets |
| `association_rules.pkl` | Mined association rules |

### Utility Files

| File | Description |
|------|-------------|
| `Pkl_Test.py` | Inspect pickle files content |

---

## Technical Details

### Association Rule Mining

**Algorithm:** Apriori (mlxtend)

**Parameters:**
- `min_support`: 0.003
- `min_lift`: 1.2

**Scoring Formula:**
```
score = confidence × lift × (match_count²) × weight
```

**Where:**
- `confidence`: P(consequent|antecedent)
- `lift`: How much more likely than random
- `match_count`: Number of input features matching rule antecedent
- `weight`: Method-specific multiplier

**Method Weights:**
- Strict Rule (Demographics+Symptoms): 1.0
- Age+Symptom Fallback: 1.3
- Gender+Symptom Fallback: 1.1
- Symptom-only Rule: 0.9

### Demographic Encoding

**Gender:**
- `male` → `G_M`
- `female` → `G_F`

**Age Groups** (pd.cut with right-inclusive bins):
- `Age_Adult`: 0 < age ≤ 45
- `Age_Middle`: 45 < age ≤ 60
- `Age_Senior`: 60 < age ≤ 100

---

## Performance

### Hit Rate: 67.71% (Recall@6)
- 65 out of 96 test cases correctly predicted

### Method Contribution
- **Strict Rule:** 56.9% of hits (high precision, demographic-aware)
- **Age+Symptom:** 29.2% of hits (good fallback when gender doesn't match)
- **Gender+Symptom:** 10.8% of hits (complementary fallback)
- **Symptom-only:** 3.1% of hits (rare but useful)

### Average Metrics by Method

| Method | Confidence | Lift |
|--------|-----------|------|
| Strict Rule | 0.42 | 5.67 |
| Age+Symptom | 0.58 | 16.41 |
| Gender+Symptom | 0.63 | 10.97 |
| Symptom-only | 0.03 | 1.27 |

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Warning: symptom_mapping.pkl not found" | Run `python Standardize.py` first |
| "Preprocessed data file not found" | Run `python Preprocessing.py` first |
| "Model failed to load" (API) | Ensure `association_rules.pkl` exists in same directory as `api.py` |
| Low hit rate after changes | Check fuzzy match threshold (80%), mining parameters, weights |
| Translation errors | Check internet connection (Google Translator needs network)<br>Or install: `pip install deep-translator rapidfuzz` |

---

## Development Notes

### Design Decisions

1. **Positive-only baskets:** No_ and Idk_ features removed (improved from 65% to 68%)
2. **Squared match_count:** Rewards more specific rules with better context
3. **Fuzzy matching at 80%:** Balances consolidation vs. specificity
4. **4-tier fallback:** Ensures recommendations even with sparse data

### Key Improvements from Initial Version

- Removed no/idk penalty logic (simplified API)
- Modular architecture (separated preprocessing/mining/evaluation)
- Pickle format for fast loading
- Balanced method weights (Strict 1.0, Age 1.3, Gender 1.1, Symptom 0.9)
- Achieved 67.71% hit rate (stable)

---

## Contact & Support

For questions or issues with this system, please refer to the evaluation results in `Test_Evaluation.py` output or review the API documentation at http://127.0.0.1:8000/docs when running.

**Last Updated:** January 2026  
**System Version:** 1.3.2
