# **Conflict Data — EDA, NLP Classification, and Interactive Visuals**

## **Project Summary**

I analyze a sampled dataset og (conflict events) with three pillars:

1. **Exploratory Data Analysis (EDA)** — structure, quality checks, and descriptive insights by event type, region, country, and time.
2. **NLP Classification** — a baseline text classifier that predicts **civilian targeting** directly from the free‑text **`notes`** field.
3. **Interactive Linking & Brushing (Altair)** — a timeline selection (brush) that dynamically filters companion charts for fast, visual exploration.

> **Data shape:** 1,000 rows × 31 columns (confirmed via `df.info()`).

---

## **Dataset**

* **Source:** Sampled ACLED data (CSV: `data/df_sample_ads.csv`).
* **Key columns:**

  * Dates/Meta: `event_date`, `year`, `time_precision`, `timestamp`
  * Typology: `disorder_type`, `event_type`, `sub_event_type`, `interaction`
  * Actors: `actor1`, `assoc_actor_1`, `actor2`, `assoc_actor_2`, `inter1`, `inter2`
  * Location: `region`, `country`, `admin1–3`, `location`, `latitude`, `longitude`
  * Impact: `fatalities`, `civilian_targeting`
  * Text: `notes`
  * Sources: `source`, `source_scale`
* **Missingness (high-level):** `assoc_actor_1`, `assoc_actor_2`, `tags`, and some admin levels have notable gaps; `civilian_targeting` is sparse (only present when applicable).

---

## **What I Do in the Notebook**

### 1) **EDA**

* Load and audit the data (`.info()`, shape, nulls).
* Plot distributions for key categoricals (e.g., `disorder_type`, `event_type`, `region`, `country`) and outcomes (`fatalities`).
* Practical takeaways:

  * **Political violence** dominates the sample.
  * Event mix is skewed toward **violence against civilians** and **armed clashes**.
  * Regional concentration in **Western Africa**; country‑level hotspots (e.g., **Nigeria**).

### 2) **NLP Pipeline (Target: Civilian Targeting)**

**Goal:** Given the text in `notes`, predict whether an event **targets civilians**.

* **Label creation:**

  ```text
  targeted = 1 if civilian_targeting == "Civilian targeting" else 0
  ```
* **Preprocessing (spaCy `en_core_web_sm`):**

  * Lowercasing
  * Remove stop words and non‑alphabetic tokens
  * **Lemmatization**
  * Save to `clean_notes`
* **Vectorization:** `TfidfVectorizer(max_features=5000)`
* **Model:** `LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)`
* **Split:** Stratified **80/20** train–test

  * Train size: **800**, test size: **200**
  * Label distribution preserved (≈ 65% class 0 / 35% class 1)
* **Results (held‑out test set):**

  * **Accuracy:** **0.89**
  * **Macro F1:** **0.87**
  * **Per‑class:**

    * Class 0 (not targeted): P=0.90, R=0.93, F1=0.91 (support=129)
    * Class 1 (targeted): P=0.86, R=0.80, F1=0.83 (support=71)
  * **Confusion matrix:**

    ```
    [[120   9]
     [ 14  57]]
    ```
* **Why this baseline?** It’s simple, fast, and **interpretable** (TF‑IDF + linear model), yet already strong. It also provides coefficients for quick term‑level insight.

**Saved artifact:** `models/civilian_targeting_model.pkl` (full scikit‑learn Pipeline)

**Quick inference example:**

```python
import joblib
from notebooks.utils import spacy_clean  # or reuse the function from the notebook

model = joblib.load("models/civilian_targeting_model.pkl")
text = "Militants attacked civilians in a village during a protest."
pred = model.predict([spacy_clean(text)])
print("Predicted label (1=civilian targeting):", int(pred[0]))
```

> **Reproducibility:** I fix `random_state=42` and stratify splits. For deployment, I save the **entire pipeline** so preprocessing stays identical.

### 3) **Interactive Linking & Brushing (Altair)**

I build an **Altair** interaction where a **time‑range brush** on a **timeline scatter** (fatalities vs. `event_date`) **filters** a **bar chart of `event_type` counts**.

---

## **Project Structure**

```
.
├── data/
│   └── df_sample_ads.csv
├── models/
│   └── civilian_targeting_model.pkl
├── notebooks/
│   └── conflict_analysis.ipynb
├── README.md
└── requirements.txt
```

---

## **How I Run This**

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```
2. **Open the notebook**

   ```bash
   jupyter notebook notebooks/conflict_analysis.ipynb
   ```
3. **Follow sections in order** (EDA → NLP → Evaluation → Interactive Visuals).
4. **(Optional) Inference**: use the saved `civilian_targeting_model.pkl` as shown above.

### **Requirements**

```
pandas
numpy
matplotlib
seaborn
altair
scikit-learn
spacy
joblib
jupyter
```

> Plus the spaCy model: `en_core_web_sm`.

## **Ethical Note**

This is a **research/analysis** exercise on conflict data. Interpret results responsibly, be mindful of **biases in reporting sources**, and avoid drawing causal conclusions from descriptive patterns.

