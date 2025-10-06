# Predicting Civil War: An Integrated Supervised and Unsupervised ML Approach  
This project is part of the LSE's [DS202W Data Science for Social Scientists](https://lse-dsi.github.io/DS202/2024/winter-term/) course.  

**How best can we predict civil war?**

In this project, we analyse Sambinis' dataset to investigate **88 political, economic, and social features** that may influence the likelihood of civil war. Our goal is to:
- Identify **most influential variables**
- Evaluate **predictive performance** across various ML models
- Explore **clustering and anomaly detection** for novel insights

---

## Workflow Overview

### 1. 📊 Feature Selection & Engineering

* **Initial Screening**: Ran LightGBM and Random Forest on all 88 features to extract top predictors by importance.
* **Literature-Informed Refinement**: Integrated features from theories of civil war onset covering:

  * Structural factors: e.g., ethnic fractionalization, GDP, military size.
  * Dynamic/process-based variables: e.g., political competition, regime type.
* **Transformations**:

  * Applied log, lag, and growth transformations where necessary.
  * Addressed multicollinearity (esp. for linear models).
  * Evaluated skewness and applied normalization when needed.

---

### 2. 🧭 Unsupervised Learning

* **Clustering**:

  * Grouped countries by decade and region to identify underlying structural similarities.
  * Applied PCA/UMAP to reveal latent patterns in civil war risk.
  * Used silhouette scores and feature-based clustering metrics to validate and prioritize features.
* **Anomaly Detection**:

  * Flagged historical outliers and unexpected conflict events.
  * Provided exploratory insights and robustness checks for supervised models.

---

### 3. 🤖 Supervised Modeling

* **Algorithms Used**: Logistic Regression, LightGBM, Random Forest.
* **Train-Test Split**: Stratified by time to preserve temporal order.
* **Imbalance Handling**:

  * Used class weighting for tree-based models.
  * Tuned probability thresholds (e.g., <0.5) to reduce false negatives.
* **Evaluation Metrics**:

  * Emphasized **recall, F2 score, and PR AUC** due to high cost of missed conflict.
 

---

### 4. 🔁 Iterative Integration

* Reconciled differing modeling approaches across team members.
* Standardized data preprocessing, evaluation, and model structure.
* Linked unsupervised insights (e.g., decade clusters) to supervised design choices.

---

## 📊 Key Results

- **Top predictors of civil war include:**
    - infant
    - illiteracy
    - lpopns
    - elfo2
    - durable
    - rcd
    - lmtnest
    - sxpsq
    - gdp\_growth\_lag1
    - gdp\_growth\_deviation\_std
    - autch98 

- **Clustering insights:**  
  KMeans clustering grouped countries into 3 clusters, broadly aligning with autocracies, transitioning regimes, and stable democracies.

- **Anomaly detection highlights:**  
  Isolation Forest flagged ~12% of countries as anomalous in the 2 years before conflict outbreak.

- **Supervised learning**  
  **Random Forest performed the best overall**, achieving the highest test F1, F2, and recall scores. Logistic Regression had good recall but lower precision, while LightGBM showed signs of overfitting. All models struggled with the extreme class imbalance, and lowering the classification threshold below 0.5 did not improve F2 scores due to increased false positives.

---

## 📁 Repository Contents
| File/Folder          | Description                                      |
|----------------------|--------------------------------------------------|
| `civic_tensor.ipynb` | Main Jupyter notebook with full analysis         |
| `requirements.txt`   | List of all Python packages needed               |
| `README.md`          | You’re reading it!                               |
| `reflections/`       | Folder containing personal notes and reflections |

---

## 📦 Required Libraries

To install the necessary packages:

```bash
pip install -r requirements.txt
```

## 📦 Required Libraries

This project uses:

- `pandas`, `numpy`, `matplotlib`, `seaborn`  
- `scikit-learn`, `xgboost`, `lightgbm`, `catboost`  
- `imblearn`, `shap`, `plotly`, `sweetviz`, `miceforest`  
- `lets-plot`, `yellowbrick`, `statsmodels`  

---

## 🧠 Theoretical Grounding

This project is backed by empirical literature in conflict studies and civil war onset. Feature engineering is informed by **Sambinis' framework** and includes:

- Political institutions  
- Ethnic fractionalization  
- Economic shocks and inequality  
- Regime type and transitions  

---

## 💬 Future Work

- Apply temporal modeling (e.g., LSTM, GRU) for sequential prediction  
- Build an interactive dashboard for policymakers  
- Integrate more recent datasets post-2000 for updated predictions  

---

## 👥 Team Contributions

| Name               | Role/Area                             | Key Contributions                                                                                                                                       |
|--------------------|----------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Vignesh & Vienna** | 🔍 EDA, Feature Selection & Engineering | - Conducted EDA and visualizations (e.g. civil war by region) <br> - Ran initial LightGBM and Random Forest models on all 88 features <br> - Identified top predictors and removed highly correlated features <br> - Mapped theory-driven features from literature (e.g. Blair & Sambanis, 2020) <br> - Engineered contextual year indicators (e.g. post-WW2, Cold War) <br> - Applied transformations (e.g. log scales, growth rates) <br> - Cleaned and preprocessed data for feature selection <br> - Ensured alignment between theoretical and statistical relevance <br> - Handled multicollinearity across different model types |
| **Andre**          | 📊 Unsupervised Learning (Clustering)   | - Applied clustering by decade and region using PCA/UMAP <br> - Evaluated feature relevance via between-cluster variance <br> - Assessed robustness using silhouette scores and cluster separability |
| **Amruh**          | ⚠️ Anomaly Detection                    | - Designed initial anomaly detection framework <br> - Explored connections between anomalies and feature reliability <br> - Proposed use of anomalies to inform supervised modeling and data validation |
| **Yuyao**          | 🧩 Project Integration & Quality Control | - Reviewed rubric and ensured compliance across all components <br> - Standardized model pipeline (train/test split, CV folds, metrics) <br> - Justified metric choices (e.g. F1 vs. accuracy, FP vs. FN tradeoffs) <br> - Tuned probability thresholds and explained rationale <br> - Integrated contributions into a cohesive project narrative |


---

## 🤖 AI Acknowledgement

Some parts of this project involved the use of ChatGPT to support our work. Specifically:

* The **initial literature search** was assisted by ChatGPT using its web search functionality to identify relevant sources on civil war prediction models. While the AI helped gather sources efficiently, every source was **read, reviewed, and interpreted by us** before being included in the analysis.
* ChatGPT also provided assistance in **structuring and drafting sections of Markdown**, such as formatting, phrasing, and improving flow. However, **all content was reviewed, edited, and finalized by us** to ensure accuracy and originality.
* ChatGPT/Deepseek were used in help with debugging our code in some parts



