
# Enhanced Credit Risk Model

## Credit Scoring Business Understanding

### 1. How does the Basel II Accord’s emphasis on risk measurement influence our need for an interpretable and well-documented model?

The Basel II Accord requires financial institutions to hold capital proportional to credit risk exposure. It mandates accurate estimation of Probability of Default (PD), Loss Given Default (LGD), and Exposure at Default (EAD). This regulatory framework demands **transparency, auditability, and documentation** of risk models. A "black-box" model, while potentially accurate, cannot justify capital reserves or be validated by auditors. Hence, **interpretable models (e.g., Logistic Regression with WoE)** and full lineage tracking via **MLflow** are essential for compliance.

### 2. Since we lack a direct "default" label, why is creating a proxy variable necessary?

We don’t have loan repayment history. However, **customer disengagement** (e.g., stopping transactions) can act as a proxy for financial distress. Using **RFM analysis**, we identify inactive users as high-risk (defaulters). This allows us to train a supervised model.

**Risks**:
- Some inactive users may have left for non-financial reasons.
- The proxy may not perfectly correlate with actual default.
- Risk of bias if behavioral patterns differ across demographics.

**Mitigation**: Use sensitivity analysis and update labels when real default data becomes available.

### 3. Trade-offs: Simple vs. Complex Models

| Factor | Logistic Regression (WoE) | Gradient Boosting |
|------|----------------------------|-------------------|
| **Interpretability** | ✅ High (coefficients = risk direction) | ❌ Low |
| **Regulatory Compliance** | ✅ Easy to explain | ⚠️ Needs SHAP/LIME |
| **Performance** | ⚠️ Moderate | ✅ High |
| **Monotonicity** | ✅ Enforceable via WoE | ❌ Hard |
| **Scorecard Translation** | ✅ Easy | ❌ Difficult |

**Conclusion**: Use **Logistic Regression with WoE** for production due to interpretability, even if XGBoost performs slightly better.

---

📌 **Screenshots**: See `notebooks/1.0-eda.ipynb` for EDA plots and MLflow UI.