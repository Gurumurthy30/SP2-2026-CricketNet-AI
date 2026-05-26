# Sona Power Predict – 2026

**College Name:** Anna University Regional Campus, Tirunelveli  
**Team Name:** CricketNet AI

## Team Members

| Name | Year | Department |
|------|------|------------|
| GURUMURTHY R | Year 4 | Geo Informatics Engineering (GEO) |
| KARAN S | Year 4 | Electronics and Communication Engineering (ECE) |

## Libraries Used

Based on `mymodelfile.py`, the following Python libraries are used:

- **pandas** – DataFrame manipulation, grouping, and handling ball-by-ball historical IPL data.
- **numpy** – Numerical operations, array handling, and linear trend fitting (`np.polyfit`).
- **scikit-learn (sklearn)** – Used for the model: `SVR` (Support Vector Regressor with Radial Basis Function (RBF) Kernel) from `sklearn.svm`.

## Approach

The goal is to predict IPL Powerplay (overs 1–6) scores using historical ball-by-ball data.

**Feature Engineering:** Each innings is described by **35 features** split into two groups:

- **Team identity (22 features):** Batting team and bowling team, each one-hot encoded across 11 franchises.
- **Match context (13 features):** Venue, inning, wickets, season, toss, venue scoring averages, venue six-rate, batting team's recent form (last 3 matches), and venue scoring trend over seasons.

**Model Selection:** We tested models across all major families — linear (Ridge, Lasso, ElasticNet), non-linear (SVR), tree-based (Decision Tree, Random Forest, Extra Trees), and boosting (Gradient Boosting, XGBoost, LightGBM) — with hyperparameters tuned via `ParameterSampler`. Models were ranked using an asymmetric MAE that penalises under-predictions 2× more than over-predictions, matching the competition scoring.

**Final Model:** `SVR` with an RBF kernel (`C=113.08`, `epsilon=3.42`) achieved the best MAE and was chosen as the final model.

## License

MIT License
