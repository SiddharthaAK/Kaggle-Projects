# Binary Classification on Bank Marketing Dataset  

Kaggle Competition: [Playground Series - Season 5, Episode 8](https://www.kaggle.com/competitions/playground-series-s5e8/data)  

This project was my attempt to strengthen foundations in supervised learning and binary classification. I treated it as a step-by-step learning exercise, moving from simple baselines to more advanced methods.  

---

## Process & Methods  

1. **Understanding the dataset**  
   - Explored the structure of the bank marketing dataset (categorical + numerical features).  
   - Checked class balance and data quality.  

2. **Baseline models**  
   - Started with **Logistic Regression** as a benchmark for interpretability.  
   - Tried a simple **Decision Tree** to understand splits on categorical variables.  

3. **Ensemble methods**  
   - Implemented **Random Forest** as the first ensemble model, which improved performance.  
   - Moved to **LightGBM**, **CatBoost** and **XGBoost** which offered faster training and better handling of categorical features.  

4. **Cross-validation & evaluation**  
   - Set up **Stratified K-Fold cross-validation** to ensure balanced splits.  
   - Used **ROC-AUC** as the main metric, as recommended for binary tasks.  

5. **Hyperparameter tuning**  
   - Used **Optuna** to optimize key LightGBM parameters (learning rate, num_leaves, depth, regularization).  
   - Achieved more stable and higher AUC compared to manual tuning.  

6. **Final model**  
   - Selected the tuned LightGBM model.  
   - Generated probability-based predictions for the test set.  
   - Created a `submission.csv` ready for Kaggle evaluation.  

---

## Key Learnings  
- The importance of building from simple baselines → advanced models.  
- How cross-validation prevents overfitting to a single train/test split.  
- Trade-offs between interpretability (Logistic Regression) and performance (LightGBM).  
- Practical use of Optuna for automated hyperparameter search.  

---

## Files  
- `main2.py` – Final training + inference pipeline with tuned LightGBM.  
- `hp-ft.py` – Hyperparameter tuning with Optuna.  
- `train.csv`, `test.csv`, `bank.csv` – Dataset files.(bank.csv is the original and train.csv and test.csv are synthetic data generated from bank.csv for the competition)  
- `submission.csv` – Final output for Kaggle.  
