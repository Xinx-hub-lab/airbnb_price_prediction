# Kaggle: Duke CS671 Fall23 Airbnb Competition

## Code Structure
### `utils`
- Contains all functions used for data preprocessing, including one-hot encoding, data type conversions, etc.

### `preprocessings`
- Scripts for the complete preprocessing workflow.
- `preprocessing1`: Prepares data for XGBoost.
- `preprocessing2`: Prepares data for Random Forest. 
(Can be used for XGBoost as well, with attribute revision)
  - `train_clean_`: Cleans the training set.
  - `test_clean_`: Cleans the validation set and test set.
- Ensures that values used in training set preprocessing are replicated in test preprocessing to prevent data leakage.

### `mains`
- Scripts for hyperparameter tuning using grid search and cross-validation.
- Imports and utilizes `preprocessings`.
- `main_xgb` for XGBoost and `main_rf` for random forest.

### `Y_report_visualization`
- Jupyter Notebook containing all codes for visualization and data report. 
- Some plots that requires run time are not explicitly displayed.
- Please refer to `plot` folder.

## Dependencies
- pandas
- numpy
- scikit-learn
- xgboost
- itertools
- datetime
- matplotlib
- seaborn
- collections

- re
- warnings


You can install these packages using pip:
```bash
pip install pandas numpy scikit-learn xgboost itertools matplotlib seaborn collections


