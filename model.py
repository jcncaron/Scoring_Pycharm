# Import libraries
import pandas as pd
from lightgbm import LGBMClassifier
import joblib

# Import train and test csv files
print('Importing train and test csv files...')
train_df_path = "C:/Users/33624/train_df.csv"
train_df = pd.read_csv("C:/Users/33624/train_df.csv")
test_df_path = "C:/Users/33624/test_df.csv"
test_df = pd.read_csv("C:/Users/33624/test_df.csv")

# Remove special characters in train_df and test_df feature names
print('Removing special characters in features names...')
train_df.columns = train_df.columns.str.replace(':', '')
train_df.columns = train_df.columns.str.replace(',', '')
train_df.columns = train_df.columns.str.replace(']', '')
train_df.columns = train_df.columns.str.replace('[', '')
train_df.columns = train_df.columns.str.replace('{', '')
train_df.columns = train_df.columns.str.replace('}', '')
train_df.columns = train_df.columns.str.replace('"', '')
test_df.columns = test_df.columns.str.replace(':', '')
test_df.columns = test_df.columns.str.replace(',', '')
test_df.columns = test_df.columns.str.replace(']', '')
test_df.columns = test_df.columns.str.replace('[', '')
test_df.columns = test_df.columns.str.replace('{', '')
test_df.columns = test_df.columns.str.replace('}', '')
test_df.columns = test_df.columns.str.replace('"', '')

# Create X_train, X_test and y_train
print('Creating train and test datasets...')
X_train = train_df.drop(['index', 'SK_ID_CURR', 'TARGET'], 1)
y_train = train_df.TARGET
X_test = test_df.drop(['index', 'SK_ID_CURR'], 1)

# Create lgbmclassifier model with parameters optimized by Bayesian optimization (Hyperopt)
# and fit the model on X_train, y_train
print('Creating and fitting lgbmclassifier model...')
lgbm_1 = LGBMClassifier(n_estimators=2260,
                        learning_rate=0.03899075221700721,
                        num_leaves=60,
                        max_depth=4,
                        feature_fraction=0.7669772202117525,
                        subsample=0.7166670596245164,
                        reg_lambda=0.6643116316114804,
                        random_state=11)
model_lgbm_1 = lgbm_1.fit(X_train, y_train)

# Serialize model_lgbm_1
print('Serializing trained lgbmclassifier model...')
joblib.dump(model_lgbm_1, 'model_lgbm_1.joblib')
