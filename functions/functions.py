import pandas as pd
import numpy as np
import joblib


# function to prepare data for 1 customer
def prepare_customer_data(customer_id):
    # Load the dictionary with indexes and SK_ID_CURR of test_df
    my_dict = np.load('dict_idx_SK_ID_CURR.npy', allow_pickle='TRUE').item()
    # Create lists of keys and values of my_dict
    my_dict_keys = list(my_dict.keys())
    my_dict_values = list(my_dict.values())
    # Save customer index
    customer_index = my_dict_keys[my_dict_values.index(customer_id)]
    # Import just 1 line (1 customer) of test_df
    test_df_path = "test_df_1000.csv"
    test_df = pd.read_csv(test_df_path,
                          nrows=1,
                          skiprows=(range(1, customer_index + 1)),
                          header='infer')
    # Remove special characters in test_df feature names
    test_df.columns = test_df.columns.str.replace(':', '')
    test_df.columns = test_df.columns.str.replace(',', '')
    test_df.columns = test_df.columns.str.replace(']', '')
    test_df.columns = test_df.columns.str.replace('[', '')
    test_df.columns = test_df.columns.str.replace('{', '')
    test_df.columns = test_df.columns.str.replace('}', '')
    test_df.columns = test_df.columns.str.replace('"', '')
    # Create X_test (features matrix) from test_df
    X_test = test_df.drop(['index', 'SK_ID_CURR'], 1)

    return customer_id, X_test


# Function to predict probability
def predict_probability(X_test, customer_id):
    # Load machine learning model
    model = joblib.load("model_lgbm_1.joblib")
    # Predict the probability of not repaying the loan and save it in result
    result = np.round(model.predict_proba(X_test)[0, 1], 3)
    # Save the variables to display on the API
    customer = f'CUSTOMER N°{customer_id}'
    if result < 0.09:
        credit = 'Credit application : ACCEPTED'
        prediction = f'Predicted probability = {result}'
    else:
        credit = f'Credit application : REJECTED'
        prediction = f'Predicted probability = {result}'

    return customer, credit, prediction
