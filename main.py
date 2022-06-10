from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np
import pickle

# Load machine learning model
model = joblib.load("model_lgbm_1.joblib")

# function to prepare data for 1 customer
def prepare_customer_data(customer_id):
    # Load the dictionary with indexes and SK_ID_CURR of test_df
    my_dict = np.load('dict_idx_SK_ID_CURR.npy', allow_pickle='TRUE').item()
    # Create lists of keys and values of my_dict
    list_keys = list(my_dict.keys())
    list_values = list(my_dict.values())
    # Save customer index
    customer_index = list_keys[list_values.index(customer_id)]
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
    result = np.round(model.predict_proba(X_test)[0, 1], 3)
    if result < 0.09:
        prediction = f'Predicted probability of not repaying the loan = {result}. ' \
                     f'Credit application of Customer n°{customer_id} is accepted'
    else:
        prediction = f'Predicted probability of not repaying the loan = {result}. ' \
                     f'Credit application of Customer n°{customer_id} is rejected'

    return prediction

# Create app object
app = Flask(__name__)


# render default webpage
@app.route('/')
def home():
    return render_template('layout.html')


# connect and run the python backend
@app.route('/', methods=['POST'])
def predict():
    user_input = int(request.form.get('customer_id'))
    # Get data for 1 customer
    customer_id, X_test = prepare_customer_data(customer_id=user_input)
    # Predict probability of not repaying the loan for random customer
    prediction = predict_probability(X_test, customer_id)
    return render_template('layout.html', prediction_text=prediction)

if __name__ == '__main__':
    app.run(debug=True)