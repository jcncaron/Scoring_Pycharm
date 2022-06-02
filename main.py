from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

# Load machine learning model
model = joblib.load("C:/Users/33624/model_lgbm_1.joblib")


# function to prepare data for 1 customer
def prepare_customer_data(customer_index):
    # Import just 1 line (1 customer) of test_df
    test_df_path = "C:/Users/33624/test_df.csv"
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

    return test_df, X_test


# Function to predict probability
def predict_probability(X_test, test_df):
    result = np.round(model.predict_proba(X_test)[0, 1], 3)
    # Save customer SK_ID_CURR number as cust_number
    cust_number = test_df['SK_ID_CURR'][0]
    if result < 0.09:
        prediction = f'Probability of not repaying the loan = {result}. ' \
                     f'Credit application of Customer n°{cust_number} is accepted'
    else:
        prediction = f'Probability of not repaying the loan = {result}. ' \
                     f'Credit application of Customer n°{cust_number} is rejected'

    return prediction

# Create app object
app = Flask(__name__)


# render default webpage
@app.route('/')
def home():
    return render_template('layout.html')


# connect and run the python backend
@app.route('/', methods=['GET', 'POST'])
def predict():
    user_input = int(request.form.get('customer_index'))
    # Get data for 1 customer
    test_df, X_test = prepare_customer_data(customer_index=user_input)
    # Predict probability of not repaying the loan for random customer
    prediction = predict_probability(X_test, test_df)
    return render_template('layout.html', prediction_text=prediction)

if __name__ == '__main__':
    app.run()