import flask
from flask import render_template, request
import joblib
import pandas as pd
import numpy as np
import shap

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
                     f'Customer n°{cust_number} should be able to repay his loan'
    else:
        prediction = f'Probability of not repaying the loan = {result}. ' \
                     f'Customer n°{cust_number} should not be able to repay his loan'

    return prediction


# Function to create shap decision plot
def shap_d_plot(X_test):
    explainer = shap.TreeExplainer(model)
    X_test_0 = X_test.iloc[0, :].to_numpy().reshape((1, 795))
    shap_values_test = explainer.shap_values(X_test_0)
    d_plot = shap.decision_plot(base_value=explainer.expected_value[0],
                                shap_values=shap_values_test[0],
                                features=X_test,
                                feature_display_range=slice(-1, -11, -1))
    return d_plot


# Create app object
app = flask.Flask(__name__)


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
    d_plot = shap_d_plot(X_test)
    return render_template('layout.html', prediction_text=prediction, shap_graph=d_plot.html())


if __name__ == '__main__':
    app.run(debug=True)
