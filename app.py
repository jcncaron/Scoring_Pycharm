from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import numpy as np

# Import test_df
print('Importing test csv files...')
test_df_path = "C:/Users/33624/test_df.csv"
test_df = pd.read_csv("C:/Users/33624/test_df.csv")

# remove special characters in test_df feature names
print('Removing special characters in features names...')
test_df.columns = test_df.columns.str.replace(':', '')
test_df.columns = test_df.columns.str.replace(',', '')
test_df.columns = test_df.columns.str.replace(']', '')
test_df.columns = test_df.columns.str.replace('[', '')
test_df.columns = test_df.columns.str.replace('{', '')
test_df.columns = test_df.columns.str.replace('}', '')
test_df.columns = test_df.columns.str.replace('"', '')

# Create X_test (features matrix) from test_df
print('Creating X_test dataset...')
X_test = test_df.drop(['index', 'SK_ID_CURR'], 1)

# start flask
print('Starting flask and loading lgbmclassifier model...')
app = Flask(__name__)
model = joblib.load("C:/Users/33624/model_lgbm_1.joblib")


# render default webpage
@app.route('/')
def home():
    return render_template('index.html')


# connect and run the python backend
@app.route('/predict', methods=['POST'])
def predict():
    cust_idx = [x for x in request.form.values()]
    cust_feat = X_test.iloc[cust_idx, :].to_numpy().reshape((1, 795))
    result = model.predict_proba(cust_feat)
    output = result[0, 1]
    cust_numbers = test_df['SK_ID_CURR']
    if output < 0.09:
        prediction = f'Probability of not repaying the loan = {output}. \n ' \
                     f'Customer {cust_numbers[cust_idx]} should be able to repay his loan'
    else:
        prediction = f'Probability of not repaying the loan = {output}. \n ' \
                     f'Customer {cust_numbers[cust_idx]} should not be able to repay his loan'
    return render_template('index.html', prediction_text=prediction)


@app.route('/results', methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict_proba([np.array(list(data.values()))])

    output = prediction[0, 1]
    return jsonify(output)


# Run the application
print("Running the app...")
if __name__ == "__main__":
    app.run(debug=True)
