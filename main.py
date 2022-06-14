from flask import Flask, render_template, request
from functions.functions import predict_probability, prepare_customer_data

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
    customer, credit, prediction = predict_probability(X_test, customer_id)
    return render_template('layout.html',
                           customer_text=customer,
                           prediction_text=prediction,
                           credit_text=credit)


if __name__ == '__main__':
    app.run(debug=True, port=2000)
