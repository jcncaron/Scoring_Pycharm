# Scoring_Pycharm
Pycharm files - Implementez un modèle de scoring

Repository to deploy an API concerning the ability of a single customer to repay or not his loan. The prediction is based on a lgbmclassifier model, trained on a dataset of about 300 k customers and 795 features.
This API consists in:
- For in single customer, displaying the predict_proba score to not repay his loan,
- displaying if the customer credit application is accepted or rejected
- providing a hyperlink to interactive dashboard

root directory:
---------------------
* main.py: app
  - Importing modules from flask library
  - Importing functions from functions/functions.py
  - Create app object with Flask
  - Prepare dataset for a single customer from test_df_1000 and predict with model_lgbm_1.joblib
  - Return the customer id, the prediction results and the credit application response
* Procfile: specifies the commands that are executed by the app
  - declare the app's web server
* requirements.txt: keeps track of the modules and packages used in the project
  - created with 'pip freeze > requirements.txt command' in terminal
* runtime.txt: declares the exact python version number to use (for Heroku deployment for example)

static directory:
-----------------
* style.css: template for the api's text style

templates directory:
---------------------
* layout.html: template for displaying the api

functions directory:
------------------------------
* functions.py : functions used in main.py (prepare_customer_data & predict_probability)

data directory:
----------------
* model_lgbm_1.joblib: lgbmclassifier serialized model in joblib format
* test_df_1000: dataset with 1000 customers only
* dict_idx_SK_ID_CURR.npy : dictionnary that has customer indexes as keys and customer ids ('SK_ID_CURR') as values
