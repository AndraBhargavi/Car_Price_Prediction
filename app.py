from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

MODEL_FILE = 'car_price_model.pkl'
DATA_FILE = 'car_data.csv'

def train_and_save_model():
    df = pd.read_csv(DATA_FILE)

    if 'Year' not in df.columns:
        raise KeyError("'Year' column not found in car_data.csv")

    # Preprocessing
    df['Age'] = 2020 - df['Year']
    df.drop('Year', axis=1, inplace=True)
    df.rename(columns={'Selling_Price': 'Selling_Price(lacs)',
                       'Present_Price': 'Present_Price(lacs)',
                       'Owner': 'Past_Owners'}, inplace=True)

    # Encode categorical variables
    le = LabelEncoder()
    for col in ['Fuel_Type', 'Seller_Type', 'Transmission']:
        df[col] = le.fit_transform(df[col])

    X = df.drop(['Car_Name', 'Selling_Price(lacs)'], axis=1)
    y = df['Selling_Price(lacs)']

    model = RandomForestRegressor()
    model.fit(X, y)

    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(model, f)

    return model

# Load or train model
if os.path.exists(MODEL_FILE):
    with open(MODEL_FILE, 'rb') as f:
        model = pickle.load(f)
else:
    model = train_and_save_model()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            car_name = request.form['car_name']
            present_price = float(request.form['present_price'])
            kms_driven = int(request.form['kms_driven'])
            fuel_type = request.form['fuel_type']
            seller_type = request.form['seller_type']
            transmission = request.form['transmission']
            past_owners = int(request.form['past_owners'])
            age = int(request.form['age'])

            fuel_dict = {'Petrol': 0, 'Diesel': 1, 'CNG': 2}
            seller_dict = {'Dealer': 0, 'Individual': 1}
            trans_dict = {'Manual': 0, 'Automatic': 1}

            input_data = np.array([[present_price, kms_driven,
                                    fuel_dict[fuel_type],
                                    seller_dict[seller_type],
                                    trans_dict[transmission],
                                    past_owners, age]])

            prediction = model.predict(input_data)[0]

            return render_template('car.html',
                                   prediction=round(prediction, 2),
                                   car_name=car_name,
                                   present_price=present_price,
                                   kms_driven=kms_driven,
                                   fuel_type=fuel_type,
                                   seller_type=seller_type,
                                   transmission=transmission,
                                   past_owners=past_owners,
                                   age=age)

        except Exception as e:
            return f"Error occurred: {str(e)}"

    return render_template('car.html')

if __name__ == '__main__':
    app.run(debug=True)
