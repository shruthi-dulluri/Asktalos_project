from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the input data
        feature1 = float(request.form["Material Quantity"])
        feature2 = float(request.form["Additive Catalyst"])
        feature3 = float(request.form["Ash Component"])
        feature4 = float(request.form["Water Mix"])
        feature5 = float(request.form["Plasticizer"])
        feature6 = float(request.form["Moderate Aggregator"])
        feature7 = float(request.form["Refined Aggregator"])
        feature8 = float(request.form["Formulation Duration"])
        # Add more features if necessary

        # Load your dataset (or use existing data)
        # Perform your multiple linear regression here
        # Replace this with your actual model training and prediction steps
        data = pd.read_csv("Material Compressive Strength Experimental Data (1).csv")

        data=data.dropna()
        data=data.reset_index(drop=True)



        X = data[['Material Quantity (gm)', 'Additive Catalyst (gm)',
                  'Ash Component (gm)', 'Water Mix (ml)', 'Plasticizer (gm)',
                 'Moderate Aggregator', 'Refined Aggregator', 'Formulation Duration (hrs)']]
        y = data['Compression Strength MPa']
        scaler = StandardScaler()
        X = scaler.fit_transform(X)


        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model
        model = RandomForestRegressor(n_estimators=10)
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        mse= mean_squared_error(y_test, prediction)

        # Display the result
        return f'Predicted value: {prediction[0]}, \n Mse  :{mse}'


if __name__ == '__main__':
    app.run(debug=True)



