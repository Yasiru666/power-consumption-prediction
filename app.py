from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib

app = Flask(__name__)

# Load your trained machine learning model
model = joblib.load('model/load_model_saved.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict(flat=True)
    print(data)

    # Convert form data to DataFrame
    input_data = pd.DataFrame([data])
    print(input_data)

    # Ensure numeric conversion
    #input_data = input_data.astype(float)
    #print(input_data)

    input_data = input_data.rename(columns={
    'current': 'Current (A)',
    'power factor': 'Power Factor',
    'temperature': 'Temperature (C)',
    'voltage': 'Voltage (V)'
})


    # Perform prediction
    prediction = model.predict(input_data)[0]
    print(prediction)


    return jsonify({
        'message': 'Prediction successful',
        'predicted_bill': prediction
    })

if __name__ == '__main__':
    app.run(debug=True)
