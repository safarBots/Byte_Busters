from flask import Flask, request, jsonify, render_template
import pickle  
import numpy as np
    

app = Flask(__name__)

# Load the model (assuming it was saved in Medimate.ipynb as a pickle file)
with open('rf_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def index():
    return render_template('homepage.html')  # Assuming your HTML file is named index.html

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Extract values from the data
    fever = 1 if data['fever'] == 'Yes' else 0
    cough = 1 if data['cough'] == 'Yes' else 0
    fatigue = 1 if data['fatigue'] == 'Yes' else 0
    difficulty_breathing = 1 if data['difficultyBreathing'] == 'Yes' else 0
    age = int(data['age']) if data['age'] else 0
    gender = 1 if data['gender'] == 'Male' else 0  # Assuming Male=1, Female=0
    blood_pressure = 1 if data['bloodPressure'] == 'High' else 0
    cholesterol_level = 1 if data['cholesterolLevel'] == 'High' else 0

    # Create feature array for the prediction model
    features = np.array([[fever, cough, fatigue, difficulty_breathing, age, gender, blood_pressure, cholesterol_level]])

    # Predict using the model
    prediction = model.predict(features)

    # Prepare response
    if prediction == 1:
        result = "Based on your symptoms, the prediction suggests you may have a high risk condition. Please consult a healthcare provider."
    else:
        result = "Your symptoms do not suggest a high-risk condition, but it's always best to consult a professional."

    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)