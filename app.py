import os
import joblib
import pandas as pd
from flask import Flask, request, render_template

app = Flask(__name__)

# Load the model at the start of the application
model_path = 'gb_model.joblib'

rf_model = joblib.load(model_path) if os.path.exists(model_path) else None


if rf_model is not None:
    print(f"Model loaded successfully from {model_path}.")
else:
    print(f"Model not found at {model_path}.")

# if scaler is not None:
#     print(f"Scaler loaded successfully from.")
# else:
#     print(f"Scaler not found at {scaler_path}.")

# Initialize LabelEncoders
# email_encoder = LabelEncoder()
# lead_quality_encoder = LabelEncoder()



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        try:

            tt_on_website = request.form['ttsw']
            tags = request.form['tags']  # Expecting 'yes' or 'no'
            lead_quality = float(request.form['lead_quality'])  # Expecting float input
            ln_actvty = request.form['ln_actvty']  # Expecting a string like 'low', 'medium', 'high'
            lead_orgin = request.form['lead_orgin']

            input_features = [
                tt_on_website,
                tags,  # Now this is an int
                lead_quality,
                ln_actvty,  # Now this is an int
                lead_orgin
            ]
            print(input_features)

            # Ensure thety correct number of values
            if len(input_features) == 5:
                # Prepare DataFrame for the model
                names = ['Total Time Spent on Website', 'Tags', 'Lead Quality', 'Last Notable Activity',  'Lead Origin']
                data = pd.DataFrame([input_features], columns=names)
                print(data)
            
                prediction = rf_model.predict(data)
                print(prediction[0])

                if prediction[0]== 0:
                    return render_template('result.html', prediction="Lead not converted")
                else:
                    return render_template('result.html', prediction="Lead converted")

                # Render the result page with the prediction
                # return render_template('result.html', prediction=prediction[0])

            else:
                raise ValueError("Invalid number of values. Expected 5 values.")
        
        except Exception as e:
            print(f"Error occurred: {e}")
            return render_template('result.html', result="Error", prediction=str(e))

if __name__ == "__main__":
    app.run(debug=True, port=4000)
