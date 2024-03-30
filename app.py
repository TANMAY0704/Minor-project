from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import pickle

app = Flask(__name__)

# Load the trained Random Forest model
with open('random_forest_model.pkl', 'rb') as file:
    rf_classifier = pickle.load(file)

def save_to_dataframe(data):
    df = pd.DataFrame([data])
    df.to_csv('survey_data.csv', mode='a', index=False, header=not any(data))

def predict_data(data):
    # Convert data to DataFrame
    df = pd.DataFrame([data])

    # Handle missing columns
    missing_columns = set(rf_classifier.feature_names_in_) - set(df.columns)
    placeholder_columns = pd.DataFrame(0, index=df.index, columns=missing_columns)
    df = pd.concat([df, placeholder_columns], axis=1)

    # One-hot encode the data
    encoded_data = rf_classifier.transform(df)

    # Predict using the loaded Random Forest model
    predictions = rf_classifier.predict(encoded_data)
    return predictions

@app.route('/')
def index():
    return redirect(url_for('survey'))

@app.route('/survey', methods=['GET', 'POST'])
def survey():
    if request.method == 'POST':
        # Process form submission
        data = request.form.to_dict()
        print("Form Data:", data)
        save_to_dataframe(data)  # Save form data to DataFrame
        # Construct query string
        query_string = '&'.join([f"{key}={value}" for key, value in data.items()])
        return redirect(url_for('result', data=query_string))  # Redirect to result page with data

    return render_template('form.html')

@app.route('/result')
def result():
    data_string = request.args.get('data')  # Get the data string from the query string
    data = dict(item.split('=') for item in data_string.split('&'))
    # Predict using the loaded Random Forest model
    predictions = predict_data(data)
    return render_template('result.html', data=data, predictions=predictions)

if __name__ == '__main__':
    app.run(debug=True)
