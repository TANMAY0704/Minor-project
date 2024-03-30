from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import pickle
import numpy as np
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import nltk

nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('wordnet')

app = Flask(__name__)

# Load the trained model
with open('random_forest_model.pkl', 'rb') as file:
    rf_classifier = pickle.load(file)

def preprocess_data(data):
    #capitalize and data
    data.loc[:, 'X3'] = data['X3'].str.strip().str.lower()
    data.loc[:, 'X3'] = data['X3'].str.replace(' ', '')  # Remove spaces

    replacements = {'up': 'Uttar Pradesh', 'uttarpradesh': 'Uttar Pradesh', 'mp': 'Madhya Pradesh', 'delhincr': 'Delhi','mahrashtra':'Maharashtra','rajisthan' : 'Rajasthan','hariyana' : 'Haryana','maharastra':'Maharashtra'}
    data.loc[:, 'X3'] = data['X3'].replace(replacements)

    data.loc[:, 'X3'] = data['X3'].apply(lambda x: x.title())
    data['X16'] = data['X16'].astype(str)

    # Sentiment analysis
    sia = SentimentIntensityAnalyzer()
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()

    negative_responses = ['no', 'nope', 'nothing', 'na', 'nope.', 'nah', 'not at all', 'no.']

    data['X16_sentiment'] = data['X16'].apply(lambda x: 0.0 if any(token in [stemmer.stem(lemmatizer.lemmatize(word)) for word in nltk.word_tokenize(x.lower())] for token in negative_responses) else sia.polarity_scores(x)['compound'] if isinstance(x, str) else 0.0)

    # One-hot encode categorical columns
    columns_to_encode = [col for col in data.columns if col not in ['X2', 'X6', 'X17', 'X16', 'X16_sentiment']]
    data_encoded = pd.get_dummies(data, columns=columns_to_encode, drop_first=False)

    # Ensure test data contains all columns present in training data
    missing_columns = set(data_encoded.columns) - set(data.columns)
    for col in missing_columns:
        data_encoded[col] = False  

    # Reorder columns to match the order in the training data
    data_encoded = data_encoded[data_encoded.columns]

    return data_encoded.drop(columns=['X16','X6'])

def map_input_values(data):
    # Define mappings for each column
    mappings = {
        'survey': {
            'NoChange': 'No change',
            'IncreasedAppetite': 'Increased appetite',
            'DecreasedAppetite': 'Decreased appetite',
            'WeightGain': 'Weight gain',
            'WeightLoss': 'Weight loss',
            'SlightDecline': 'Slight decline',
            'SignificantDecline': 'Significant decline'
        },
        # Define mappings for other columns here if needed
    }
    
    # Iterate over each column
    for column in data.columns:
        # Check if column is present in mappings
        if column in mappings:
            # Apply mapping to the column
            data[column] = data[column].map(mappings[column]).fillna(data[column])
    
    return data

@app.route('/')
def index():
    return redirect(url_for('survey'))

@app.route('/survey', methods=['GET', 'POST'])
def survey():
    if request.method == 'POST':
        # Process form submission
        data = request.form.to_dict()
     
        # Map input values
        #data = map_input_values(pd.DataFrame([data])) 

        print("Form Data:", data)
        processed_data = preprocess_data(data)  # Preprocess form data

        # Predict using the trained model
        
        
        # Construct query string
        query_string = '&'.join([f"{key}={value}" for key, value in data.items()])
        return redirect(url_for('result', data=query_string, prediction=''))  # Redirect to result page with data and prediction

    return render_template('form.html')

@app.route('/result')
def result():
    data_string = request.args.get('data')  # Get the data string from the query string
    data = dict(item.split('=') for item in data_string.split('&'))
    prediction = request.args.get('prediction')  # Get the prediction
    return render_template('result.html', data=data, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
