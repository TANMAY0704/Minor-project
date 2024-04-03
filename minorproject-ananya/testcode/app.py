from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import joblib
import nltk
from data import *
app = Flask(__name__)

# Load the trained model
random_forest_model = joblib.load('random_forest_model.pkl')

def preprocess_data(data, rf):
    df = pd.DataFrame.from_dict(data, orient='index').T

    # Check if 'X3' is present in the data
    if 'X3' in data:
        # Preprocessing 'X3'
        if isinstance(data['X3'], str):
            df.loc[:, 'X3'] = data['X3'].strip().lower()
        else:
            # Handle non-string values for 'X3'
            df.loc[:, 'X3'] = ''



    df.loc[:, 'X3'] = df['X3'].replace(replacements)
    
    df.loc[:, 'X3'] = df['X3'].apply(lambda x: x.title())


    df = df.replace(input_mapping)


    df['X3'] = df['X3'].apply(lambda x: x if x in indian_states_ut else 'Others')

    df['X16'] = df['X16'].astype(str)

    # Sentiment analysis
    sia = SentimentIntensityAnalyzer()
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()

    negative_responses = {'no', 'nope', 'nothing', 'na', 'nope.', 'nah', 'not at all', 'no.'}

    df['X16_sentiment'] = df['X16'].apply(lambda x: 0.0 if any(token in [stemmer.stem(lemmatizer.lemmatize(word)) for word in nltk.word_tokenize(x.lower())] for token in negative_responses) else sia.polarity_scores(x)['compound'] if isinstance(x, str) else 0.0)

    columns_to_encode = [col for col in df.columns if col not in ['X2', 'X16', 'X16_sentiment']]
    df_encoded = pd.get_dummies(df, columns=columns_to_encode, drop_first=False)
    df_encoded.drop(columns=['X16'])
    
    rf_feature_names = rf.feature_names_in_

    # Reorder DataFrame columns to match the feature order of the RandomForest classifier
    df_encoded = df_encoded.reindex(columns=rf_feature_names, fill_value=False)

    return df_encoded

@app.route('/')
def index():
    return redirect(url_for('survey'))

@app.route('/survey', methods=['GET', 'POST'])
def survey():
    if request.method == 'POST':
        # Process form submission
        data = request.form.to_dict()

        # Preprocess form data
        processed_data = preprocess_data(data,random_forest_model)

        if processed_data.empty:
            # Handle case where processed_data is empty
            return render_template('error.html', message="Form data is empty or preprocessing failed.")

        #Predict using the trained model
        prediction = random_forest_model.predict(processed_data)[0]

        # Construct query string
        query_string = '&'.join([f"{key}={value}" for key, value in data.items()])
        return redirect(url_for('result', data=query_string, prediction=prediction, processed_data=processed_data.to_html()))  # Redirect to result page with data, prediction, and processed data

    return render_template('form.html')


@app.route('/result')
def result():
    data_string = request.args.get('data')  # Get the data string from the query string
    data = dict(item.split('=') for item in data_string.split('&'))
    prediction = request.args.get('prediction')  # Get the prediction
    processed_data = request.args.get('processed_data')  # Get the processed data HTML representation
    return render_template('result.html', data=data, prediction=prediction, processed_data=processed_data)

if __name__ == '__main__':
    app.run(debug=True)
