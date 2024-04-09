from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import joblib
import nltk
from data import *
app = Flask(__name__, static_folder='static', template_folder='templates')




# Load the trained model and feature names
random_forest_model_tuple = joblib.load('C:/Users/pande/OneDrive/Documents/GitHub/Minor project/Final/model/Models/random_forest_model.pkl')

random_forest_model = random_forest_model_tuple[0]  # Extract the RandomForestClassifier from the tuple
feature_names = random_forest_model_tuple[1]  # Extract the feature names

def preprocess_data(data, rf):
    # Convert data to DataFrame
    df = pd.DataFrame.from_dict(data, orient='index').T

    # Preprocessing 'X3'
    if 'X3' in data:
        if isinstance(data['X3'], str):
            df.loc[:, 'X3'] = data['X3'].strip().lower()
        else:
            df.loc[:, 'X3'] = ''

    # Apply replacements and title case to 'X3'
    df.loc[:, 'X3'] = df['X3'].replace(replacements)
    df.loc[:, 'X3'] = df['X3'].apply(lambda x: x.title())

    # Replace categories in 'X6' using category_mapping
    df.loc[:,'X6'] = df['X6'].replace(category_mapping)

    # Convert 'X16' to string and perform sentiment analysis
    df['X16'] = df['X16'].astype(str)
    sia = SentimentIntensityAnalyzer()
    negative_responses = {'no', 'nope', 'nothing', 'na', 'nope.', 'nah', 'not at all', 'no.'}
    df['X16_sentiment'] = df['X16'].apply(lambda x: 0.0 if any(token in [PorterStemmer().stem(WordNetLemmatizer().lemmatize(word)) for word in nltk.word_tokenize(x.lower())] for token in negative_responses) else sia.polarity_scores(x)['compound'] if isinstance(x, str) else 0.0)

    # Encode categorical columns
    columns_to_encode = [col for col in df.columns if col not in ['X2', 'X16', 'X16_sentiment']]
    df_encoded = pd.get_dummies(df, columns=columns_to_encode, drop_first=False)

    # Transform columns starting with X3_, X5_, and X6_ to X3_Others, X5_Others, and X6_Others respectively
    for column in df_encoded.columns:
        if column.startswith('X3_') and column not in feature_names:
            df_encoded['X3_Others'] = True
            df_encoded.drop(columns=[column], inplace=True)
        elif column.startswith('X5_') and column not in feature_names:
            df_encoded['X5_Others'] = True
            df_encoded.drop(columns=[column], inplace=True)
        elif column.startswith('X6_') and column not in feature_names:
            df_encoded['X6_Others'] = True
            df_encoded.drop(columns=[column], inplace=True)

    # Reorder DataFrame columns to match the feature order of the RandomForest classifier
    df_encoded = df_encoded.reindex(columns=feature_names , fill_value=False)

    return df_encoded

@app.route('/')
def index():
    return render_template('form.html')

@app.route('/survey', methods=['POST'])
def survey():
    data = request.form.to_dict()

    # Preprocess form data
    processed_data = preprocess_data(data, random_forest_model)

    # Predict using the trained model
    prediction = random_forest_model.predict(processed_data)[0]

    # Redirect to the result page with prediction and processed data
    return redirect(url_for('result', prediction=prediction, processed_data=processed_data.to_html()))

@app.route('/result')
def result():
    prediction = request.args.get('prediction')
    processed_data = request.args.get('processed_data')
    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)