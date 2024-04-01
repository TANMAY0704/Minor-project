from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import pickle
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import nltk

app = Flask(__name__)

# Load the trained model
with open('random_forest_model.pkl', 'rb') as file:
    rf_classifier = pickle.load(file)

def preprocess_data(data, rf_classifier):
    df = pd.DataFrame.from_dict(data, orient='index').T

    # Preprocessing steps
    df['X3'] = df['X3'].str.strip().str.lower().replace({
        'uttar pradesh': 'Uttar Pradesh',
        'up': 'Uttar Pradesh',
        'madhya pradesh': 'Madhya Pradesh',
        'mp': 'Madhya Pradesh',
        'maharashtra': 'Maharashtra',
        'maharastra': 'Maharashtra',
        'mahrashtra': 'Maharashtra',
        'haryana': 'Haryana',
        'odisha': 'Odisha',
        'west bengal': 'West Bengal',
        'chhattisgarh': 'Chhattisgarh',
        'karnataka': 'Karnataka',
        'delhi': 'Delhi',
        'bhubaneswar': 'Odisha',
        'faridabad': 'Haryana',
        'rajasthan': 'Rajasthan',
        'rajisthan': 'Rajasthan',
        'tamil nadu': 'Tamil Nadu',
        'Karnataka': 'Karnataka',
        'delhi ncr': 'Delhi',
    })

    df['X16'] = df['X16'].astype(str)

    # Sentiment analysis
    sia = SentimentIntensityAnalyzer()
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()

    negative_responses = {'no', 'nope', 'nothing', 'na', 'nope.', 'nah', 'not at all', 'no.'}

    df['X16_sentiment'] = df['X16'].apply(lambda x: 0.0 if any(token in [stemmer.stem(lemmatizer.lemmatize(word)) for word in nltk.word_tokenize(x.lower())] for token in negative_responses) else sia.polarity_scores(x)['compound'] if isinstance(x, str) else 0.0)

    # One-hot encode categorical columns
    columns_to_encode = [col for col in df.columns if col not in ['X2', 'X16', 'X16_sentiment']]
    df_encoded = pd.get_dummies(df, columns=columns_to_encode, drop_first=False)

    # Ensure test data contains all columns present in training data
    if hasattr(rf_classifier, 'estimators_'):
        feature_names = rf_classifier.estimators_[0].feature_importances_
    else:
        feature_names = rf_classifier.feature_importances_
    feature_names_in_ = df_encoded.columns

    missing_columns = set(feature_names) - set(feature_names_in_)
    for col in missing_columns:
        df_encoded[col] = False

    # Reorder columns to match the order in the training data
    df_encoded = df_encoded[feature_names_in_]

    return df_encoded.drop(columns=['X16'])
@app.route('/')
def index():
    return redirect(url_for('survey'))

@app.route('/survey', methods=['GET', 'POST'])
def survey():
    if request.method == 'POST':
        # Process form submission
        data = request.form.to_dict()

        # Preprocess form data
        processed_data = preprocess_data(data, rf_classifier)  # Pass rf_classifier argument

        # Predict using the trained model
        prediction = rf_classifier.predict(processed_data)[0]

        # Construct query string
        query_string = '&'.join([f"{key}={value}" for key, value in data.items()])
        return redirect(url_for('result', data=query_string, prediction=prediction))  # Redirect to result page with data and prediction

    return render_template('form.html')

@app.route('/result')
def result():
    data_string = request.args.get('data')  # Get the data string from the query string
    data = dict(item.split('=') for item in data_string.split('&'))
    prediction = request.args.get('prediction')  # Get the prediction
    return render_template('result.html', data=data, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)