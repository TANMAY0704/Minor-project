from flask import Flask, render_template, request, redirect, url_for
import pandas as pd

app = Flask(__name__)

def save_to_dataframe(data):
    df = pd.DataFrame([data])
    df.to_csv('survey_data.csv', mode='a', index=False, header=not any(data))
    # If any(data) returns True, it means the header is already present in the file

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
    return render_template('result.html', data=data)

if __name__ == '__main__':
    app.run(debug=True)
