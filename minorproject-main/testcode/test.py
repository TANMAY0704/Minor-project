import requests
import json

# Sample JSON data to send in the POST request
data = {
    "X2": 20,
    "X4_Female": False,
    "X4_Male": True,
    "X4_Others": False,
    "X4_Prefer_not_to_say": False,
    "X7_Everyday": True,
    "X7_No": False,
    "X7_Often": False,
    "X7_Rarely": False,
    "X7_Sometimes": False,
    "X8_Everyday": False,
    "X8_No": False,
    "X8_Often": False,
    "X8_Rarely": True,
    "X8_Sometimes": False,
    "X9_Decreased_appetite": False,
    "X9_Increased_appetite": True,
    "X9_No_change": False,
    "X9_Weight_gain": False,
    "X9_Weight_loss": False,
    "X10_Everyday": False,
    "X10_No": True,
    "X10_Often": False,
    "X10_Rarely": False,
    "X10_Sometimes": False,
    "X11_Everyday": False,
    "X11_No": True,
    "X11_Often": False,
    "X11_Rarely": False,
    "X11_Sometimes": False,
    "X12_Everyday": False,
    "X12_No": True,
    "X12_Often": False,
    "X12_Rarely": False,
    "X12_Sometimes": False,
    "X13_Everyday": False,
    "X13_No": True,
    "X13_Often": False,
    "X13_Rarely": False,
    "X13_Sometimes": False,
    "X14_Everyday": False,
    "X14_No": True,
    "X14_Often": False,
    "X14_Rarely": False,
    "X14_Sometimes": False,
    "X15_No_change": False,
    "X15_Significant_decline": False,
    "X15_Slight_decline": True,
    "X3_Delhi": False,
    "X3_Haryana": True,
    "X3_Madhya_Pradesh": False,
    "X3_Maharashtra": False,
    "X3_Odisha": False,
    "X3_Other": False,
    "X3_Uttar_Pradesh": False,
    "X3_West_Bengal": False,
    "X3_assam": False,
    "X3_bihar": False,
    "X3_jharkhand": False,
    "X5_11th": False,
    "X5_12th": True,
    "X5_Bachelor's_1st_Year": False,
    "X5_Bachelor's_2nd_Year": False,
    "X5_Bachelor's_3rd_Year": False,
    "X5_Bachelor's_4th_Year": False,
    "X5_Chartered_Accountancy": False,
    "X5_Graduate": False,
    "X5_Master's_Final_year": False,
    "X5_Other": False,
    "X5_Teaching_profession": False,
    "X6_B.Tech": False,
    "X6_B.tech": False,
    "X6_BTech": False,
    "X6_BTech_CSE": False,
    "X6_Btech": False,
    "X6_CSE": False,
    "X6_Computer_Science": True,
    "X6_Computer_Science_Engineering": False,
    "X6_Computer_science": False,
    "X6_Computer_science_engineering": False,
    "X6_Other": False
}

# Send POST request to the Flask app
json_data = json.dumps(data)

# Send POST request to the Flask app
url = 'http://127.0.0.1:5000/predict'
headers = {'Content-Type': 'application/json'}
response = requests.post(url, headers=headers, data=json_data)

# Check if the request was successful
if response.status_code == 200:
    # Print the JSON response from the Flask app
    print(response.json())
else:
    print(f"Request failed with status code: {response.status_code}")