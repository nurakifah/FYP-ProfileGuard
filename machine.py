from flask import Flask, render_template, request, redirect, url_for
import joblib
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from flask_scss import Scss

app = Flask(__name__ , static_url_path='/static')
Scss(app, static_dir='static', asset_dir='scss') #SCSS

# Load the pre-trained model
model_filename = "multinomial_nb_model.joblib"
MultiNB = joblib.load(model_filename)

# Mapping function to convert numeric predictions to labels
def map_prediction_label(prediction):
    return "Fake" if prediction == 1 else "Real"

@app.route('/')
def index():
    return redirect(url_for('home'))

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    prediction_label = None
    probability_scores = None
    img_str = None  # Define img_str here to avoid UnboundLocalError

    input_values = {}  # Store input values to preserve them after submission

    if request.method == 'POST':
        # Retrieve form data
        userName = request.form.get('userName')
        screenName = request.form.get('screenName')

        # Initialize variables to store counts
        digitsUserName = 0
        lengthUserName = len(userName)

        digitsUserScreenName = 0
        lengthUserScreenName = len(screenName)

        # Calculate counts for digits in UserName
        for char in userName:
            if char.isdigit():
                digitsUserName += 1

        # Calculate counts for digits in UserScreenName
        for char in screenName:
            if char.isdigit():
                digitsUserScreenName += 1

        # Now you can use these counts in your input_values dictionary
        input_values['digitsUserName'] = digitsUserName
        input_values['lengthUserName'] = lengthUserName
        input_values['digitsUserScreenName'] = digitsUserScreenName
        input_values['lengthUserScreenName'] = lengthUserScreenName
        input_values['userProtected'] = int(request.form.get('userProtected'))
        input_values['userFollowersCount'] = int(request.form.get('userFollowersCount'))
        input_values['userFriendsCount'] = int(request.form.get('userFriendsCount'))
        input_values['userListedCount'] = int(request.form.get('userListedCount'))
        input_values['userLikesCount'] = int(request.form.get('userLikesCount'))
        input_values['userVerified'] = int(request.form.get('userVerified'))
        input_values['userDefaultProfile'] = int(request.form.get('userDefaultProfile'))

        # Add more variables for other input fields 

        # Convert the input to a format suitable for prediction # Adjust as needed

        # Store input values in the dictionary
        input_values['userName'] = userName
        input_values['screenName'] = screenName

         # Convert the input to a format suitable for prediction
        X_new = [
            [
                input_values['digitsUserName'], input_values['lengthUserName'],
                input_values['digitsUserScreenName'], input_values['lengthUserScreenName'],
                input_values['userProtected'], input_values['userFollowersCount'],
                input_values['userFriendsCount'], input_values['userListedCount'],
                input_values['userLikesCount'], input_values['userVerified'],
                input_values['userDefaultProfile']
            ]
        ]  # Adjust as needed

        # Make predictions using the pre-trained model
        prediction = MultiNB.predict(X_new)

        # Map the numeric prediction to labels
        prediction_label = map_prediction_label(prediction[0])

    # Get the probability scores for each class
        probability_scores = MultiNB.predict_proba(X_new)[0]

        # Plot the results and save the plot as a base64-encoded image string
        plt.figure(figsize=(2, 2))
        labels = ['Real', 'Fake']
        colors = ['green', 'red']
        plt.pie(probability_scores, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        #plt.title(f'Prediction Probability for {prediction_label} Account')

        # Save the plot as a base64-encoded image string
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png')
        img_buffer.seek(0)
        img_str = base64.b64encode(img_buffer.read()).decode('utf-8')

    # Render the result on the same page and pass input values and image string
    return render_template('predict.html', prediction=prediction_label, probability_scores=probability_scores, input_values=input_values, img_str=img_str)

@app.route('/home', methods=['GET', 'POST'])
def home():

    return render_template('home.html')

@app.route('/chart')
def chart():

    # Pass the prediction to the template
    return render_template('chart.html')

@app.route('/performance')
def performance():

    return render_template('performance.html')

if __name__ == '__main__':
    app.run(debug=True)