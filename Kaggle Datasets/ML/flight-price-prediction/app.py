# Importing Libraries.
# -------------------------------------

from flask import Flask, request, render_template
from flask_cors import cross_origin

import pickle

import pandas as pd
from sklearn.model_selection import train_test_split


# Initializing Flask web application instance.
# -------------------------------------

app = Flask(__name__)


# Load the trained CatBoost model
# -------------------------------------

with open("best_model.pkl", "rb") as model_file:
    cat_model = pickle.load(model_file)


# Mapping functions for categorical variables
# -------------------------------------


def airline_mapping(airline):
    """
    Map the selected airline to a binary vector representation.

    Parameters:
    - airline (str): The selected airline.

    Returns:
    - List of binary values representing the airline.
    """

    airlines = [
        "Jet Airways",
        "IndiGo",
        "Air India",
        "Multiple carriers",
        "SpiceJet",
        "Vistara",
        "GoAir",
        "Multiple carriers Premium economy",
        "Jet Airways Business",
        "Vistara Premium economy",
        "Trujet",
    ]

    return airlines.index(airline)
    # return [1 if airline == a else 0 for a in airlines]

def airline_mapping(airline):
    """
    Map the selected airline to a binary vector representation.

    Parameters:
    - airline (str): The selected airline.

    Returns:
    - List of binary values representing the airline.
    """

    airlines = [
        "Jet Airways",
        "IndiGo",
        "Air India",
        "Multiple carriers",
        "SpiceJet",
        "Vistara",
        "GoAir",
        "Multiple carriers Premium economy",
        "Jet Airways Business",
        "Vistara Premium economy",
        "Trujet",
    ]

    return airlines.index(airline)
    # return [1 if airline == a else 0 for a in airlines]
    
def source_destination_mapping(location, locations):

    """
    Map the selected source or destination location to a binary vector representation.

    Parameters:
    - location (str): The selected source or destination location.
    - locations (list): List of possible source or destination locations.

    Returns:
    - List of binary values representing the source or destination location.
    """

    return locations.index(location)
    # return [1 if location == loc else 0 for loc in locations]


# -------------------------------------


@app.route("/")
@cross_origin()
def home():
    return render_template("base.html")


@app.route("/predict", methods=["POST"])
@cross_origin()
def predict():

    """
    Handle the flight price prediction based on the form data submitted.

    Returns:
    - Rendered HTML template with the predicted flight price.
    """

    if request.method == "POST":
        # Extract features from form data

        features = {
            "Total_Stops": int(request.form["stops"]),

            "Journey_Day": int(pd.to_datetime(request.form["Dep_Time"]).day),

            "Journey_Month": int(pd.to_datetime(request.form["Dep_Time"]).month),

            "Dep_Hour": int(pd.to_datetime(request.form["Dep_Time"]).hour),

            "Dep_Min": int(pd.to_datetime(request.form["Dep_Time"]).minute),

            "Arrival_Hour": int(
                request.form["Arrival_Time"].split("T")[1].split(":")[0]
            ),

            "Arrival_Min": int(pd.to_datetime(request.form["Arrival_Time"]).minute),

            "Duration_Hour": abs(
                int(request.form["Arrival_Time"].split("T")[1].split(":")[0])
                - int(request.form["Dep_Time"].split("T")[1].split(":")[0])
            ),

            "Duration_Min": abs(
                int(request.form["Arrival_Time"].split("T")[1].split(":")[1])
                - int(request.form["Dep_Time"].split("T")[1].split(":")[1])
            ),

            "Airline": airline_mapping(request.form["airline"]),

            "Source": source_destination_mapping(
                request.form["Source"], 
                ["Chennai", "Delhi", "Kolkata", "Mumbai"]
            ),

            "Destination": source_destination_mapping(
                request.form["Destination"],
                ["Cochin", "Delhi", "Hyderabad", "Kolkata"],
            ),
        }

        # Making predictions using the model.
        prediction = cat_model.predict([list(features.values())])

        output = round(prediction[0], 2)

        return render_template(
            "base.html", prediction_text=f"Your Flight Price is : ₹ {output}"
        )

    return render_template("base.html")


# -------------------------------------

if __name__ == "__main__":
    app.run(port=8000, debug=False)
