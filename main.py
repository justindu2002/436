import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from flask import Flask, request, render_template, redirect, url_for
import os
import random

app = Flask(__name__)

REQUIRED_COLUMNS = [
    "FL_DATE",
    "OP_CARRIER",
    "ORIGIN",
    "CRS_DEP_TIME",
    "DEST",
    "DISTANCE",
]

ALL_FEATURE_COLUMNS = [
    "MONTH",
    "DAY_OF_WEEK",
    "OP_CARRIER",
    "ORIGIN",
    "DEST",
    "CRS_DEP_TIME",
    "CARRIER_DELAY",
    "WEATHER_DELAY",
    "NAS_DELAY",
    "SECURITY_DELAY",
    "LATE_AIRCRAFT_DELAY",
]

FEATURE_COLUMNS = [
    "MONTH",
    "DAY_OF_WEEK",
    "OP_CARRIER",
    "ORIGIN",
    "DEST",
    "CRS_DEP_TIME",
    "CARRIER_DELAY",
    "WEATHER_DELAY",
    "NAS_DELAY",
    "SECURITY_DELAY",
    "LATE_AIRCRAFT_DELAY",
]


def load_data(file):
    if file.filename.endswith(".csv"):
        try:
            return pd.read_csv(file)
        except UnicodeDecodeError:
            file.seek(0)
            return pd.read_csv(file, encoding="ISO-8859-1")
        except pd.errors.ParserError as e:
            print(f"ParserError: {e}")
            file.seek(0)
            return pd.read_csv(file, error_bad_lines=False, warn_bad_lines=True)
    elif file.filename.endswith(".xlsx"):
        try:
            return pd.read_excel(file)
        except Exception as e:
            print(f"Error reading Excel file: {e}")
            raise
    else:
        raise ValueError("Unsupported file format")


def clean_data(df):
    status = "Cleaning data..."
    print(status)
    print(f"Columns in uploaded file: {list(df.columns)}")

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    df_cleaned = df[REQUIRED_COLUMNS].copy()
    df_cleaned = df_cleaned.dropna(subset=REQUIRED_COLUMNS)

    df_cleaned["FL_DATE"] = pd.to_datetime(df_cleaned["FL_DATE"])
    df_cleaned["CRS_DEP_TIME"] = df_cleaned["CRS_DEP_TIME"].astype(int)
    df_cleaned["MONTH"] = df_cleaned["FL_DATE"].dt.month
    df_cleaned["DAY_OF_WEEK"] = df_cleaned["FL_DATE"].dt.dayofweek

    for col in ALL_FEATURE_COLUMNS:
        if col not in df_cleaned.columns:
            df_cleaned[col] = 0

    return df_cleaned, status


def encode_features(df, encoder):
    status = "Encoding features..."
    print(status)
    df["OP_CARRIER"] = encoder.fit_transform(df["OP_CARRIER"])
    df["ORIGIN"] = encoder.fit_transform(df["ORIGIN"])
    df["DEST"] = encoder.fit_transform(df["DEST"])
    return df, status


def train_and_save_model(df):
    status_messages = []
    status = "Training model..."
    print(status)
    status_messages.append(status)

    encoder = LabelEncoder()
    df_encoded, status = encode_features(df, encoder)
    status_messages.append(status)

    X = df_encoded[FEATURE_COLUMNS]

    y = df_encoded["DISTANCE"]

    status = "Splitting data into training and testing sets..."
    print(status)
    status_messages.append(status)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1
    )

    status = "Training RandomForestRegressor model..."
    print(status)
    status_messages.append(status)
    model_rf = RandomForestRegressor(n_estimators=100, random_state=1)
    model_rf.fit(X_train, y_train)

    status = "Evaluating model..."
    print(status)
    status_messages.append(status)
    y_pred_rf = model_rf.predict(X_test)
    mse_rf = mean_squared_error(y_test, y_pred_rf)
    r2_rf = r2_score(y_test, y_pred_rf)
    status = f"Random Forest Regression Mean Squared Error: {mse_rf}"
    print(status)
    status_messages.append(status)
    status = f"Random Forest Regression R^2 Score: {r2_rf}"
    print(status)
    status_messages.append(status)

    status = "Saving model and encoder..."
    print(status)
    status_messages.append(status)
    joblib.dump(model_rf, "random_forest_model.pkl")
    joblib.dump(encoder, "encoder.pkl")
    status = "Model and encoder saved successfully."
    print(status)
    status_messages.append(status)

    return status_messages


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    status_messages = []
    if "file" not in request.files:
        status = "No file part in the request."
        print(status)
        status_messages.append(status)
        return render_template("index.html", status_messages=status_messages)

    file = request.files["file"]
    if file.filename == "":
        status = "No selected file."
        print(status)
        status_messages.append(status)
        return render_template("index.html", status_messages=status_messages)

    if file:
        status = "Processing uploaded file..."
        print(status)
        status_messages.append(status)
        try:
            df = load_data(file)
        except Exception as e:
            status = f"Error loading file: {e}"
            print(status)
            status_messages.append(status)
            return render_template("index.html", status_messages=status_messages)

        try:
            df_cleaned, status = clean_data(df)
            status_messages.append(status)
        except Exception as e:
            status = f"Error cleaning data: {e}"
            print(status)
            status_messages.append(status)
            return render_template("index.html", status_messages=status_messages)

        if not os.path.exists("random_forest_model.pkl") or not os.path.exists(
            "encoder.pkl"
        ):
            status_messages += train_and_save_model(df_cleaned)

        status = "Loading pre-trained model and encoder..."
        print(status)
        status_messages.append(status)
        model_rf = joblib.load("random_forest_model.pkl")
        encoder = joblib.load("encoder.pkl")
        status = "Model and encoder loaded successfully."
        print(status)
        status_messages.append(status)

        df_encoded, status = encode_features(df_cleaned, encoder)
        status_messages.append(status)

        status = "Predicting..."
        print(status)
        status_messages.append(status)
        predictions = model_rf.predict(df_encoded[FEATURE_COLUMNS])
        df_cleaned["Predicted_DELAY"] = predictions

        df_cleaned["Flight_Number"] = [
            f"FL{random.randint(1000, 9999)}" for _ in range(len(df_cleaned))
        ]

        result_df = df_cleaned[
            ["Flight_Number", "FL_DATE", "CRS_DEP_TIME", "Predicted_DELAY"]
        ]

        average_delay = df_cleaned["Predicted_DELAY"].mean()
        max_delay = df_cleaned["Predicted_DELAY"].max()
        min_delay = df_cleaned["Predicted_DELAY"].min()

        stats = {
            "average_delay": average_delay,
            "max_delay": max_delay,
            "min_delay": min_delay,
        }

        status = "Rendering results..."
        print(status)
        status_messages.append(status)
        return render_template(
            "result.html",
            tables=[result_df.to_html(classes="data")],
            titles=result_df.columns.values,
            status_messages=status_messages,
            stats=stats,
        )

    status = "Redirecting to index..."
    print(status)
    status_messages.append(status)
    return redirect(url_for("index"))


if __name__ == "__main__":
    print("Starting Flask app...")
    app.run(debug=True)
