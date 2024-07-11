import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

def load_data(file_path):
    return pd.read_csv(file_path)

def clean_data(df):
    important_columns = ['FL_DATE', 'OP_CARRIER', 'ORIGIN', 'CRS_DEP_TIME', 'DEP_DELAY', 'DEST', 'DISTANCE', 'CARRIER_DELAY', 'WEATHER_DELAY', 'NAS_DELAY', 'SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY']
    df_cleaned = df[important_columns]
    df_cleaned = df_cleaned.dropna(subset=important_columns)

    #sampling the csv data
    df_cleaned = df_cleaned.sample(n=100000, random_state=1)

    #data type check
    df_cleaned['FL_DATE'] = pd.to_datetime(df_cleaned['FL_DATE'])
    df_cleaned['CRS_DEP_TIME'] = df_cleaned['CRS_DEP_TIME'].astype(int)
    df_cleaned['DEP_DELAY'] = df_cleaned['DEP_DELAY'].astype(float)
    df_cleaned['MONTH'] = df_cleaned['FL_DATE'].dt.month
    df_cleaned['DAY_OF_WEEK'] = df_cleaned['FL_DATE'].dt.dayofweek
    return df_cleaned

#label encoding for airlines and airports
def encode_features(df):
    encoder = LabelEncoder()
    df['OP_CARRIER'] = encoder.fit_transform(df['OP_CARRIER'])
    df['ORIGIN'] = encoder.fit_transform(df['ORIGIN'])
    df['DEST'] = encoder.fit_transform(df['DEST'])
    return df

def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def train_random_forest_regressor(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=1)
    model.fit(X_train, y_train)
    return model

def train_random_forest_classifier(X_train, y_train):
    model = RandomForestClassifier(random_state=1)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, regression=True):
    y_pred = model.predict(X_test)
    if regression:
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return mse, r2
    else:
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy

def main():
    #load data
    df = load_data('2018.csv')

    #clean data
    df_cleaned = clean_data(df)
    df_encoded = encode_features(df_cleaned)

    #model features
    X = df_encoded[['MONTH', 'DAY_OF_WEEK', 'OP_CARRIER', 'ORIGIN', 'DEST', 'CRS_DEP_TIME', 'CARRIER_DELAY', 'WEATHER_DELAY', 'NAS_DELAY', 'SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY']]
    #model target
    y = df_encoded['DEP_DELAY']
    
    #split into training and testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    
    #train lr model
    model_lr = train_linear_regression(X_train, y_train)
    #evaluate lr model
    mse_lr, r2_lr = evaluate_model(model_lr, X_test, y_test)
    print(f"Linear Regression Mean Squared Error: {mse_lr}")
    print(f"Linear Regression R^2 Score: {r2_lr}")
    
    #train randomforest regressor model
    model_rf = train_random_forest_regressor(X_train, y_train)
    #evaluate model
    mse_rf, r2_rf = evaluate_model(model_rf, X_test, y_test)
    print(f"Random Forest Regression Mean Squared Error: {mse_rf}")
    print(f"Random Forest Regression R^2 Score: {r2_rf}")
    
    #Classification model
    #model target
    y_class = (df_encoded['DEP_DELAY'] > 0).astype(int)
    
    #split dataset
    X_train, X_test, y_train_class, y_test_class = train_test_split(X, y_class, test_size=0.2, random_state=1)
    
    #train and evalute model
    model_class = train_random_forest_classifier(X_train, y_train_class)
    class_accuracy = evaluate_model(model_class, X_test, y_test_class, regression=False)
    print(f'Classification Accuracy: {class_accuracy}')

if __name__ == "__main__":
    main()
