import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle


def preprocess_data(data):
    # Preprocessing and feature engineering steps
    data['Age'].fillna(data['Age'].median(), inplace=True)
    data['Cabin'].fillna('Unknown', inplace=True)
    
    # Fill missing values in CryoSleep with mode
    cryosleep_mode = data['CryoSleep'].mode().iloc[0]
    data['CryoSleep'].fillna(cryosleep_mode, inplace=True)
    data['CryoSleep'] = data['CryoSleep'].astype(int)

    data['TotalSpent'] = data['RoomService'] + data['FoodCourt'] + data['ShoppingMall'] + data['Spa'] + data['VRDeck']
    data['HasCabin'] = data['Cabin'].apply(lambda x: 0 if x == 'Unknown' else 1)
    data['Deck'] = data['Cabin'].apply(lambda x: x[0] if x != 'Unknown' else 'Unknown')

    data.drop(['PassengerId', 'Cabin', 'Name'], axis=1, inplace=True)
    data = pd.get_dummies(data, drop_first=True)

    # Fill missing values in numeric columns with median
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    for column in numeric_columns:
        data[column].fillna(data[column].median(), inplace=True)
    
    # Fill missing values in categorical columns with 'Unknown'
    categorical_columns = data.select_dtypes(include=['object']).columns
    for column in categorical_columns:
        data[column].fillna('Unknown', inplace=True)

    return data


def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'SVM': SVC(random_state=42)
    }

    for model_name, model in models.items():
        accuracy = evaluate_model(model, X_train, X_test, y_train, y_test)
        print(f"{model_name} Accuracy: {accuracy:.2f}")


def optimize_best_model(X_train, y_train):
    param_grid = {
        # Add hyperparameters for grid search
    }

    grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    return best_model


def save_model_to_pickle(model, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)


def generate_submission_file(model, scaler, filename):
    # Load test data and preprocess
    test_data = pd.read_csv('data/test.csv')
    passenger_ids = test_data['PassengerId']
    test_data = preprocess_data(test_data)

    # Scale test data
    test_data = scaler.transform(test_data)

    # Predict using the best model
    test_predictions = model.predict(test_data)

    # Create submission DataFrame
    submission = pd.DataFrame({'PassengerId': passenger_ids, 'Transported': test_predictions})

    # Save submission DataFrame to CSV file
    submission.to_csv(filename, index=False)


def main():
    train_data = pd.read_csv('data/train.csv')
    train_data = preprocess_data(train_data)

    X = train_data.drop('Transported', axis=1)
    y = train_data['Transported']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    train_and_evaluate_models(X_train, X_test, y_train, y_test)

    best_model = optimize_best_model(X_train, y_train)
    print(f"Best model: {best_model}")

    save_model_to_pickle(best_model, 'spaceship_titanic_model.pkl')

    generate_submission_file(best_model, scaler, 'spaceship_titanic_submission.csv')


if __name__ == '__main__':
    main()
