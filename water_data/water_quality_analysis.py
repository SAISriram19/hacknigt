import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv('water_data 1.csv')

# Advanced Preprocessing
def advanced_preprocessing(df):
    # Handle missing values using advanced imputation
    imputer = SimpleImputer(strategy='median')
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    
    # Feature selection using SelectKBest
    X = df_imputed.drop('Potability', axis=1)  # Assuming 'Potability' is the target
    y = df_imputed['Potability']
    selector = SelectKBest(score_func=f_classif, k=8)
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()].tolist()
    
    return X_selected, y, selected_features

# Model comparison function
def compare_models(X_train, X_test, y_train, y_test):
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'SVM': SVC(kernel='rbf', random_state=42),
        'Neural Network (sklearn)': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy
        print(f"\n{name} Results:")
        print(classification_report(y_test, y_pred))
    
    return results

# Deep Learning Model
def create_deep_learning_model(input_dim):
    model = Sequential([
        Dense(128, activation='relu', input_dim=input_dim),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001),
                 loss='binary_crossentropy',
                 metrics=['accuracy'])
    return model

def main():
    # Load and preprocess data
    X, y, selected_features = advanced_preprocessing(df)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features using different scalers
    scalers = {
        'Standard': StandardScaler(),
        'Robust': RobustScaler(),
        'MinMax': MinMaxScaler()
    }
    
    # Compare models with different scalers
    for scaler_name, scaler in scalers.items():
        print(f"\n=== Using {scaler_name} Scaler ===")
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        results = compare_models(X_train_scaled, X_test_scaled, y_train, y_test)
        
        # Plot results
        plt.figure(figsize=(10, 6))
        plt.bar(results.keys(), results.values())
        plt.title(f'Model Comparison with {scaler_name} Scaler')
        plt.xticks(rotation=45)
        plt.ylabel('Accuracy')
        plt.tight_layout()
        plt.savefig(f'model_comparison_{scaler_name.lower()}.png')
        plt.close()
    
    # Deep Learning with best scaler
    best_scaler = StandardScaler()
    X_train_dl = best_scaler.fit_transform(X_train)
    X_test_dl = best_scaler.transform(X_test)
    
    # Create and train deep learning model
    dl_model = create_deep_learning_model(X_train.shape[1])
    history = dl_model.fit(X_train_dl, y_train,
                          epochs=50,
                          batch_size=32,
                          validation_split=0.2,
                          verbose=0)
    
    # Evaluate deep learning model
    dl_loss, dl_accuracy = dl_model.evaluate(X_test_dl, y_test)
    print(f"\nDeep Learning Model Accuracy: {dl_accuracy:.4f}")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig('deep_learning_history.png')
    plt.close()

if __name__ == "__main__":
    main() 