import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Water Quality Analysis",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("Water Quality Analysis Dashboard")
st.markdown("""
    <div style='background-color: #e6f3ff; padding: 1rem; border-radius: 0.5rem;'>
        This dashboard provides comprehensive analysis of water quality prediction using various machine learning models.
        Explore the data, compare model performances, and make predictions using our trained models.
    </div>
""", unsafe_allow_html=True)

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv('water_data 1.csv')
    # Convert Classification.1 to binary (1 for P.S., 0 for U.S.)
    df['Potability'] = (df['Classification.1'] == 'P.S.').astype(int)
    return df

# Advanced Preprocessing
@st.cache_data
def advanced_preprocessing(df):
    # Select numerical features for prediction
    numerical_features = ['pH', 'E.C', 'TDS', 'CO3', 'HCO3', 'Cl', 'F', 'NO3', 'SO4', 
                         'Na', 'K', 'Ca', 'Mg', 'T.H', 'SAR', 'RSC  meq  / L']
    
    # Handle missing values
    imputer = SimpleImputer(strategy='median')
    df_imputed = pd.DataFrame(imputer.fit_transform(df[numerical_features]), 
                            columns=numerical_features)
    
    X = df_imputed
    y = df['Potability']
    
    # Feature selection
    selector = SelectKBest(score_func=f_classif, k=8)
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()].tolist()
    
    return X_selected, y, selected_features

# Model comparison function with enhanced metrics
@st.cache_data
def compare_models(X_train, X_test, y_train, y_test):
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=50,
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        ),
        'SVM': SVC(
            kernel='rbf',
            C=0.1,
            gamma='scale',
            random_state=42,
            probability=True
        ),
        'Neural Network (sklearn)': MLPClassifier(
            hidden_layer_sizes=(50, 25),
            max_iter=500,
            alpha=0.1,
            learning_rate_init=0.01,
            random_state=42
        )
    }
    
    results = {}
    for name, model in models.items():
        # Add cross-validation
        from sklearn.model_selection import cross_val_score
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        
        # Fit model on training data
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        results[name] = {
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'model': model,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
    
    return results

# Deep Learning Model
@st.cache_data
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

def plot_feature_distributions(df, selected_features):
    n_features = len(selected_features)
    n_cols = 2
    n_rows = (n_features + 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten()
    
    for idx, feature in enumerate(selected_features):
        sns.boxplot(data=df, x='Potability', y=feature, ax=axes[idx])
        axes[idx].set_title(f'{feature} Distribution by Potability')
    
    plt.tight_layout()
    return fig

def main():
    # Load data
    df = load_data()
    
    # Sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Data Overview", "Model Comparison", "Deep Learning", "Make Prediction"])
    
    if page == "Home":
        st.header("Welcome to Water Quality Analysis")
        
        # Key metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Samples", len(df))
        with col2:
            st.metric("Potable Water %", f"{(df['Potability'].mean()*100):.1f}%")
        with col3:
            st.metric("Features", len(df.columns)-1)
        
        # Quick insights
        st.subheader("Quick Insights")
        insights = [
            f"Dataset contains {len(df)} water samples",
            f"{(df['Potability'].mean()*100):.1f}% of samples are potable",
            f"Features include: pH, E.C, TDS, CO3, HCO3, Cl, F, NO3, SO4, Na, K, Ca, Mg, T.H, SAR, RSC",
            "Target variable is 'Classification.1' (P.S.: Potable, U.S.: Unpotable)"
        ]
        for insight in insights:
            st.write(f"• {insight}")
        
        # Feature importance preview
        X, y, selected_features = advanced_preprocessing(df)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        feature_importance = pd.DataFrame({
            'Feature': selected_features,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        st.subheader("Top 5 Most Important Features")
        fig = plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importance.head())
        plt.title('Top 5 Feature Importance')
        st.pyplot(fig)
        
    elif page == "Data Overview":
        st.header("Data Overview")
        
        # Display dataset info
        st.subheader("Dataset Information")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"Total samples: {len(df)}")
            st.write(f"Features: {', '.join(df.columns)}")
        with col2:
            st.write(f"Potable samples: {df['Potability'].sum()}")
            st.write(f"Non-potable samples: {len(df) - df['Potability'].sum()}")
        
        # Display sample data
        st.subheader("Sample Data")
        st.dataframe(df.head())
        
        # Display basic statistics
        st.subheader("Basic Statistics")
        st.dataframe(df.describe())
        
        # Display correlation matrix
        st.subheader("Correlation Matrix")
        numerical_features = ['pH', 'E.C', 'TDS', 'CO3', 'HCO3', 'Cl', 'F', 'NO3', 'SO4', 
                            'Na', 'K', 'Ca', 'Mg', 'T.H', 'SAR', 'RSC  meq  / L', 'Potability']
        plt.figure(figsize=(10, 8))
        sns.heatmap(df[numerical_features].corr(), annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Feature Correlation Matrix')
        st.pyplot(plt)
        
        # Feature distributions
        st.subheader("Feature Distributions by Potability")
        X, y, selected_features = advanced_preprocessing(df)
        fig = plot_feature_distributions(df, selected_features)
        st.pyplot(fig)
        
    elif page == "Model Comparison":
        st.header("Model Comparison")
        
        # Preprocess data
        X, y, selected_features = advanced_preprocessing(df)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Select scaler
        scaler_option = st.selectbox(
            "Select Scaler",
            ["Standard", "Robust", "MinMax"]
        )
        
        # Scale data
        if scaler_option == "Standard":
            scaler = StandardScaler()
        elif scaler_option == "Robust":
            scaler = RobustScaler()
        else:
            scaler = MinMaxScaler()
            
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Compare models
        results = compare_models(X_train_scaled, X_test_scaled, y_train, y_test)
        
        # Display results
        st.subheader("Model Performance Comparison")
        
        # Create metrics table with cross-validation scores
        metrics_df = pd.DataFrame({
            'Model': list(results.keys()),
            'Accuracy': [results[model]['accuracy'] for model in results],
            'ROC AUC': [results[model]['roc_auc'] for model in results],
            'CV Mean': [results[model]['cv_mean'] for model in results],
            'CV Std': [results[model]['cv_std'] for model in results]
        }).sort_values('Accuracy', ascending=False)
        
        st.dataframe(metrics_df.style.format({
            'Accuracy': '{:.3f}',
            'ROC AUC': '{:.3f}',
            'CV Mean': '{:.3f}',
            'CV Std': '{:.3f}'
        }))
        
        # Create bar plot with error bars
        plt.figure(figsize=(10, 6))
        models = list(results.keys())
        accuracies = [results[model]['accuracy'] for model in models]
        cv_stds = [results[model]['cv_std'] for model in models]
        plt.bar(models, accuracies, yerr=cv_stds, capsize=5)
        plt.title(f'Model Accuracy Comparison with {scaler_option} Scaler')
        plt.xticks(rotation=45)
        plt.ylabel('Accuracy')
        plt.tight_layout()
        st.pyplot(plt)
        
        # ROC curves
        st.subheader("ROC Curves")
        plt.figure(figsize=(10, 6))
        for name, result in results.items():
            fpr, tpr, _ = roc_curve(y_test, result['y_pred_proba'])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves for All Models')
        plt.legend()
        st.pyplot(plt)
        
        # Display detailed metrics
        st.subheader("Detailed Metrics")
        for model_name in results.keys():
            with st.expander(f"{model_name} Details"):
                y_pred = results[model_name]['y_pred']
                st.text(classification_report(y_test, y_pred))
                st.write(f"Cross-validation mean score: {results[model_name]['cv_mean']:.3f} (+/- {results[model_name]['cv_std']*2:.3f})")
                
    elif page == "Deep Learning":
        st.header("Deep Learning Model")
        
        # Preprocess data
        X, y, selected_features = advanced_preprocessing(df)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Create and train model
        dl_model = create_deep_learning_model(X_train.shape[1])
        history = dl_model.fit(X_train_scaled, y_train,
                             epochs=50,
                             batch_size=32,
                             validation_split=0.2,
                             verbose=0)
        
        # Display training history
        st.subheader("Training History")
        
        col1, col2 = st.columns(2)
        with col1:
            plt.figure(figsize=(8, 6))
            plt.plot(history.history['accuracy'], label='Training Accuracy')
            plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
            plt.title('Model Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            st.pyplot(plt)
            
        with col2:
            plt.figure(figsize=(8, 6))
            plt.plot(history.history['loss'], label='Training Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title('Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            st.pyplot(plt)
        
        # Display final accuracy
        dl_loss, dl_accuracy = dl_model.evaluate(X_test_scaled, y_test)
        st.metric("Test Accuracy", f"{dl_accuracy:.4f}")
        
        # Display model architecture
        st.subheader("Model Architecture")
        model_summary = []
        for layer in dl_model.layers:
            model_summary.append({
                'Layer': layer.name,
                'Output Shape': layer.output_shape,
                'Parameters': layer.count_params()
            })
        st.dataframe(pd.DataFrame(model_summary))
        
    else:  # Make Prediction
        st.header("Make Prediction")
        
        # Preprocess data
        X, y, selected_features = advanced_preprocessing(df)
        
        st.subheader("Input Parameters")
        st.write("Please enter the values for the following parameters:")
        
        # Create input fields for each feature
        input_values = {}
        col1, col2 = st.columns(2)
        
        for i, feature in enumerate(selected_features):
            with col1 if i < len(selected_features)//2 else col2:
                input_values[feature] = st.number_input(
                    f"Enter {feature}",
                    min_value=float(df[feature].min()),
                    max_value=float(df[feature].max()),
                    value=float(df[feature].mean()),
                    help=f"Range: {df[feature].min():.2f} to {df[feature].max():.2f}"
                )
        
        if st.button("Predict Water Quality"):
            # Scale input
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Create and train model
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_scaled, y)
            
            # Prepare input for prediction
            input_data = np.array([list(input_values.values())])
            input_scaled = scaler.transform(input_data)
            
            # Make prediction
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0]
            
            # Display results with enhanced visualization
            st.subheader("Prediction Results")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "Predicted Water Quality",
                    "Potable" if prediction == 1 else "Not Potable",
                    f"{max(probability)*100:.1f}% confidence"
                )
            
            with col2:
                plt.figure(figsize=(3, 2))
                plt.pie(probability, labels=['Not Potable', 'Potable'], 
                       colors=['#ff9999', '#66b3ff'], autopct='%1.1f%%')
                plt.title('Prediction Probabilities')
                st.pyplot(plt)
            
            # Display feature importance
            st.subheader("Feature Importance for This Prediction")
            feature_importance = pd.DataFrame({
                'Feature': selected_features,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            plt.figure(figsize=(10, 6))
            sns.barplot(x='Importance', y='Feature', data=feature_importance)
            plt.title('Feature Importance')
            st.pyplot(plt)
            
            # Display feature values compared to dataset statistics
            st.subheader("Input Values vs Dataset Statistics")
            comparison_df = pd.DataFrame({
                'Feature': selected_features,
                'Your Input': [input_values[feature] for feature in selected_features],
                'Dataset Mean': [df[feature].mean() for feature in selected_features],
                'Dataset Std': [df[feature].std() for feature in selected_features]
            })
            st.dataframe(comparison_df.style.format({
                'Your Input': '{:.2f}',
                'Dataset Mean': '{:.2f}',
                'Dataset Std': '{:.2f}'
            }))

if __name__ == "__main__":
    main() 