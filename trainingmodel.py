# this is a guide by team 70 to run the project: use pip install numpy pandas matplotlib seaborn scikit-learn xgboost joblib
#to import all the required files to run the program 

import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt    
import seaborn as sns
from pandas.api.types import CategoricalDtype
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
import joblib

# Load data
df = pd.read_csv('data/Crop_recommendation.csv')

# Data Exploration
print("Dataset Info:")
print(df.info())
print("\nDescriptive Statistics:")
print(df.describe())

# Visualizations
plt.figure(figsize=(12, 8))
for i, col in enumerate(df.drop(columns='label', axis=1).columns, 1):
    plt.subplot(3, 3, i)
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()

# Target distribution
plt.figure(figsize=(10, 8))
counts = df['label'].value_counts()
plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Crop Distribution')
plt.show()

# Feature Engineering
def feature_engineer(df):
    df['NPK'] = (df['N'] + df['P'] + df['K']) / 3
    df['THI'] = df['temperature'] * df['humidity'] / 100
    df['rainfall_level'] = pd.cut(df['rainfall'],
                              bins=[0, 50, 100, 200, 300],
                              labels=['Low', 'Medium', 'High', 'Very High'])
    
    def ph_category(p):
        if p < 5.5:
            return 'Acidic'
        elif p <= 7.5:
            return 'Neutral'
        else:
            return 'Alkaline'
    
    df['ph_category'] = df['ph'].apply(ph_category)
    df['temp_rain_interaction'] = df['temperature'] * df['rainfall']
    df['ph_rain_interaction'] = df['ph'] * df['rainfall']
    return df

df_fe = feature_engineer(df)

# Set proper category order for pH
ph_order = CategoricalDtype(categories=["Acidic", "Neutral", "Alkaline"], ordered=True)
df_fe["ph_category"] = df_fe["ph_category"].astype(ph_order)

# Prepare data
X = df_fe.drop(columns='label', axis=1)
y = df_fe['label']

# Encode target
le_target = LabelEncoder()
y_enc = le_target.fit_transform(y)

# Split data
X_train, X_val, y_train, y_val = train_test_split(
    X, y_enc, 
    test_size=0.2, 
    random_state=42, 
    stratify=y_enc
)

# Preprocessing pipeline
num_cols = X.select_dtypes(include='number').columns
cat_cols = X.select_dtypes(include='category').columns

preprocess = ColumnTransformer([
    ('num', MinMaxScaler(), num_cols),
    ('cat', OrdinalEncoder(), cat_cols)
])

# Create two separate pipelines
# 1. For cross-validation (no early stopping)
cv_pipe = Pipeline([
    ('preprocess', preprocess),
    ('model', XGBClassifier(
        random_state=42,
        eval_metric='merror'  # No early stopping here
    ))
])

# 2. For final training (with early stopping)
final_pipe = Pipeline([
    ('preprocess', preprocess),
    ('model', XGBClassifier(
        random_state=42,
        early_stopping_rounds=10,
        eval_metric='merror'
    ))
])

# Cross-validation (using the simple pipeline)
fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(cv_pipe, X_train, y_train, cv=fold, scoring='accuracy')

print("\nCross-Validation Scores:")
for i, score in enumerate(cv_scores, 1):
    print(f'Fold {i}: {score:.2%}')

# Final training with early stopping
# Need to preprocess the validation set separately
X_train_preprocessed = preprocess.fit_transform(X_train)
X_val_preprocessed = preprocess.transform(X_val)

final_pipe.fit(
    X_train, y_train,
    model__eval_set=[(X_val_preprocessed, y_val)]  # Provide validation set
)

y_pred = final_pipe.predict(X_val)

print("\nValidation Accuracy:", f"{accuracy_score(y_val, y_pred):.2%}")

# Classification report
y_pred_inverse = le_target.inverse_transform(y_pred)
y_val_inverse = le_target.inverse_transform(y_val)

print("\nClassification Report:")
print(classification_report(y_val_inverse, y_pred_inverse))

# Confusion matrix
plt.figure(figsize=(15, 15))
cm = confusion_matrix(y_val_inverse, y_pred_inverse)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=le_target.classes_
)
disp.plot(xticks_rotation='vertical')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

# Feature importance (using the final trained model)
importances = final_pipe.named_steps['model'].feature_importances_
feat_imp_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feat_imp_df)
plt.title('Feature Importances')
plt.tight_layout()
plt.show()

# Save model (save the final_pipe which has early stopping)
joblib.dump(final_pipe, 'crop_recommender.pkl')
joblib.dump(le_target, 'label_encoder.pkl')

print("\nModel and label encoder saved successfully!")

# Load the saved model and label encoder

def solve_farm_problem(farm_data):
    """Function to solve real-life farm problems using the trained model"""
    
    # Convert to DataFrame and apply feature engineering
    farm_df = pd.DataFrame(farm_data)
    farm_df_fe = feature_engineer(farm_df)
    
    # Get prediction (using the final trained model)
    prediction_encoded = final_pipe.predict(farm_df_fe)
    predicted_crop = le_target.inverse_transform(prediction_encoded)[0]
    
    # Get prediction probabilities
    probabilities = final_pipe.predict_proba(farm_df_fe)[0]
    crop_probabilities = pd.DataFrame({
        'Crop': le_target.classes_,
        'Probability': probabilities
    }).sort_values('Probability', ascending=False)
    
    # Analyze sensitivity to rainfall
    rainfall_range = np.linspace(50, 300, 10)
    rainfall_sensitivity = []
    
    for rain in rainfall_range:
        temp_data = farm_data.copy()
        temp_data['rainfall'] = [rain]
        temp_df = pd.DataFrame(temp_data)
        temp_df_fe = feature_engineer(temp_df)
        pred = final_pipe.predict(temp_df_fe)
        rainfall_sensitivity.append(le_target.inverse_transform(pred)[0])
    
    # Output results
    print("\n=== Farm Analysis Results ===")
    print(f"\n1. Recommended crop: {predicted_crop}")
    print("\n2. Top 5 suitable crops with probabilities:")
    print(crop_probabilities.head(5).to_string(index=False))
    
    print("\n3. Rainfall sensitivity analysis:")
    print("At different rainfall levels, the model recommends:")
    for rain, crop in zip(rainfall_range, rainfall_sensitivity):
        print(f"{rain:.0f}mm: {crop}")
    
    # Visualize crop probabilities
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Probability', y='Crop', data=crop_probabilities.head(10))
    plt.title('Top 10 Recommended Crops with Probabilities')
    plt.tight_layout()
    plt.show()
    
    return {
        'recommended_crop': predicted_crop,
        'crop_probabilities': crop_probabilities,
        'rainfall_sensitivity': list(zip(rainfall_range, rainfall_sensitivity))
    }

# Example farm data (can be modified for different scenarios) and datas can be changed to train the model again from the data base.
farm_data = {
    'N': [14],
    'P': [44],
    'K': [28],
    'temperature': [27.5],
    'humidity': [68],
    'ph': [6.8],
    'rainfall': [50]
}

# Solve the problem
solution = solve_farm_problem(farm_data)