import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import numpy as np

# Load the dataset
data = pd.read_csv("data/Dataset_Ads.csv")

# Drop irrelevant columns
data = data.drop(columns=["Clicks", "Click Time", "Conversion Rate"])

# Define categorical and numerical columns
categorical_cols = ["Gender", "Location", "Ad Type", "Ad Topic", "Ad Placement"]
numerical_cols = ["Age", "Income"]

# Apply StandardScaler to numerical columns
numerical_transformer = StandardScaler()
data[numerical_cols] = numerical_transformer.fit_transform(data[numerical_cols])

# One-hot encode categorical columns
data = pd.get_dummies(data, columns=categorical_cols)

# ========== FEATURE ENGINEERING: Add Interaction Features ==========
def add_interaction_features(df):
    """Add interaction features to capture complex patterns"""
    # Age-Income interaction
    df['age_income_interaction'] = df['Age'] * df['Income']
    
    # Gender-Topic interactions (captures preferences like females preferring fashion)
    for gender in ['Gender_Female', 'Gender_Male', 'Gender_Other']:
        for topic in ['Ad Topic_Fashion', 'Ad Topic_Finance', 'Ad Topic_Food', 
                      'Ad Topic_Health', 'Ad Topic_Technology', 'Ad Topic_Travel']:
            if gender in df.columns and topic in df.columns:
                df[f'{gender}_{topic}'] = df[gender] * df[topic]
    
    # Location-Placement interactions (e.g., urban users prefer social media)
    for loc in ['Location_Rural', 'Location_Suburban', 'Location_Urban']:
        for place in ['Ad Placement_Search Engine', 'Ad Placement_Social Media', 
                      'Ad Placement_Website']:
            if loc in df.columns and place in df.columns:
                df[f'{loc}_{place}'] = df[loc] * df[place]
    
    # Age-Gender interactions
    for gender in ['Gender_Female', 'Gender_Male', 'Gender_Other']:
        if gender in df.columns:
            df[f'Age_{gender}'] = df['Age'] * df[gender]
    
    return df

# Apply interaction features
data = add_interaction_features(data)
print(f"Added interaction features. Total features: {len(data.columns)}")

# Split data into DL and RL datasets
# dl_data for model training (first 5000 records)
dl_data = data[:5000]
dl_data.to_csv("data/dl_data.csv", index=False)
print(f"dl_data.csv created with {len(dl_data)} records.")

# rl_data for simulation (remaining records)
rl_data = data[5000:]
rl_data.to_csv("data/rl_data.csv", index=False)
print(f"rl_data.csv created with {len(rl_data)} records.")

# Prepare rl_data for prediction by dropping original ad columns and adding placeholders
ad_types = ["Native", "Text", "Video", "Banner"]
ad_topics = ["Finance", "Food", "Health", "Technology", "Travel", "Fashion"]
ad_placements = ["Social Media", "Website", "Search Engine"]
all_features_ad_columns = ad_types + ad_topics + ad_placements

# Drop the one-hot encoded ad columns from rl_data, as they will be dynamically added for prediction
# This requires reconstructing the column names as they appear after get_dummies for these original columns
# Reconstruct original column names that were dummified related to ads, topics, and placements
cols_to_drop_from_rl = []
for col_prefix, values in zip(["Ad Type", "Ad Topic", "Ad Placement"], [ad_types, ad_topics, ad_placements]):
    for val in values:
        if f"{col_prefix}_{val}" in rl_data.columns:
            cols_to_drop_from_rl.append(f"{col_prefix}_{val}")

rl_data_processed = data[5000:].copy()
rl_data_processed = rl_data_processed.drop(columns=["CTR"]) # CTR is the target for DL, but for RL we predict it

# Drop original ad-related columns as they will be re-added for each prediction
rl_data_processed = rl_data_processed.drop(columns=cols_to_drop_from_rl, errors='ignore')

# Add placeholder columns for new ad features dynamically (without prefix for simpler integration with predict_ctr_for_ads)
for col in all_features_ad_columns:
    rl_data_processed[col] = np.nan

rl_data_processed.to_csv("data/rl_data_processed_for_simulation.csv", index=False)
print(f"rl_data_processed_for_simulation.csv created for RL simulation.")

# Save the StandardScaler for numerical features if needed for new data
import joblib
joblib.dump(numerical_transformer, 'models/numerical_scaler.pkl')
print("Numerical scaler saved to models/numerical_scaler.pkl")

# To get the order of columns that the model expects, save X_train columns (excluding CTR and Unnamed: 0)
# This is crucial for consistent input to the Keras model during prediction
# Create a temporary DataFrame to get the columns for X_dl without "CTR" or "Unnamed: 0"
# This ensures that 'Unnamed: 0' is only dropped if it actually exists in the original df
X_dl_temp = dl_data.drop(columns=["CTR"], errors='ignore')
if 'Unnamed: 0' in X_dl_temp.columns:
    X_dl = X_dl_temp.drop(columns=["Unnamed: 0"])
else:
    X_dl = X_dl_temp
X_dl_columns = X_dl.columns.tolist()
pd.DataFrame(columns=X_dl_columns).to_csv("data/model_input_columns.csv", index=False)
print("model_input_columns.csv created to define the input feature order for the model.")