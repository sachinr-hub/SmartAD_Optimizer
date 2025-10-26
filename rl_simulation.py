import pandas as pd
import numpy as np
import tensorflow as tf
# import time # Uncomment if you want to use time.sleep for delay in simulation
import random

# --- Dummy Model and Data Setup (Replace with your actual loaded model and scaled data) ---
# This is a placeholder for your actual model loading and data processing
# In your rl_simulation.py, you would load your trained model here.
class DummyModel:
    def predict(self, data):
        # Simulate predictions - replace with your actual model.predict(data)
        # Ensure the dummy prediction returns an array of the correct shape (e.g., num_ads, 1)
        return np.random.rand(data.shape[0], 1)

# Load your actual model here
# model = tf.keras.models.load_model('your_ctr_model.h5')
model = DummyModel() # Using a dummy model for demonstration

# Define the expected model input columns.
# IMPORTANT: This list MUST EXACTLY match the columns and their order used during your model's training.
# Based on your Dataset_Ads.csv and typical demographic features:
MODEL_INPUT_COLUMNS = [
    'Age', 'Income', 'Gender_Female', 'Gender_Male', 'Gender_Other',
    'Location_Rural', 'Location_Suburban', 'Location_Urban',
    'Ad Type_Banner', 'Ad Type_Native', 'Ad Type_Text', 'Ad Type_Video',
    'Ad Topic_Fashion', 'Ad Topic_Finance', 'Ad Topic_Food', 'Ad Topic_Health', 'Ad Topic_Technology', 'Ad Topic_Travel',
    'Ad Placement_Search Engine', 'Ad Placement_Social Media', 'Ad Placement_Website'
]

# --- Helper Function to Generate All Ad Variants (Important for fixing KeyError) ---
# This function creates a DataFrame representing all possible ad variants
# by combining unique categories from your training data.
# If you have a pre-defined set of 72 ad variants, you should load them here
# and ensure they are one-hot encoded consistently.
def get_all_ad_variants_df():
    # These unique values should come from your training data's categorical columns
    # or directly from Dataset_Ads.csv as we inspected.
    unique_ad_types = ['Banner', 'Video', 'Text', 'Native']
    unique_ad_topics = ['Travel', 'Food', 'Health', 'Fashion', 'Technology', 'Finance']
    unique_ad_placements = ['Social Media', 'Search Engine', 'Website']

    # Create all combinations of ad features
    ad_variants_list = []
    for ad_type in unique_ad_types:
        for ad_topic in unique_ad_topics:
            for ad_placement in unique_ad_placements:
                ad_variants_list.append({
                    'Ad Type': ad_type,
                    'Ad Topic': ad_topic,
                    'Ad Placement': ad_placement
                })

    # Create a DataFrame from all combinations
    all_ad_variants_raw_df = pd.DataFrame(ad_variants_list)

    # One-hot encode the categorical ad features
    # Ensure all possible categories are present as columns, even if not in a specific combination
    all_ad_variants_encoded_df = pd.get_dummies(
        all_ad_variants_raw_df,
        columns=['Ad Type', 'Ad Topic', 'Ad Placement'],
        dtype=int
    )

    # Ensure all expected one-hot encoded columns are present, filling with 0 if not
    # This step is crucial if some combinations don't naturally yield all columns
    for col in MODEL_INPUT_COLUMNS:
        if col.startswith('Ad Type_') or col.startswith('Ad Topic_') or col.startswith('Ad Placement_'):
            if col not in all_ad_variants_encoded_df.columns:
                all_ad_variants_encoded_df[col] = 0

    # Select and reorder only the ad-related columns as they appear in MODEL_INPUT_COLUMNS
    ad_feature_columns = [col for col in MODEL_INPUT_COLUMNS if col.startswith(('Ad Type_', 'Ad Topic_', 'Ad Placement_'))]
    all_ad_variants_encoded_df = all_ad_variants_encoded_df[ad_feature_columns]

    return all_ad_variants_encoded_df

# Pre-compute all ad variants once
ALL_AD_VARIANTS_DF = get_all_ad_variants_df()
NUM_AD_VARIANTS = len(ALL_AD_VARIANTS_DF)
print(f"Number of ad variants generated: {NUM_AD_VARIANTS}")

# --- Corrected predict_ctr_for_ads Function ---
def predict_ctr_for_ads(customer, model):
    """
    Predicts CTR for all ad variants for a given customer.

    Args:
        customer (dict): A dictionary containing the customer's demographics.
                         Example: {'Age': ..., 'Income': ..., 'Gender_Female': True, ...}
        model: The trained CTR prediction model.

    Returns:
        numpy.ndarray: An array of CTR predictions for each ad variant.
    """
    # Convert customer demographics to a DataFrame row
    customer_df_row = pd.DataFrame([customer])

    # Replicate customer demographics for each ad variant
    # This creates a DataFrame where customer info is repeated for all ads
    customer_repeated_df = pd.concat([customer_df_row] * NUM_AD_VARIANTS, ignore_index=True)

    # Combine customer demographics with all ad variants
    # Make sure both DataFrames have matching indices for concat or merge
    input_df_combined = pd.concat([customer_repeated_df, ALL_AD_VARIANTS_DF.reset_index(drop=True)], axis=1)

    # Crucial: Ensure the input_df has all and only the MODEL_INPUT_COLUMNS in the correct order
    # Fill any missing columns (e.g., if a gender/location was not in customer_df_row for a specific customer)
    # with 0s for one-hot encoded features.
    for col in MODEL_INPUT_COLUMNS:
        if col not in input_df_combined.columns:
            input_df_combined[col] = 0
            # For numerical features like Age, Income, if they somehow get missing, you might fill with mean/median
            # For this context, it's assumed Age/Income are always present in 'customer' dict.

    # Select and reorder columns to match the model's expected input order
    input_df = input_df_combined[MODEL_INPUT_COLUMNS]

    # Ensure all columns are numeric (float32 for TensorFlow compatibility)
    # You might need to cast boolean columns to int (0/1) if your model expects floats
    for col in input_df.columns:
        if input_df[col].dtype == 'bool':
            input_df[col] = input_df[col].astype(int)
        elif input_df[col].dtype == 'object': # Should not happen if encoding is correct
             pass # Handle or raise error if unexpected object columns remain

    # Make predictions
    ctr_predictions = model.predict(input_df).flatten()
    return ctr_predictions

# --- Simulation Logic (Simplified from your traceback) ---
# Replace with your actual customer generation, click simulation, and RL logic
def generate_random_customer():
    """Generates a random customer demographic profile."""
    age = random.uniform(-1.0, 1.0) # Assuming age is scaled
    income = random.uniform(-1.0, 1.0) # Assuming income is scaled

    genders = ['Female', 'Male', 'Other']
    gender = random.choice(genders)
    gender_ohe = {f'Gender_{g}': (1 if g == gender else 0) for g in genders}

    locations = ['Rural', 'Suburban', 'Urban']
    location = random.choice(locations)
    location_ohe = {f'Location_{loc}': (1 if loc == location else 0) for loc in locations}

    customer = {
        'Age': age,
        'Income': income,
        **gender_ohe,
        **location_ohe
    }
    return customer

def simulate_click(predicted_ctr, median_ctr):
    """Simulates a click based on predicted CTR."""
    # Clip CTR to [0, 1] range
    clipped_ctr = max(0.0, min(1.0, predicted_ctr))
    return 1 if random.random() < clipped_ctr else 0

def calculate_reward(ctr):
    """Calculate reward with linear normalization: CTR=0→-100, CTR=0.5→0, CTR=1→+100"""
    # Clip CTR to [0, 1] range
    clipped_ctr = max(0.0, min(1.0, ctr))
    return (clipped_ctr - 0.5) * 200

def run_simulation(model, num_steps=10, delay=0.1):
    """Simulates customer interactions and ad optimization."""
    median_ctr = 0.05 # Placeholder, replace with your actual median CTR from training data

    for step in range(1, num_steps + 1):
        print(f"\n*Step {step}/{num_steps}*")

        # 1. Generate a new customer
        customer = generate_random_customer()
        print(f"  *New Customer (demographics):* {customer}")

        # 2. Predict CTR for all ad variants for this customer
        # This is where the KeyError occurred
        ctr_predictions_all_ads = predict_ctr_for_ads(customer, model)

        # 3. Select an ad (e.g., based on highest predicted CTR)
        best_ad_index = np.argmax(ctr_predictions_all_ads)
        predicted_ctr_for_best_ad = ctr_predictions_all_ads[best_ad_index]
        print(f"  *Selected Ad (index {best_ad_index}) with Predicted CTR:* {predicted_ctr_for_best_ad:.4f}")

        # 4. Simulate click
        click = simulate_click(predicted_ctr_for_best_ad, median_ctr)
        print(f"  *Simulated Click:* {'YES' if click == 1 else 'NO'}")

        # 5. Calculate reward using new normalization
        reward = calculate_reward(predicted_ctr_for_best_ad)
        print(f"  *Reward:* {reward:.2f}")

        # 6. (Your RL agent update logic would go here)
        # E.g., agent.update(customer, best_ad_index, click, predicted_ctr_for_best_ad, reward)

        # time.sleep(delay) # Uncomment if you need a delay

# --- Run the Simulation ---
if __name__ == "__main__":
    # Ensure your actual model is loaded before running
    # model = tf.keras.models.load_model('path/to/your_ctr_model.h5')
    run_simulation(model, num_steps=5, delay=0.1) # Run a few steps for demonstration