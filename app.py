import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from itertools import product
from tensorflow.keras.models import load_model
import tensorflow as tf
import joblib
import base64
import os
from datetime import datetime
from user_auth import require_authentication

# ------------------- PAGE CONFIG -------------------
st.set_page_config(
    layout="wide", 
    page_title="SmartAd Optimizer",
    page_icon="🎯",
    initial_sidebar_state="expanded"
)

# ------------------- GLOBAL DEFINITIONS -------------------
ad_types = ["Native", "Text", "Video", "Banner"]
ad_topics = ["Finance", "Food", "Health", "Technology", "Travel", "Fashion"]
ad_placements = ["Social Media", "Website", "Search Engine"]

# Background image removed per request

# ------------------- LOAD ARTIFACTS -------------------
@st.cache_resource
def load_artifacts():
    # Load CTR model
    try:
        model = load_model("models/ctr_prediction_model.keras")

    except:
        class DummyModel:
            def predict(self, data, verbose=0): return np.random.rand(data.shape[0], 1)
        model = DummyModel()

    # Load scaler
    try:
        scaler = joblib.load("models/numerical_scaler.pkl")
    except:
        class DummyScaler:
            def transform(self, data): return data
        scaler = DummyScaler()

    # Load model input columns
    try:
        model_input_columns = pd.read_csv("data/model_input_columns.csv").columns.tolist()
    except:
        model_input_columns = [
            'Age', 'Income', 'Gender_Female', 'Gender_Male', 'Gender_Other',
            'Location_Rural', 'Location_Suburban', 'Location_Urban',
            'Ad Type_Banner', 'Ad Type_Native', 'Ad Type_Text', 'Ad Type_Video',
            'Ad Topic_Fashion', 'Ad Topic_Finance', 'Ad Topic_Food',
            'Ad Topic_Health', 'Ad Topic_Technology', 'Ad Topic_Travel',
            'Ad Placement_Search Engine', 'Ad Placement_Social Media', 'Ad Placement_Website'
        ]

    ad_variants = list(product(ad_types, ad_topics, ad_placements))
    num_ads = len(ad_variants)

    # Init Thompson Sampling params
    if "alpha" not in st.session_state: st.session_state.alpha = np.ones(num_ads)
    if "beta" not in st.session_state: st.session_state.beta = np.ones(num_ads)

    # Median CTR from data
    try:
        dl_data = pd.read_csv("data/dl_data.csv")
        median_ctr = np.median(dl_data["CTR"])
    except:
        median_ctr = 0.05

    return model, scaler, model_input_columns, ad_variants, num_ads, median_ctr

# ------------------- HELPERS -------------------
def build_input(user_age, user_income, user_gender, user_location,
                ad_type, ad_topic, ad_place, scaler, model_input_columns):
    # Base input
    input_data = {
        'Age': user_age,
        'Income': user_income,
        'Gender_Female': user_gender == 'Female',
        'Gender_Male': user_gender == 'Male',
        'Gender_Other': user_gender == 'Other',
        'Location_Rural': user_location == 'Rural',
        'Location_Suburban': user_location == 'Suburban',
        'Location_Urban': user_location == 'Urban'
    }
    # One-hot ads
    for t in ad_types: input_data[f'Ad Type_{t}'] = int(t == ad_type)
    for tp in ad_topics: input_data[f'Ad Topic_{tp}'] = int(tp == ad_topic)
    for pl in ad_placements: input_data[f'Ad Placement_{pl}'] = int(pl == ad_place)

    df = pd.DataFrame([input_data])
    df[['Age', 'Income']] = scaler.transform(df[['Age', 'Income']])

    for col in model_input_columns:
        if col not in df.columns: df[col] = 0

    df = df[model_input_columns]
    df = df.astype(float)
    return df.to_numpy(dtype=np.float64).reshape(1, df.shape[1], 1)

def predict_ctr(input_array, model):
    return model.predict(input_array, verbose=0)[0][0]

def thompson_sampling_select():
    sampled = np.random.beta(st.session_state.alpha, st.session_state.beta)
    return np.argmax(sampled)

def update_thompson(ad_index, ctr, median_ctr):
    clicked = ctr > median_ctr
    if clicked: st.session_state.alpha[ad_index] += 1
    else: st.session_state.beta[ad_index] += 1
    return clicked

def plot_ctr_trend(ctrs):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(ctrs) + 1), ctrs, marker="o")
    ax.set_title("Cumulative CTR over Simulation Steps")
    ax.set_xlabel("Step")
    ax.set_ylabel("CTR")
    ax.grid(True)
    st.pyplot(fig)

# --- RL Simulation Helpers ---
def generate_customer_data_for_sim(data: pd.DataFrame):
    """Sample one customer row and return only demographic features (no ad OHE)."""
    item = data.sample(n=1, ignore_index=True)
    customer_features = {}
    for col in item.columns:
        if not (col.startswith('Ad Type_') or col.startswith('Ad Topic_') or col.startswith('Ad Placement_')):
            customer_features[col] = item[col].iloc[0]
    return customer_features

def predict_ctr_for_all_ads_sim(customer_features: dict, model, model_input_columns, ad_variants):
    """Build batched inputs for all ad variants and predict CTRs."""
    # Base customer row
    customer_df_row = pd.DataFrame([customer_features])

    # Replicate for each ad variant
    customer_repeated_df = pd.concat([customer_df_row] * len(ad_variants), ignore_index=True)

    # Build ad OHE columns for all variants
    ad_rows = []
    for ad_type, ad_topic, ad_place in ad_variants:
        row = {f'Ad Type_{ad_type}': 1, f'Ad Topic_{ad_topic}': 1, f'Ad Placement_{ad_place}': 1}
        ad_rows.append(row)
    ad_df = pd.DataFrame(ad_rows).fillna(0)

    # Combine customer and ad features
    combined = pd.concat([customer_repeated_df.reset_index(drop=True), ad_df.reset_index(drop=True)], axis=1)

    # Ensure all expected model columns are present
    for col in model_input_columns:
        if col not in combined.columns:
            combined[col] = 0
    combined = combined[model_input_columns]

    # Cast booleans to ints
    for col in combined.columns:
        if combined[col].dtype == 'bool':
            combined[col] = combined[col].astype(int)

    # Reshape for Conv1D
    n_rows = combined.shape[0]
    input_array = combined.to_numpy(dtype=np.float64).reshape(n_rows, len(model_input_columns), 1)
    ctr_predictions = model.predict(input_array, verbose=0).flatten()
    return np.array(ctr_predictions)

def thompson_sampling_select_sim():
    sampled = np.random.beta(st.session_state.alpha, st.session_state.beta)
    return np.argmax(sampled)

def update_thompson_and_reward(ad_index: int, ctr_pred: float, median_ctr: float):
    """Update TS parameters based on simulated click; return (clicked, reward).

    Click is sampled from Bernoulli(p=ctr_pred clipped to [0,1]) for a more realistic signal.
    """
    p = float(np.clip(ctr_pred, 0.0, 1.0))
    clicked = (np.random.rand() < p)
    if clicked:
        st.session_state.alpha[ad_index] += 1
        reward = float(ctr_pred) * 1000.0
    else:
        st.session_state.beta[ad_index] += 1
        reward = float(-100.0)
    return clicked, reward

def update_thompson_and_reward_enhanced(ad_index: int, ctr_pred: float, median_ctr: float):
    """Optimized reward for realistic CTR (0.03-0.10). Balanced to avoid extreme negative cumulative rewards."""
    p = float(np.clip(ctr_pred, 0.0, 1.0))
    clicked = (np.random.rand() < p)
    
    if clicked:
        st.session_state.alpha[ad_index] += 1
        # High reward for successful clicks
        reward = ctr_pred * 2000
    else:
        st.session_state.beta[ad_index] += 1
        # Small penalty proportional to missed opportunity
        reward = -ctr_pred * 50
    
    return clicked, reward

def show_detailed_statistics(ad_variants):
    """Show detailed Thompson Sampling statistics"""
    if 'alpha' not in st.session_state or 'beta' not in st.session_state:
        st.warning("No simulation data available yet.")
        return
    
    st.markdown("### 📊 Detailed Thompson Sampling Statistics")
    
    alpha = st.session_state.alpha
    beta = st.session_state.beta
    
    # Calculate statistics
    success_rates = alpha / (alpha + beta)
    total_trials = alpha + beta - 2  # Subtract initial values
    confidence_intervals = 1.96 * np.sqrt(alpha * beta / ((alpha + beta)**2 * (alpha + beta + 1)))
    
    # Create comprehensive statistics DataFrame
    stats_df = pd.DataFrame({
        'Ad_Type': [ad[0] for ad in ad_variants],
        'Ad_Topic': [ad[1] for ad in ad_variants],
        'Ad_Placement': [ad[2] for ad in ad_variants],
        'Success_Rate': success_rates,
        'Total_Trials': total_trials,
        'Successes': alpha - 1,  # Subtract initial value
        'Failures': beta - 1,    # Subtract initial value
        'Confidence_Interval': confidence_intervals
    })
    
    # Sort by success rate
    stats_df = stats_df.sort_values('Success_Rate', ascending=False).reset_index(drop=True)
    
    # Display top performers
    st.markdown("#### 🏆 Top 10 Performing Ad Variants")
    st.dataframe(stats_df.head(10), use_container_width=True)
    
    # Visualization
    col1, col2 = st.columns(2)
    
    with col1:
        # Success rate distribution
        fig_dist = px.histogram(
            stats_df, 
            x='Success_Rate', 
            nbins=20,
            title="Success Rate Distribution"
        )
        st.plotly_chart(fig_dist, use_container_width=True)
    
    with col2:
        # Trials vs Success Rate scatter
        fig_scatter = px.scatter(
            stats_df, 
            x='Total_Trials', 
            y='Success_Rate',
            title="Trials vs Success Rate",
            hover_data=['Ad_Type', 'Ad_Topic']
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

def build_training_row(customer_features: dict, ad_tuple, clicked: bool, predicted_ctr: float, model_input_columns):
    """Create a single-row DataFrame matching model_input_columns plus CTR target.

    We use clicked (0/1) as an online label proxy for CTR.
    """
    ad_type, ad_topic, ad_place = ad_tuple
    row = dict(customer_features)
    # One-hot encode selected ad
    row.update({
        f'Ad Type_{ad_type}': 1,
        f'Ad Topic_{ad_topic}': 1,
        f'Ad Placement_{ad_place}': 1,
    })
    # Ensure all expected columns
    for col in model_input_columns:
        if col not in row:
            row[col] = 0
    # Target
    row['CTR'] = 1.0 if clicked else 0.0
    row['predicted_ctr'] = float(predicted_ctr)
    row['timestamp'] = datetime.utcnow().isoformat()
    return pd.DataFrame([row])

def log_online_event(row_df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    header = not os.path.exists(path)
    row_df.to_csv(path, mode='a', header=header, index=False)

def retrain_from_logs(model_input_columns):
    """Retrain/fine-tune CTR model using base data + logged online events.

    Saves a new model to models/ctr_prediction_model.keras and refreshes cached artifacts.
    """
    base_path = "data/dl_data.csv"
    logs_path = "data/online_events.csv"

    if not os.path.exists(base_path) and not os.path.exists(logs_path):
        st.warning("No training data available (dl_data.csv or online_events.csv)")
        return False

    dfs = []
    if os.path.exists(base_path):
        dfs.append(pd.read_csv(base_path))
    if os.path.exists(logs_path):
        logs = pd.read_csv(logs_path)
        # Keep only model columns + CTR
        keep_cols = [c for c in logs.columns if c in model_input_columns or c == 'CTR']
        logs = logs[keep_cols]
        # Fill missing columns
        for col in model_input_columns:
            if col not in logs.columns:
                logs[col] = 0
        # Reorder
        logs = logs[model_input_columns + ['CTR']]
        dfs.append(logs)

    data = pd.concat(dfs, ignore_index=True)
    # Drop any unexpected columns like 'Unnamed: 0'
    data = data.drop(columns=[c for c in data.columns if c not in model_input_columns + ['CTR']], errors='ignore')

    # Prepare arrays with proper data validation
    try:
        # Ensure all columns are numeric and handle any non-numeric values
        for col in model_input_columns:
            if col in data.columns:
                # Convert boolean columns to int
                if data[col].dtype == 'bool':
                    data[col] = data[col].astype(int)
                # Fill any NaN values with 0
                data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)
        
        # Convert CTR column
        data['CTR'] = pd.to_numeric(data['CTR'], errors='coerce').fillna(0)
        
        # Extract arrays
        X = data[model_input_columns].astype(np.float64).to_numpy()
        y = data['CTR'].astype(np.float64).to_numpy()
        X_reshaped = X.reshape(X.shape[0], X.shape[1], 1)
        
    except Exception as e:
        st.error(f"Data conversion error: {e}")
        return False

    # Load the existing model and fine-tune
    try:
        model = load_model("models/ctr_prediction_model.keras")
    except Exception:
        # Fallback tiny model if missing
        model = tf.keras.Sequential([
            tf.keras.layers.Conv1D(filters=16, kernel_size=3, activation='relu', input_shape=(X.shape[1], 1)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='mse')
    model.fit(X_reshaped, y, epochs=5, batch_size=32, validation_split=0.1, verbose=0)
    model.save("models/ctr_prediction_model.keras")

    # Refresh cached artifacts
    try:
        load_artifacts.clear()
    except Exception:
        pass
    st.success("CTR model retrained and reloaded from logged events.")
    return True

# ------------------- MAIN APP -------------------
def main():
    # Background removed

    # Custom header with styling
    st.markdown(
        """
        <div style="
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            text-align: center;
        ">
            <h1 style="color: white; margin: 0; font-size: 3rem;">🎯 SMART AD OPTIMIZER</h1>
            <p style="color: white; margin: 0.5rem 0 0 0; font-size: 1.2rem; opacity: 0.9;">
                AI-Powered Advertisement Optimization Platform
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Authentication gate: show login/registration until user is authenticated
    if not require_authentication():
        return
    
    # Note: Removed sidebar Navigation and page routing to simplify the app page.
    # The app now directly shows the RL Simulation content below.
    user_info = st.session_state.get('user_info', {})
    is_admin = user_info.get('role') == 'admin'
    
    # Default: RL Simulation page
    model, scaler, model_input_columns, ad_variants, num_ads, median_ctr = load_artifacts()

    # Load RL simulation data
    try:
        rl_data = pd.read_csv("data/rl_data_processed_for_simulation.csv")
    except Exception as e:
        # Fallback dummy demographics if file not available
        demo_cols = [c for c in model_input_columns if not c.startswith(('Ad Type_', 'Ad Topic_', 'Ad Placement_'))]
        rl_data = pd.DataFrame(np.random.randn(100, len(demo_cols)), columns=demo_cols)

    # Enhanced sidebar with rich controls
    with st.sidebar:
        st.markdown("### ⚙️ Simulation Settings")
        
        # Simulation parameters with better UX
        num_steps = st.slider(
            "🔢 Number of simulation steps", 
            min_value=1, 
            max_value=50, 
            value=5,
            help="Number of customer interactions to simulate"
        )
        
        delay = st.slider(
            "⏱️ Delay between steps (seconds)", 
            min_value=0.0, 
            max_value=5.0, 
            value=1.0,
            step=0.5,
            help="Time delay between each simulation step"
        )
        
        st.markdown("---")
        
        # Model management section
        # Note: Removed '🤖 Model Management' and model info from sidebar per request
        
        # Real-time metrics section
        st.markdown("### 📈 Live Metrics")
        reward_plot = st.empty()
        
        # Thompson Sampling stats
        if 'alpha' in st.session_state and 'beta' in st.session_state:
            total_trials = st.session_state.alpha.sum() + st.session_state.beta.sum() - 2 * len(ad_variants)
            if total_trials > 0:
                st.metric("Total Trials", int(total_trials))
                
                # Best performing ad variant
                success_rates = st.session_state.alpha / (st.session_state.alpha + st.session_state.beta)
                best_variant_idx = np.argmax(success_rates)
                st.metric(
                    "Best Variant Success Rate", 
                    f"{success_rates[best_variant_idx]:.2%}"
                )
        
        st.markdown("---")
        
        # Quick actions
        st.markdown("### 🚀 Quick Actions")
        
        if st.button("📊 Show Stats", key="show_stats_button", help="Show detailed Thompson Sampling statistics"):
            st.session_state.show_detailed_stats = True

    # Main simulation section with enhanced UI
    st.markdown("### 🎮 Reinforcement Learning Simulation")
    st.markdown(
        "This simulation uses **Thompson Sampling** to optimize ad selection based on customer interactions. "
        "The algorithm learns which ad variants perform best for different customer segments."
    )
    
    # Simulation controls
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        start_simulation = st.button(
            "🚀 Start RL Simulation", 
            key="start_simulation_button",
            type="primary",
            use_container_width=True
        )
    
    with col2:
        if st.button("⏸️ Pause", key="pause_button", disabled=True):
            pass  # Placeholder for pause functionality
    
    with col3:
        if st.button("⏹️ Stop", key="stop_button", disabled=True):
            pass  # Placeholder for stop functionality
    
    # Show detailed stats if requested
    if st.session_state.get('show_detailed_stats', False):
        show_detailed_statistics(ad_variants)
        st.session_state.show_detailed_stats = False
    
    if start_simulation:
        st.markdown("### 🔄 Simulation in Progress")
        
        # Initialize tracking variables
        rewards_over_time = []
        ctr_over_time = []
        selected_ads_count = np.zeros(num_ads)
        
        # Create containers for real-time updates
        progress_container = st.container()
        metrics_container = st.container()
        results_container = st.container()

        # Progress tracking
        overall_progress = progress_container.progress(0)
        step_info = progress_container.empty()
        
        for step in range(num_steps):
            # Update overall progress
            progress = (step + 1) / num_steps
            overall_progress.progress(progress)
            step_info.info(f"🔄 Processing Step {step+1}/{num_steps}")
            
            with st.expander(f"📊 Step {step+1}/{num_steps} - Customer Interaction", expanded=(step < 3)):
                # 1) New customer
                customer = generate_customer_data_for_sim(rl_data)

                # Enhanced customer display
                display_customer = dict(customer)
                if 'Age' in display_customer and 'Income' in display_customer:
                    try:
                        inv = scaler.inverse_transform(np.array([[display_customer['Age'], display_customer['Income']]]))
                        display_customer['Age'] = float(inv[0, 0])
                        display_customer['Income'] = float(inv[0, 1])
                    except Exception:
                        pass
                
                # Customer profile card
                customer_data = {k: v for k, v in display_customer.items() if k in [
                    'Age','Income','Gender_Female','Gender_Male','Gender_Other','Location_Rural','Location_Suburban','Location_Urban'
                ]}
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**👤 Customer Profile:**")
                    
                    # Display Age and Income
                    if 'Age' in customer_data:
                        st.write(f"• Age: {customer_data['Age']:.0f}")
                    if 'Income' in customer_data:
                        st.write(f"• Income: ${customer_data['Income']:.0f}")
                    
                    # Display Gender
                    if customer_data.get('Gender_Female', 0) == 1:
                        st.write(f"• Gender: Female")
                    elif customer_data.get('Gender_Male', 0) == 1:
                        st.write(f"• Gender: Male")
                    elif customer_data.get('Gender_Other', 0) == 1:
                        st.write(f"• Gender: Other")
                    
                    # Display Location
                    if customer_data.get('Location_Rural', 0) == 1:
                        st.write(f"• Location: Rural")
                    elif customer_data.get('Location_Suburban', 0) == 1:
                        st.write(f"• Location: Suburban")
                    elif customer_data.get('Location_Urban', 0) == 1:
                        st.write(f"• Location: Urban")

                # 2) Predict CTR for all ads
                ctr_predictions = predict_ctr_for_all_ads_sim(customer, model, model_input_columns, ad_variants)

                # 3) Select ad via Thompson Sampling
                best_ad_index = thompson_sampling_select_sim()
                best_ad = ad_variants[best_ad_index]
                ctr_pred = float(ctr_predictions[best_ad_index])
                selected_ads_count[best_ad_index] += 1

                with col2:
                    # Ad recommendation card
                    st.markdown(
                        f"""
                        <div style="
                            background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
                            padding: 15px;
                            border-radius: 8px;
                            color: white;
                            margin: 10px 0;
                        ">
                            <h4 style="margin: 0; color: white;">🎯 Selected Ad</h4>
                            <p style="margin: 5px 0;">📺 {best_ad[0]} | 🏷️ {best_ad[1]} | 📍 {best_ad[2]}</p>
                            <p style="margin: 5px 0; font-size: 18px;"><strong>CTR: {ctr_pred:.4f}</strong></p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                # 4) Update Thompson Sampling + compute reward
                clicked, reward = update_thompson_and_reward_enhanced(best_ad_index, ctr_pred, median_ctr)
                rewards_over_time.append(reward)
                ctr_over_time.append(ctr_pred)
                
                # Show interaction result
                col1, col2, col3 = st.columns(3)
                with col1:
                    if clicked:
                        st.success(f"✅ Click! Reward: +{reward:.2f}")
                    else:
                        st.error(f"❌ No Click. Reward: {reward:.2f}")
                
                with col2:
                    st.metric("Cumulative Reward", f"{sum(rewards_over_time):.2f}")
                
                with col3:
                    st.metric("Avg CTR", f"{np.mean(ctr_over_time):.4f}")

                # Log online event for potential retraining
                try:
                    row = build_training_row(customer, best_ad, clicked, ctr_pred, model_input_columns)
                    log_online_event(row, "data/online_events.csv")
                except Exception as e:
                    st.warning(f"⚠️ Logging failed: {e}")

            # Update live metrics in sidebar
            with st.sidebar:
                # Reward trend chart
                if len(rewards_over_time) > 1:
                    fig_reward = go.Figure()
                    fig_reward.add_trace(go.Scatter(
                        x=list(range(1, len(rewards_over_time) + 1)),
                        y=rewards_over_time,
                        mode='lines+markers',
                        name='Reward',
                        line=dict(color='blue', width=2)
                    ))
                    fig_reward.update_layout(
                        title="Reward Trend",
                        xaxis_title="Step",
                        yaxis_title="Reward",
                        height=200,
                        margin=dict(l=0, r=0, t=30, b=0)
                    )
                    reward_plot.plotly_chart(fig_reward, use_container_width=True)
            
            time.sleep(delay)

        # Clear progress indicators
        overall_progress.empty()
        step_info.empty()
        
        # Simulation complete - show comprehensive results
        st.success("🎉 Simulation Complete!")
        
        # Final results dashboard
        st.markdown("### 📊 Simulation Results Dashboard")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Reward",
                f"{sum(rewards_over_time):.2f}",
                delta=f"{rewards_over_time[-1]:.2f}" if rewards_over_time else None
            )
        
        with col2:
            st.metric(
                "Average CTR",
                f"{np.mean(ctr_over_time):.4f}",
                delta=f"{(np.mean(ctr_over_time) - median_ctr):.4f}"
            )
        
        with col3:
            total_clicks = sum(1 for r in rewards_over_time if r > 0)
            click_rate = total_clicks / len(rewards_over_time) if rewards_over_time else 0
            st.metric(
                "Click Rate",
                f"{click_rate:.2%}",
                delta=None
            )
        
        with col4:
            best_reward = max(rewards_over_time) if rewards_over_time else 0
            st.metric(
                "Best Reward",
                f"{best_reward:.2f}",
                delta=None
            )
        
        # Detailed analytics
        col1, col2 = st.columns(2)
        
        with col1:
            # Reward over time chart
            fig_final_reward = px.line(
                x=range(1, len(rewards_over_time) + 1),
                y=rewards_over_time,
                title="Reward Evolution Over Time",
                labels={'x': 'Step', 'y': 'Reward'}
            )
            fig_final_reward.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig_final_reward, use_container_width=True)
        
        with col2:
            # Ad selection frequency
            if len(selected_ads_count) > 0:
                top_ads_indices = np.argsort(selected_ads_count)[-10:][::-1]
                top_ads_labels = [f"{ad_variants[i][0]}-{ad_variants[i][1]}" for i in top_ads_indices]
                top_ads_counts = selected_ads_count[top_ads_indices]
                
                fig_ads = px.bar(
                    x=top_ads_labels,
                    y=top_ads_counts,
                    title="Most Selected Ad Variants",
                    labels={'x': 'Ad Variant', 'y': 'Selection Count'}
                )
                fig_ads.update_xaxes(tickangle=45)
                st.plotly_chart(fig_ads, use_container_width=True)
        
        # Thompson Sampling learning progress
        if 'alpha' in st.session_state and 'beta' in st.session_state:
            st.markdown("### 🧠 Thompson Sampling Learning Progress")
            
            success_rates = st.session_state.alpha / (st.session_state.alpha + st.session_state.beta)
            confidence_intervals = np.sqrt(st.session_state.alpha * st.session_state.beta / 
                                         ((st.session_state.alpha + st.session_state.beta)**2 * 
                                          (st.session_state.alpha + st.session_state.beta + 1)))
            
            # Show top performing variants
            top_performers = np.argsort(success_rates)[-10:][::-1]
            
            performance_data = pd.DataFrame({
                'Ad_Variant': [f"{ad_variants[i][0]}-{ad_variants[i][1]}-{ad_variants[i][2]}" for i in top_performers],
                'Success_Rate': success_rates[top_performers],
                'Confidence': confidence_intervals[top_performers],
                'Trials': (st.session_state.alpha + st.session_state.beta - 2)[top_performers]
            })
            
            st.dataframe(performance_data, use_container_width=True)
        
        # Export simulation results
        st.markdown("### 💾 Export Simulation Data")
        
        simulation_results = pd.DataFrame({
            'Step': range(1, len(rewards_over_time) + 1),
            'Reward': rewards_over_time,
            'CTR': ctr_over_time,
            'Cumulative_Reward': np.cumsum(rewards_over_time)
        })
        
        csv_results = simulation_results.to_csv(index=False)
        st.download_button(
            label="📥 Download Simulation Results",
            data=csv_results,
            file_name=f"simulation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

# ------------------- RUN -------------------
if __name__ == "__main__":
    # Initialize session state variables
    if 'show_detailed_stats' not in st.session_state:
        st.session_state.show_detailed_stats = False
    
    main()
