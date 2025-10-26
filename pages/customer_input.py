import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import os
from itertools import product
import sys
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime

# Add parent directory to path to import user_auth
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from user_auth import require_authentication

# ------------------- LOAD ARTIFACTS -------------------
@st.cache_resource
def load_artifacts():
    # Load CTR model
    try:
        model = load_model("models/ctr_prediction_model.keras")

    except:
        st.error("CTR model not found. Please train the model first.")
        return None, None, None, None

    # Load scaler
    try:
        scaler = joblib.load("models/numerical_scaler.pkl")
    except:
        st.error("Scaler not found. Please run preprocessing first.")
        return None, None, None, None

    # Load model input columns
    try:
        model_input_columns = pd.read_csv(os.path.join("data", "model_input_columns.csv")).columns.tolist()
    except:
        st.error("Model input columns file missing.")
        return None, None, None, None

    # Define possible ad variants
    ad_types = ["Native", "Text", "Video", "Banner"]
    ad_topics = ["Finance", "Food", "Health", "Technology", "Travel", "Fashion"]
    ad_placements = ["Social Media", "Website", "Search Engine"]

    ad_variants = list(product(ad_types, ad_topics, ad_placements))
    return model, scaler, model_input_columns, ad_variants


# ------------------- HELPERS -------------------
def build_input(user_features, ad_tuple, scaler, model_input_columns):
    ad_type, ad_topic, ad_place = ad_tuple
    input_data = {
        'Age': user_features['Age'],
        'Income': user_features['Income'],
        'Gender_Female': user_features['Gender'] == 'Female',
        'Gender_Male': user_features['Gender'] == 'Male',
        'Gender_Other': user_features['Gender'] == 'Other',
        'Location_Rural': user_features['Location'] == 'Rural',
        'Location_Suburban': user_features['Location'] == 'Suburban',
        'Location_Urban': user_features['Location'] == 'Urban'
    }

    # One-hot encode ads
    for t in ["Native", "Text", "Video", "Banner"]:
        input_data[f'Ad Type_{t}'] = int(t == ad_type)
    for tp in ["Finance", "Food", "Health", "Technology", "Travel", "Fashion"]:
        input_data[f'Ad Topic_{tp}'] = int(tp == ad_topic)
    for pl in ["Social Media", "Website", "Search Engine"]:
        input_data[f'Ad Placement_{pl}'] = int(pl == ad_place)

    df = pd.DataFrame([input_data])
    df[['Age', 'Income']] = scaler.transform(df[['Age', 'Income']])

    # Add missing columns if needed
    for col in model_input_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[model_input_columns]
    return df.to_numpy(dtype=np.float64).reshape(1, df.shape[1], 1)


def predict_ctr(model, input_array):
    ctr = float(model.predict(input_array, verbose=0)[0][0])
    # Clip CTR to [0, 1] range
    return max(0.0, min(1.0, ctr))


# ------------------- STREAMLIT PAGE -------------------
def main():
    st.title("📊 Customer Ad Prediction")
    
    # Require authentication before accessing the page
    if not require_authentication():
        return

    # Add sidebar info only after authentication
    add_sidebar_info()

    model, scaler, model_input_columns, ad_variants = load_artifacts()
    if model is None:
        return

    # Enhanced input form with better UX
    st.markdown("### 👤 Customer Profile")
    st.markdown("Enter the target customer's demographic information to get personalized ad recommendations.")
    
    with st.form("customer_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.slider(
                "🎂 Age", 
                min_value=10, 
                max_value=100, 
                value=25,
                help="Customer's age in years"
            )
            
            gender = st.selectbox(
                "👤 Gender", 
                ["Female", "Male", "Other"],
                help="Customer's gender identity"
            )
        
        with col2:
            income = st.number_input(
                "💰 Annual Income ($)", 
                min_value=1000.0, 
                max_value=200000.0, 
                value=30000.0,
                step=1000.0,
                help="Customer's annual income in USD"
            )
            
            location = st.selectbox(
                "🏠 Location Type", 
                ["Rural", "Suburban", "Urban"],
                help="Customer's residential area type"
            )
        
        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            submit = st.form_submit_button(
                "🚀 Generate Ad Recommendation",
                use_container_width=True,
                type="primary"
            )
    
    if submit:
        # Add loading animation
        with st.spinner('🤖 Analyzing customer profile and generating recommendations...'):
            time.sleep(1)  # Simulate processing time

    if submit:
        user_features = {"Age": age, "Income": income, "Gender": gender, "Location": location}
        ctrs = []

        # Show progress bar while processing
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, ad in enumerate(ad_variants):
            input_array = build_input(user_features, ad, scaler, model_input_columns)
            ctrs.append(predict_ctr(model, input_array))
            
            # Update progress
            progress = (i + 1) / len(ad_variants)
            progress_bar.progress(progress)
            status_text.text(f'Processing ad variant {i+1}/{len(ad_variants)}: {ad[0]} - {ad[1]} - {ad[2]}')
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()

        ctrs = np.array(ctrs)
        best_index = np.argmax(ctrs)
        best_ad = ad_variants[best_index]
        best_ctr = ctrs[best_index]

        # Compute reward with linear normalization: CTR=0→-100, CTR=0.5→0, CTR=1→+100
        reward = (best_ctr - 0.5) * 200
        
        # Add celebration for good results
        if best_ctr > 0.7:
            st.balloons()
        elif best_ctr > 0.5:
            st.success("Great result! 🎉")

        # Enhanced results display with rich visualizations
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("🎯 Best Ad Recommendation")
            
            # Create recommendation card
            with st.container():
                st.markdown(
                    f"""
                    <div style="
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        padding: 20px;
                        border-radius: 10px;
                        color: white;
                        margin: 10px 0;
                    ">
                        <h3 style="margin: 0; color: white;">📺 {best_ad[0]} Ad</h3>
                        <p style="margin: 5px 0; font-size: 16px;">🏷️ Topic: {best_ad[1]}</p>
                        <p style="margin: 5px 0; font-size: 16px;">📍 Placement: {best_ad[2]}</p>
                        <div style="display: flex; justify-content: space-between; margin-top: 15px;">
                            <div>
                                <p style="margin: 0; font-size: 14px; opacity: 0.8;">Predicted CTR</p>
                                <p style="margin: 0; font-size: 24px; font-weight: bold;">{best_ctr:.4f}</p>
                            </div>
                            <div>
                                <p style="margin: 0; font-size: 14px; opacity: 0.8;">Reward Score</p>
                                <p style="margin: 0; font-size: 24px; font-weight: bold;">{reward:.2f}</p>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        
        with col2:
            # CTR Gauge Chart
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = best_ctr,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "CTR Score"},
                delta = {'reference': 0.5},
                gauge = {
                    'axis': {'range': [None, 1]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 0.3], 'color': "lightgray"},
                        {'range': [0.3, 0.7], 'color': "gray"}],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 0.9}}
            ))
            fig_gauge.update_layout(height=300)
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Top 5 Ad Variants Comparison
        st.subheader("📊 Top 5 Ad Variants Comparison")
        
        # Get top 5 CTRs with their indices
        top_5_indices = np.argsort(ctrs)[-5:][::-1]
        top_5_ctrs = ctrs[top_5_indices]
        top_5_ads = [ad_variants[i] for i in top_5_indices]
        top_5_rewards = [(ctr - 0.5) * 200 for ctr in top_5_ctrs]
        
        # Create comparison chart
        ad_labels = [f"{ad[0]}\n{ad[1]}\n{ad[2]}" for ad in top_5_ads]
        
        fig_comparison = make_subplots(
            rows=1, cols=2,
            subplot_titles=("CTR Comparison", "Reward Comparison"),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # CTR bars
        fig_comparison.add_trace(
            go.Bar(x=ad_labels, y=top_5_ctrs, name="CTR", marker_color="skyblue"),
            row=1, col=1
        )
        
        # Reward bars
        colors = ['green' if r >= 0 else 'red' for r in top_5_rewards]
        fig_comparison.add_trace(
            go.Bar(x=ad_labels, y=top_5_rewards, name="Reward", marker_color=colors),
            row=1, col=2
        )
        
        fig_comparison.update_layout(height=400, showlegend=False)
        fig_comparison.update_xaxes(tickangle=45)
        st.plotly_chart(fig_comparison, use_container_width=True)
        
        # Performance Metrics Dashboard
        st.subheader("📈 Performance Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Best CTR",
                value=f"{best_ctr:.4f}",
                delta=f"{(best_ctr - np.mean(ctrs)):.4f}"
            )
        
        with col2:
            st.metric(
                label="Average CTR",
                value=f"{np.mean(ctrs):.4f}",
                delta=f"{(np.mean(ctrs) - 0.05):.4f}"
            )
        
        with col3:
            st.metric(
                label="CTR Range",
                value=f"{(np.max(ctrs) - np.min(ctrs)):.4f}",
                delta=None
            )
        
        with col4:
            confidence = min(100, max(0, (best_ctr - np.mean(ctrs)) / np.std(ctrs) * 50 + 50))
            st.metric(
                label="Confidence %",
                value=f"{confidence:.1f}%",
                delta=None
            )
        
        # CTR Distribution Chart
        st.subheader("📊 CTR Distribution Analysis")
        
        fig_dist = px.histogram(
            x=ctrs,
            nbins=20,
            title="Distribution of CTR Predictions Across All Ad Variants",
            labels={'x': 'CTR', 'y': 'Frequency'},
            color_discrete_sequence=['lightblue']
        )
        fig_dist.add_vline(x=best_ctr, line_dash="dash", line_color="red", 
                          annotation_text="Best CTR")
        fig_dist.add_vline(x=np.mean(ctrs), line_dash="dash", line_color="green", 
                          annotation_text="Average CTR")
        st.plotly_chart(fig_dist, use_container_width=True)
        
        # Export functionality
        st.subheader("💾 Export Results")
        
        # Prepare export data
        export_data = pd.DataFrame({
            'Ad_Type': [ad[0] for ad in ad_variants],
            'Ad_Topic': [ad[1] for ad in ad_variants],
            'Ad_Placement': [ad[2] for ad in ad_variants],
            'Predicted_CTR': ctrs,
            'Reward': [(ctr - 0.5) * 200 for ctr in ctrs],
            'Rank': range(1, len(ctrs) + 1)
        }).sort_values('Predicted_CTR', ascending=False).reset_index(drop=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            csv = export_data.to_csv(index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name=f"ad_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col2:
            if st.button("🔄 Generate New Prediction", key="generate_new_prediction"):
                st.rerun()
        
        # Success animation
        st.success("🎉 Analysis Complete! Your optimal ad recommendation is ready.")
        
        # Add some spacing
        st.markdown("---")
        
        # Additional insights
        with st.expander("🔍 Detailed Insights"):
            st.write("**Key Findings:**")
            st.write(f"• The recommended {best_ad[0]} ad targeting {best_ad[1]} audience has a {best_ctr:.2%} predicted click-through rate")
            st.write(f"• This CTR is {((best_ctr - np.mean(ctrs)) / np.mean(ctrs) * 100):+.1f}% compared to the average across all variants")
            st.write(f"• Expected reward score: {reward:.2f} (Range: -100 to +100)")
            
            if best_ctr > 0.5:
                st.write("✅ **High Performance**: This ad variant shows excellent potential")
            elif best_ctr > 0.3:
                st.write("⚠️ **Moderate Performance**: Consider A/B testing with other variants")
            else:
                st.write("❌ **Low Performance**: Recommend exploring different targeting options")


# Add sidebar enhancements
def add_sidebar_info():
    with st.sidebar:
        st.markdown("### 🎯 Ad Optimization Info")
        st.info(
            "This tool uses machine learning to predict the best ad variant "
            "for your target customer based on demographics and preferences."
        )
        
        st.markdown("### 📊 How CTR is Calculated")
        st.markdown(
            """
            - **CTR Range**: 0.0 to 1.0 (0% to 100%)
            - **Best Performance**: CTR > 0.5 (Positive reward)
            - **Poor Performance**: CTR < 0.5 (Negative reward)
            """
        )
        
        st.markdown("### 🔧 Model Performance")
        try:
            # Load some basic stats if available
            if os.path.exists("data/Dataset_Ads.csv"):
                data = pd.read_csv("data/Dataset_Ads.csv")
                st.metric("Training Samples", f"{len(data):,}")
                st.metric("Avg CTR in Data", f"{data['CTR'].mean():.4f}")
                st.metric("CTR Std Dev", f"{data['CTR'].std():.4f}")
        except:
            st.write("Model stats unavailable")

if __name__ == "__main__":
    main()
