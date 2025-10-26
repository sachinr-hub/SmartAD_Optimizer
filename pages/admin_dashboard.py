import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import os
import numpy as np
import sys
from pathlib import Path
import io
import runpy
import traceback
import shutil

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from user_auth import require_authentication
from database import db_auth
from audit_logger import audit_logger, log_admin_action, log_config_action

# ------------------- ADMIN AUTHENTICATION -------------------
def require_admin_authentication():
    """Check if user is authenticated and has admin privileges"""
    if not require_authentication():
        return False
    
    # Check if user has admin role
    if 'user_info' in st.session_state:
        user_info = st.session_state.user_info
        if user_info and user_info.get('role') == 'admin':
            return True
    
    # If not admin, show error and return False
    st.error("🚫 Access Denied: Admin privileges required")
    st.info("Contact your system administrator for access.")
    return False

# ------------------- ADMIN FUNCTIONS -------------------
def get_system_stats():
    """Get comprehensive system statistics"""
    stats = {}
    
    # User statistics from enhanced database
    try:
        user_stats = db_auth.get_user_statistics()
        stats.update(user_stats)
        
    except Exception as e:
        stats = {
            'total_users': 0,
            'active_users': 0,
            'admin_users': 0,
            'recent_users': 0
        }
    
    # Model statistics
    try:
        if os.path.exists("models/ctr_prediction_model.keras"):
            model_size = os.path.getsize("models/ctr_prediction_model.keras") / (1024*1024)
            model_time = datetime.fromtimestamp(os.path.getmtime("models/ctr_prediction_model.keras"))
            stats['model_size'] = f"{model_size:.2f} MB"
            stats['model_last_updated'] = model_time.strftime('%Y-%m-%d %H:%M:%S')
        else:
            stats['model_size'] = "N/A"
            stats['model_last_updated'] = "N/A"
    except Exception:
        stats['model_size'] = "Error"
        stats['model_last_updated'] = "Error"
    
    # Data statistics
    try:
        if os.path.exists("data/Dataset_Ads.csv"):
            df = pd.read_csv("data/Dataset_Ads.csv")
            stats['training_samples'] = len(df)
            stats['avg_ctr'] = df['CTR'].mean()
        else:
            stats['training_samples'] = 0
            stats['avg_ctr'] = 0
            
        if os.path.exists("data/online_events.csv"):
            df_online = pd.read_csv("data/online_events.csv")
            stats['online_events'] = len(df_online)
        else:
            stats['online_events'] = 0
    except Exception:
        stats['training_samples'] = 0
        stats['avg_ctr'] = 0
        stats['online_events'] = 0
    
    return stats

def get_user_list():
    """Get list of all users with their information"""
    try:
        users = db_auth.get_all_users()
        if users:
            df = pd.DataFrame(users)
            # Ensure consistent column names
            if 'username' not in df.columns and '_id' in df.columns:
                df['username'] = df['_id']
            # Ensure required admin columns exist to prevent KeyErrors downstream
            for col, default in [
                ('email', ''),
                ('full_name', ''),
                ('company', ''),
                ('role', 'user'),
                ('is_active', True)
            ]:
                if col not in df.columns:
                    df[col] = default
            # Normalize is_active to boolean
            try:
                df['is_active'] = df['is_active'].astype(bool)
            except Exception:
                df['is_active'] = True
            return df
        else:
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error fetching users: {e}")
        return pd.DataFrame()

def update_user_role(username, new_role):
    """Update user role"""
    try:
        success, message = db_auth.update_user_role(username, new_role)
        if not success:
            st.error(f"Error updating user role: {message}")
        return success
    except Exception as e:
        st.error(f"Error updating user role: {e}")
        return False

def toggle_user_status(username, is_active):
    """Toggle user active status"""
    try:
        success, message = db_auth.toggle_user_status(username, is_active)
        if not success:
            st.error(f"Error updating user status: {message}")
        return success
    except Exception as e:
        st.error(f"Error updating user status: {e}")
        return False

def get_model_performance_data():
    """Get model performance metrics over time"""
    try:
        if os.path.exists("data/online_events.csv"):
            # Robust CSV read: skip malformed lines to avoid tokenizing errors
            df = pd.read_csv(
                "data/online_events.csv",
                engine="python",
                on_bad_lines="skip"
            )
            # Ensure required columns exist; gracefully degrade if missing
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df['date'] = df['timestamp'].dt.date
                # Build aggregation dict based on available columns
                agg_dict = {}
                if 'CTR' in df.columns:
                    agg_dict['CTR'] = ['mean', 'count']
                if 'predicted_ctr' in df.columns:
                    agg_dict['predicted_ctr'] = 'mean'

                if agg_dict:
                    daily_metrics = df.groupby('date').agg(agg_dict).round(4)
                    daily_metrics = daily_metrics.reset_index()
                    # Flatten any MultiIndex columns to plain strings
                    flat_cols = []
                    for col in daily_metrics.columns:
                        if isinstance(col, tuple):
                            # Join non-empty parts
                            flat = '_'.join([str(c) for c in col if str(c) != ''])
                            flat_cols.append(flat)
                        else:
                            flat_cols.append(col)
                    daily_metrics.columns = flat_cols
                    # Prepare standardized columns
                    daily_metrics['actual_ctr'] = np.nan
                    daily_metrics['interactions'] = 0
                    daily_metrics['predicted_ctr'] = np.nan
                    if 'CTR_mean' in daily_metrics.columns:
                        daily_metrics['actual_ctr'] = daily_metrics['CTR_mean']
                    if 'CTR_count' in daily_metrics.columns:
                        daily_metrics['interactions'] = daily_metrics['CTR_count']
                    if 'predicted_ctr_mean' in daily_metrics.columns:
                        daily_metrics['predicted_ctr'] = daily_metrics['predicted_ctr_mean']
                    # Keep only standardized columns for plotting
                    daily_metrics = daily_metrics[['date', 'actual_ctr', 'interactions', 'predicted_ctr']]
                    return daily_metrics
            return pd.DataFrame()
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading performance data: {e}")
        return pd.DataFrame()

def get_ad_performance_analytics():
    """Get ad variant performance analytics"""
    try:
        if os.path.exists("data/online_events.csv"):
            df = pd.read_csv(
                "data/online_events.csv",
                engine="python",
                on_bad_lines="skip"
            )
            
            # Check if CTR column exists and is numeric
            if 'CTR' not in df.columns:
                return pd.DataFrame()
            
            # Convert CTR to numeric, handling any string values
            df['CTR'] = pd.to_numeric(df['CTR'], errors='coerce')
            df = df.dropna(subset=['CTR'])
            
            # Extract ad features from one-hot encoded columns
            ad_type_cols = [col for col in df.columns if col.startswith('Ad Type_')]
            ad_topic_cols = [col for col in df.columns if col.startswith('Ad Topic_')]
            ad_placement_cols = [col for col in df.columns if col.startswith('Ad Placement_')]
            
            if ad_type_cols and ad_topic_cols and ad_placement_cols:
                # Decode one-hot encoding safely
                df['ad_type'] = df[ad_type_cols].idxmax(axis=1).str.replace('Ad Type_', '')
                df['ad_topic'] = df[ad_topic_cols].idxmax(axis=1).str.replace('Ad Topic_', '')
                df['ad_placement'] = df[ad_placement_cols].idxmax(axis=1).str.replace('Ad Placement_', '')
                
                # Performance by ad variant
                performance = df.groupby(['ad_type', 'ad_topic', 'ad_placement']).agg({
                    'CTR': ['mean', 'count', 'sum']
                }).round(4)
                
                performance.columns = ['avg_ctr', 'impressions', 'clicks']
                performance = performance.reset_index()
                
                # Ensure all numeric columns are properly typed
                performance['avg_ctr'] = pd.to_numeric(performance['avg_ctr'], errors='coerce')
                performance['impressions'] = pd.to_numeric(performance['impressions'], errors='coerce')
                performance['clicks'] = pd.to_numeric(performance['clicks'], errors='coerce')
                
                return performance
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading ad analytics: {e}")
        return pd.DataFrame()

# ------------------- UTILITIES -------------------
def _safe_write_file(target_path: str, data: bytes) -> tuple[bool, str]:
    try:
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        with open(target_path, 'wb') as f:
            f.write(data)
        return True, f"Saved to {target_path}"
    except Exception as e:
        return False, f"Failed to save file: {e}"

def _read_json_file(path: str):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None

def _run_script_capture(script_path: str) -> tuple[bool, str]:
    """Run a Python script located in the project and capture stdout/stderr safely."""
    if not os.path.exists(script_path):
        return False, f"Script not found: {script_path}"
    buf = io.StringIO()
    try:
        # Capture prints
        import contextlib, sys as _sys
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            # Ensure working directory is project root
            runpy.run_path(script_path, run_name='__main__')
        output = buf.getvalue()
        return True, output
    except Exception:
        output = buf.getvalue() + "\n" + traceback.format_exc()
        return False, output
    finally:
        buf.close()

# ------------------- NEW ADMIN PAGES -------------------
def show_data_management():
    st.markdown("## 🗂 Data Management")

    data_dir = "data"
    dataset_path = os.path.join(data_dir, "Dataset_Ads.csv")

    # Current dataset info
    st.markdown("### Current Dataset")
    if os.path.exists(dataset_path):
        try:
            df_head = pd.read_csv(dataset_path, nrows=200)
            st.write(f"Rows (first 200 loaded): {len(df_head)}")
            st.dataframe(df_head.head(20), use_container_width=True)
        except Exception as e:
            st.warning(f"Could not preview dataset: {e}")
        stats_col1, stats_col2 = st.columns(2)
        with stats_col1:
            st.info(f"Last Modified: {datetime.fromtimestamp(os.path.getmtime(dataset_path)).strftime('%Y-%m-%d %H:%M:%S')}")
        with stats_col2:
            st.info(f"File Size: {os.path.getsize(dataset_path)/(1024*1024):.2f} MB")
    else:
        st.warning("Dataset_Ads.csv not found in data/")

    st.markdown("### Upload/Replace Dataset")
    uploaded = st.file_uploader("Upload CSV to replace Dataset_Ads.csv", type=["csv"], key="upload_ds")
    if uploaded is not None:
        try:
            test_df = pd.read_csv(uploaded)
            required_cols = ["Age", "Income", "CTR", "Gender", "Location", "Ad Type", "Ad Topic", "Ad Placement"]
            missing = [c for c in required_cols if c not in test_df.columns]
            if missing:
                st.error(f"Uploaded CSV missing required columns: {missing}")
            else:
                # Save
                ok, msg = _safe_write_file(dataset_path, uploaded.getvalue())
                if ok:
                    st.success("Dataset replaced successfully.")
                    st.info("You may want to run Preprocessing next in Model Management.")
                    st.rerun()
                else:
                    st.error(msg)
        except Exception as e:
            st.error(f"Failed to read uploaded CSV: {e}")

def show_model_management():
    st.markdown("## 🤖 Model Management")

    models_dir = "models"
    model_file = os.path.join(models_dir, "ctr_prediction_model.keras")

    # Model info
    st.markdown("### Model Info")
    if os.path.exists(model_file):
        st.metric("Model Size", f"{os.path.getsize(model_file)/(1024*1024):.2f} MB")
        st.info(f"Last Updated: {datetime.fromtimestamp(os.path.getmtime(model_file)).strftime('%Y-%m-%d %H:%M:%S')}")
        with open(model_file, 'rb') as f:
            st.download_button("📥 Download Model", f.read(), file_name="ctr_prediction_model.keras")
    else:
        st.warning("Model file not found. Train a model to create it.")

    st.markdown("---")
    st.markdown("### Actions")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        if st.button("Run Preprocessing", key="btn_preprocess"):
            with st.spinner("Running preprocess_data.py ..."):
                ok, out = _run_script_capture("preprocess_data.py")
            st.code(out or "", language="bash")
            st.success("Preprocessing completed." if ok else "Preprocessing finished with errors.")
    with col_b:
        if st.button("Train Model", key="btn_train"):
            with st.spinner("Running train_ctr_model.py ..."):
                ok, out = _run_script_capture("train_ctr_model.py")
            st.code(out or "", language="bash")
            st.success("Training completed." if ok else "Training finished with errors.")
    with col_c:
        uploaded_model = st.file_uploader("Upload Model (.keras)", type=["keras"], key="upload_model")
        if uploaded_model is not None:
            ok, msg = _safe_write_file(model_file, uploaded_model.getvalue())
            if ok:
                st.success("Model uploaded successfully.")
                st.rerun()
            else:
                st.error(msg)

    st.markdown("---")
    st.markdown("### Logs & Reports")
    # Show training/report images if present
    results_img = os.path.join("results", "training_history.png")
    if os.path.exists(results_img):
        st.image(results_img, caption="Training History", use_container_width=True)
    else:
        st.info("No training history image found yet.")

def show_audit_logs():
    st.markdown("## 📜 Audit Logs")
    log_path = os.path.join("data", "audit_log.json")
    logs = _read_json_file(log_path) or []

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        user_filter = st.text_input("Filter by Username")
    with col2:
        action_filter = st.text_input("Filter by Action")
    with col3:
        limit = st.number_input("Limit", min_value=10, max_value=1000, value=200, step=10)

    if logs:
        # Apply filters
        df = pd.DataFrame(logs)
        if user_filter:
            df = df[df['username'].astype(str).str.contains(user_filter, case=False, na=False)]
        if action_filter:
            df = df[df['action'].astype(str).str.contains(action_filter, case=False, na=False)]
        df = df.sort_values(by='timestamp', ascending=False).head(int(limit))
        st.dataframe(df, use_container_width=True)
        st.download_button(
            label="📥 Download Filtered Logs (JSON)",
            data=df.to_json(orient='records', indent=2),
            file_name=f"audit_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    else:
        st.info("No audit logs found yet.")

# ------------------- ADMIN DASHBOARD PAGES -------------------
def show_dashboard_overview():
    """Main dashboard overview"""
    st.markdown("## 📊 System Overview")
    
    # Get system statistics
    stats = get_system_stats()
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Users",
            stats.get('total_users', 0),
            delta=f"+{stats.get('recent_users', 0)} recent"
        )
    
    with col2:
        st.metric(
            "Active Users",
            stats.get('active_users', 0),
            delta=None
        )
    
    with col3:
        st.metric(
            "Admin Users",
            stats.get('admin_users', 0),
            delta=None
        )
    
    with col4:
        st.metric(
            "Model Status",
            stats.get('model_size', 'N/A'),
            delta=None
        )
    
    # Model information
    st.markdown("### 🤖 Model Status")
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"**Model Size:** {stats.get('model_size', 'N/A')}")
        st.info(f"**Last Updated:** {stats.get('model_last_updated', 'N/A')}")
    
    with col2:
        st.info("Model retraining is handled automatically based on performance and schedule.")
    
    # Performance charts
    st.markdown("### 📈 Performance Trends")
    
    performance_data = get_model_performance_data()
    if not performance_data.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            fig_ctr = px.line(
                performance_data, 
                x='date', 
                y=['actual_ctr', 'predicted_ctr'],
                title="CTR Trends Over Time",
                labels={'value': 'CTR', 'date': 'Date'}
            )
            st.plotly_chart(fig_ctr, use_container_width=True)
        
        with col2:
            fig_interactions = px.bar(
                performance_data,
                x='date',
                y='interactions',
                title="Daily Interactions",
                labels={'interactions': 'Number of Interactions', 'date': 'Date'}
            )
            st.plotly_chart(fig_interactions, use_container_width=True)
    else:
        st.info("No performance data available yet. Run some simulations to see trends.")

def show_user_management():
    """User management interface"""
    st.markdown("## 👥 User Management")
    
    # Add new user section
    with st.expander("➕ Add New User", expanded=False):
        with st.form("add_user_form"):
            st.markdown("### Create New User")
            col1, col2 = st.columns(2)
            
            with col1:
                new_username = st.text_input("Username", key="new_username")
                new_email = st.text_input("Email", key="new_email")
                new_role = st.selectbox("Role", ["user", "admin"], key="new_role")
            
            with col2:
                new_full_name = st.text_input("Full Name", key="new_full_name")
                new_company = st.text_input("Company (Optional)", key="new_company")
                new_password = st.text_input("Password", type="password", key="new_password")
            
            if st.form_submit_button("Create User", type="primary"):
                if new_username and new_email and new_full_name and new_password:
                    try:
                        if new_role == "admin":
                            success, message = db_auth.create_admin_user(new_username, new_email, new_password, new_full_name)
                        else:
                            success, message = db_auth.register_user(new_username, new_email, new_password, new_full_name, new_company)
                        
                        if success:
                            st.success(f"✅ {message}")
                            log_admin_action(st.session_state.get('username', 'admin'), "user_created", new_username, {
                                "email": new_email,
                                "role": new_role,
                                "full_name": new_full_name
                            })
                            st.rerun()
                        else:
                            st.error(f"❌ {message}")
                    except Exception as e:
                        st.error(f"❌ Error creating user: {e}")
                else:
                    st.error("❌ Please fill in all required fields")
    
    # Get user list
    users_df = get_user_list()
    
    if users_df.empty:
        st.warning("No users found in the database.")
        return
    
    # User statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_users = len(users_df)
        st.metric("Total Users", total_users)
    
    with col2:
        # Count booleans robustly regardless of type
        active_users = int((users_df['is_active'] == True).sum())
        st.metric("Active Users", active_users)
    
    with col3:
        admin_users = len(users_df[users_df['role'] == 'admin'])
        st.metric("Admin Users", admin_users)
    
    # User table with management options
    st.markdown("### 📋 User List")
    
    # Add filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        role_filter = st.selectbox("Filter by Role", ["All", "admin", "user"])
    
    with col2:
        status_filter = st.selectbox("Filter by Status", ["All", "Active", "Inactive"])
    
    with col3:
        search_term = st.text_input("Search Users", placeholder="Username or email")
    
    # Apply filters
    filtered_df = users_df.copy()
    
    if role_filter != "All":
        filtered_df = filtered_df[filtered_df['role'] == role_filter]
    
    if status_filter == "Active":
        filtered_df = filtered_df[filtered_df['is_active'] == True]
    elif status_filter == "Inactive":
        filtered_df = filtered_df[filtered_df['is_active'] == False]
    
    if search_term:
        mask = (
            filtered_df['username'].str.contains(search_term, case=False, na=False) |
            filtered_df['email'].str.contains(search_term, case=False, na=False)
        )
        filtered_df = filtered_df[mask]
    
    # Display user table with actions
    st.markdown(f"**Showing {len(filtered_df)} users**")
    
    for idx, user in filtered_df.iterrows():
        with st.container():
            col1, col2, col3, col4, col5 = st.columns([2, 2, 1, 1, 1])
            
            with col1:
                st.write(f"**{user['username']}**")
                st.write(f"📧 {user['email']}")
            
            with col2:
                st.write(f"👤 {user.get('full_name', 'N/A')}")
                st.write(f"🏢 {user.get('company', 'N/A')}")
            
            with col3:
                role_color = "🔴" if user['role'] == 'admin' else "🔵"
                st.write(f"{role_color} {user['role']}")
                status_color = "🟢" if user['is_active'] else "🔴"
                st.write(f"{status_color} {'Active' if user['is_active'] else 'Inactive'}")
            
            with col4:
                # Role change
                new_role = st.selectbox(
                    "Role", 
                    ["user", "admin"], 
                    index=0 if user['role'] == 'user' else 1,
                    key=f"role_{user['username']}"
                )
                if st.button("Update Role", key=f"update_role_btn_{user['username']}"):
                    if update_user_role(user['username'], new_role):
                        st.success("Role updated!")
                        log_admin_action(st.session_state.get('username', 'admin'), "role_updated", user['username'], {
                            "old_role": user['role'],
                            "new_role": new_role
                        })
                        st.rerun()
            
            with col5:
                # Status toggle
                if st.button(
                    "Deactivate" if user['is_active'] else "Activate",
                    key=f"toggle_{user['username']}",
                    type="secondary"
                ):
                    new_status = not user['is_active']
                    if toggle_user_status(user['username'], new_status):
                        action = "activated" if new_status else "deactivated"
                        st.success(f"User {action}!")
                        log_admin_action(st.session_state.get('username', 'admin'), "status_changed", user['username'], {
                            "new_status": new_status
                        })
                        st.rerun()
                
                # Delete user
                if st.button(
                    "🗑️ Delete",
                    key=f"delete_{user['username']}",
                    type="secondary",
                    help="Permanently delete this user"
                ):
                    # Confirmation dialog
                    if st.session_state.get(f"confirm_delete_{user['username']}", False):
                        try:
                            success, message = db_auth.delete_user(user['username'])
                            if success:
                                st.success(f"✅ {message}")
                                log_admin_action(st.session_state.get('username', 'admin'), "user_deleted", user['username'], {
                                    "email": user['email'],
                                    "role": user['role']
                                })
                                st.rerun()
                            else:
                                st.error(f"❌ {message}")
                        except Exception as e:
                            st.error(f"❌ Error deleting user: {e}")
                        finally:
                            st.session_state[f"confirm_delete_{user['username']}"] = False
                    else:
                        st.session_state[f"confirm_delete_{user['username']}"] = True
                        st.warning(f"⚠️ Click Delete again to confirm deletion of {user['username']}")
            
            st.divider()
    
    # User management actions
    st.markdown("### ⚙️ User Actions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Change User Role**")
        selected_user = st.selectbox("Select User", users_df['username'].tolist())
        new_role = st.selectbox("New Role", ["user", "admin"])
        
        if st.button("Update Role", key="bulk_update_role_btn"):
            if update_user_role(selected_user, new_role):
                st.success(f"✅ Updated {selected_user} role to {new_role}")
                st.rerun()
    
    with col2:
        st.markdown("**Toggle User Status**")
        selected_user_status = st.selectbox("Select User ", users_df['username'].tolist(), key="status_user")
        current_status = users_df[users_df['username'] == selected_user_status]['is_active'].iloc[0]
        new_status = not bool(current_status)
        
        if st.button(f"{'Activate' if new_status else 'Deactivate'} User", key="toggle_status_btn"):
            if toggle_user_status(selected_user_status, new_status):
                status_text = "activated" if new_status else "deactivated"
                st.success(f"✅ User {selected_user_status} {status_text}")
                st.rerun()


# ------------------- MAIN ADMIN INTERFACE -------------------
def main():
    st.set_page_config(
        page_title="Admin Dashboard",
        page_icon="🔧",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom header
    st.markdown(
        """
        <div style="
            background: linear-gradient(90deg, #FF6B6B 0%, #4ECDC4 100%);
            padding: 2rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            text-align: center;
        ">
            <h1 style="color: white; margin: 0; font-size: 3rem;">🔧 ADMIN DASHBOARD</h1>
            <p style="color: white; margin: 0.5rem 0 0 0; font-size: 1.2rem; opacity: 0.9;">
                System Administration & Management Portal
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Check admin authentication
    if not require_admin_authentication():
        return
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown("### 🧭 Navigation")
        
        page = st.radio(
            "Select Page",
            [
                "📊 Dashboard Overview",
                "👥 User Management",
                "🗂 Data Management",
                "🤖 Model Management",
                "📜 Audit Logs"
            ]
        )
        
        st.markdown("---")
        
        # Quick stats
        stats = get_system_stats()
        st.markdown("### 📋 Quick Stats")
        st.metric("Total Users", stats.get('total_users', 0))
        st.metric("Active Users", stats.get('active_users', 0))
        st.metric("Model Size", stats.get('model_size', 'N/A'))
        
        st.markdown("---")
        
        # Quick actions
        st.markdown("### ⚡ Quick Actions")
        
        if st.button("🔄 Refresh Data", key="refresh_data_btn"):
            st.rerun()
        
        if st.button("📊 Export System Report", key="export_report_btn"):
            # Generate system report
            report_data = {
                'timestamp': datetime.now().isoformat(),
                'system_stats': stats,
                'users': get_user_list().to_dict('records') if not get_user_list().empty else []
            }
            
            report_json = json.dumps(report_data, indent=2, default=str)
            st.download_button(
                label="📥 Download Report",
                data=report_json,
                file_name=f"system_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

        # Removed Email Settings and password reset test functionality per request
    
    # Main content based on selected page
    if page == "📊 Dashboard Overview":
        show_dashboard_overview()
    elif page == "👥 User Management":
        show_user_management()
    elif page == "🗂 Data Management":
        show_data_management()
    elif page == "🤖 Model Management":
        show_model_management()
    elif page == "📜 Audit Logs":
        show_audit_logs()

if __name__ == "__main__":
    main()
