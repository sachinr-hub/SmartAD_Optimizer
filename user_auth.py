import streamlit as st
try:
    # If used as a package under auth/
    from .database import db_auth
except Exception:
    # Backwards compatibility when at project root
    from database import db_auth

def show_login_page():
    """Display login page"""
    st.title("🔐 Login to Ad Optimization Platform")
    
    with st.form("login_form"):
        st.subheader("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        login_button = st.form_submit_button("Login")
        
        if login_button:
            if username and password:
                success, message = db_auth.authenticate_user(username, password)
                if success:
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.session_state.user_info = db_auth.get_user_info(username)
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)
            else:
                st.error("Please enter both username and password")

    # Removed Forgot Password button per request

def show_registration_page():
    """Display registration page"""
    st.title("📝 Register for Ad Optimization Platform")
    
    with st.form("registration_form"):
        st.subheader("Create New Account")
        
        col1, col2 = st.columns(2)
        with col1:
            username = st.text_input("Username*", help="Choose a unique username")
            email = st.text_input("Email*", help="Enter a valid email address")
            full_name = st.text_input("Full Name*", help="Enter your full name")
        
        with col2:
            password = st.text_input("Password*", type="password", 
                                   help="Min 8 chars, 1 uppercase, 1 lowercase, 1 digit")
            confirm_password = st.text_input("Confirm Password*", type="password")
            company = st.text_input("Company (Optional)", help="Your company or organization")
        
        register_button = st.form_submit_button("Register")
        
        if register_button:
            # Validation
            if not all([username, email, password, confirm_password, full_name]):
                st.error("Please fill in all required fields marked with *")
            elif password != confirm_password:
                st.error("Passwords do not match")
            else:
                success, message = db_auth.register_user(username, email, password, full_name, company)
                if success:
                    st.success(message)
                    st.info("You can now login with your credentials")
                    # Use form submit button inside form context
                    if st.form_submit_button("Go to Login"):
                        st.session_state.show_registration = False
                        st.rerun()
                else:
                    st.error(message)

def show_user_profile():
    """Display user profile management"""
    if 'user_info' not in st.session_state:
        return
    
    st.sidebar.subheader("👤 User Profile")
    user_info = st.session_state.user_info
    
    st.sidebar.write(f"**Welcome, {user_info['full_name']}!**")
    st.sidebar.write(f"Username: {st.session_state.username}")
    st.sidebar.write(f"Email: {user_info['email']}")
    if user_info.get('company'):
        st.sidebar.write(f"Company: {user_info['company']}")
    
    # Profile management
    if st.sidebar.button("Edit Profile"):
        st.session_state.show_profile_edit = True
    
    if st.sidebar.button("Change Password"):
        st.session_state.show_password_change = True
    
    if st.sidebar.button("Logout"):
        for key in ['logged_in', 'username', 'user_info', 'show_profile_edit', 'show_password_change']:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

def show_profile_edit():
    """Display profile edit form"""
    st.subheader("✏️ Edit Profile")
    user_info = st.session_state.user_info
    
    with st.form("profile_edit_form"):
        full_name = st.text_input("Full Name", value=user_info['full_name'])
        email = st.text_input("Email", value=user_info['email'])
        company = st.text_input("Company", value=user_info.get('company', ''))
        
        col1, col2 = st.columns(2)
        with col1:
            save_button = st.form_submit_button("Save Changes")
        with col2:
            cancel_button = st.form_submit_button("Cancel")
        
        if save_button:
            success, message = db_auth.update_user_profile(
                st.session_state.username, full_name, company, email
            )
            if success:
                st.success(message)
                st.session_state.user_info = db_auth.get_user_info(st.session_state.username)
                st.session_state.show_profile_edit = False
                st.rerun()
            else:
                st.error(message)

# Removed Forgot/Reset password flows per request

def show_password_change():
    """Display password change form"""
    st.subheader("🔒 Change Password")
    
    with st.form("password_change_form"):
        current_password = st.text_input("Current Password", type="password")
        new_password = st.text_input("New Password", type="password")
        confirm_password = st.text_input("Confirm New Password", type="password")
        
        col1, col2 = st.columns(2)
        with col1:
            change_button = st.form_submit_button("Change Password")
        with col2:
            cancel_button = st.form_submit_button("Cancel")
        
        if change_button:
            if not all([current_password, new_password, confirm_password]):
                st.error("Please fill in all fields")
            elif new_password != confirm_password:
                st.error("New passwords do not match")
            else:
                success, message = db_auth.change_password(
                    st.session_state.username, current_password, new_password
                )
                if success:
                    st.success(message)
                    st.session_state.show_password_change = False
                    st.rerun()
                else:
                    st.error(message)
        
        if cancel_button:
            st.session_state.show_password_change = False
            st.rerun()

def require_authentication():
    """Check if user is authenticated, show login/register if not"""
    if 'logged_in' not in st.session_state or not st.session_state.logged_in:
        # Show login/register toggle
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Login", use_container_width=True):
                st.session_state.show_registration = False
        with col2:
            if st.button("Register", use_container_width=True):
                st.session_state.show_registration = True
        
        # Show appropriate page (Forgot/Reset password removed)
        if st.session_state.get('show_registration', False):
            show_registration_page()
        else:
            show_login_page()
        
        return False
    
    # User is authenticated, show profile in sidebar
    show_user_profile()
    
    # Handle profile edit/password change
    if st.session_state.get('show_profile_edit', False):
        show_profile_edit()
        return False
    
    if st.session_state.get('show_password_change', False):
        show_password_change()
        return False
    
    return True

def require_authentication_login_only():
    """Authentication guard that only shows Login (no registration UI). For admin dashboard use."""
    if 'logged_in' not in st.session_state or not st.session_state.logged_in:
        # Ensure registration UI is not shown
        if 'show_registration' in st.session_state:
            try:
                del st.session_state['show_registration']
            except Exception:
                pass
        # Show only login page
        show_login_page()
        return False

    # User is authenticated, show profile in sidebar
    show_user_profile()

    # Handle profile edit/password change
    if st.session_state.get('show_profile_edit', False):
        show_profile_edit()
        return False

    if st.session_state.get('show_password_change', False):
        show_password_change()
        return False

    return True
