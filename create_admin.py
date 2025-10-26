#!/usr/bin/env python3
"""
Direct Admin User Creation Script
Creates an admin user with specified credentials
"""

import sys
import os
from database import db_auth
from audit_logger import log_admin_action

def create_admin_directly(username, password, email, full_name):
    """Create admin user with specified credentials"""
    print(f"Creating admin user: {username}")
    
    try:
        success, message = db_auth.create_admin_user(username, email, password, full_name)
        
        if success:
            print(f"[SUCCESS] {message}")
            
            # Log the admin creation
            log_admin_action("system", "admin_user_created", username, {
                "email": email,
                "full_name": full_name,
                "created_by": "direct_script"
            })
            
            print(f"\nAdmin user '{username}' created successfully!")
            return True
        else:
            print(f"[ERROR] {message}")
            return False
            
    except Exception as e:
        print(f"[ERROR] Failed to create admin user: {e}")
        return False

if __name__ == "__main__":
    # Set your admin credentials here
    ADMIN_USERNAME = "admin"
    ADMIN_PASSWORD = "Admin123@"
    ADMIN_EMAIL = "admin@gmail.com"
    ADMIN_FULL_NAME = "System Administrator"
    
    print("SmartAd Optimizer - Direct Admin Creation")
    print("=" * 45)
    
    # Create the admin user
    if create_admin_directly(ADMIN_USERNAME, ADMIN_PASSWORD, ADMIN_EMAIL, ADMIN_FULL_NAME):
        print("\n[SUCCESS] Admin setup completed!")
        print(f"Username: {ADMIN_USERNAME}")
        print(f"Password: {ADMIN_PASSWORD}")
        print("\nYou can now:")
        print("1. Run: streamlit run app.py")
        print("2. Login with the admin credentials")
        print("3. Access the Admin Dashboard")
    else:
        print("\n[ERROR] Admin creation failed!")
