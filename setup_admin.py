#!/usr/bin/env python3
"""
Admin Setup Script for SmartAd Optimizer
This script helps create the first admin user and initialize the admin module.
"""

import sys
import os
from getpass import getpass
from database import db_auth
from audit_logger import log_admin_action

def create_admin_user():
    """Interactive admin user creation"""
    print("=== SmartAd Optimizer Admin Setup ===")
    print("This script will create the first admin user for the system.\n")
    
    # Get admin user details
    print("Enter details for the admin user:")
    username = input("Username: ").strip()
    if not username:
        print("Error: Username cannot be empty")
        return False
    
    email = input("Email: ").strip()
    if not email:
        print("Error: Email cannot be empty")
        return False
    
    full_name = input("Full Name: ").strip()
    if not full_name:
        print("Error: Full name cannot be empty")
        return False
    
    # Get password securely
    while True:
        password = getpass("Password: ")
        if not password:
            print("Error: Password cannot be empty")
            continue
        
        confirm_password = getpass("Confirm Password: ")
        if password != confirm_password:
            print("Error: Passwords do not match. Please try again.")
            continue
        
        break
    
    # Create admin user
    print("\nCreating admin user...")
    try:
        success, message = db_auth.create_admin_user(username, email, password, full_name)
        
        if success:
            print(f"[SUCCESS] {message}")
            
            # Log the admin creation
            log_admin_action("system", "admin_user_created", username, {
                "email": email,
                "full_name": full_name,
                "created_by": "setup_script"
            })
            
            print(f"\nAdmin user '{username}' has been created successfully!")
            print("You can now access the admin dashboard at: /admin_dashboard")
            return True
        else:
            print(f"[ERROR] {message}")
            return False
            
    except Exception as e:
        print(f"[ERROR] Error creating admin user: {e}")
        return False

def check_system_status():
    """Check if the system is properly configured"""
    print("\n=== System Status Check ===")
    
    # Check database connection
    try:
        stats = db_auth.get_user_statistics()
        print(f"[OK] Database connection: OK")
        print(f"   Total users: {stats.get('total_users', 0)}")
        print(f"   Admin users: {stats.get('admin_users', 0)}")
    except Exception as e:
        print(f"[FAILED] Database connection: FAILED ({e})")
        return False
    
    # Check required directories
    required_dirs = ['data', 'models', 'config', 'logs']
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"[OK] Directory '{dir_name}': OK")
        else:
            print(f"[WARNING] Directory '{dir_name}': Missing (will be created)")
            os.makedirs(dir_name, exist_ok=True)
    
    # Check configuration file
    config_file = 'config/system_config.json'
    if os.path.exists(config_file):
        print(f"[OK] Configuration file: OK")
    else:
        print(f"[WARNING] Configuration file: Missing")
    
    # Check model file
    model_file = 'models/ctr_prediction_model.keras'
    if os.path.exists(model_file):
        print(f"[OK] CTR Model: OK")
    else:
        print(f"[WARNING] CTR Model: Missing (train model first)")
    
    return True

def main():
    """Main setup function"""
    print("SmartAd Optimizer Admin Setup Script")
    print("=" * 40)
    
    # Check system status first
    if not check_system_status():
        print("\n[ERROR] System check failed. Please fix the issues above before proceeding.")
        return
    
    # Check if admin users already exist
    try:
        stats = db_auth.get_user_statistics()
        admin_count = stats.get('admin_users', 0)
        
        if admin_count > 0:
            print(f"\n[WARNING] {admin_count} admin user(s) already exist in the system.")
            response = input("Do you want to create another admin user? (y/N): ").strip().lower()
            if response not in ['y', 'yes']:
                print("Setup cancelled.")
                return
    except Exception as e:
        print(f"Warning: Could not check existing admin users: {e}")
    
    # Create admin user
    if create_admin_user():
        print("\n[SUCCESS] Admin setup completed successfully!")
        print("\nNext steps:")
        print("1. Start the Streamlit application: streamlit run app.py")
        print("2. Navigate to the Admin Dashboard page")
        print("3. Log in with your admin credentials")
        print("4. Configure system settings as needed")
    else:
        print("\n[ERROR] Admin setup failed. Please try again.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSetup cancelled by user.")
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        sys.exit(1)
