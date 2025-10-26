import json
import os
import hashlib
import re
from datetime import datetime, timedelta

# Fallback JSON-based authentication when MongoDB is not available
USER_DATA_FILE = "data/users.json"

class FallbackAuth:
    def __init__(self):
        self.ensure_user_file_exists()
    
    def ensure_user_file_exists(self):
        """Ensure the user data file exists"""
        os.makedirs("data", exist_ok=True)
        if not os.path.exists(USER_DATA_FILE):
            with open(USER_DATA_FILE, 'w') as f:
                json.dump({}, f)
    
    def hash_password(self, password):
        """Hash password using SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def verify_password(self, password, hashed):
        """Verify password against hash"""
        return self.hash_password(password) == hashed
    
    def validate_email(self, email):
        """Validate email format"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    def validate_password(self, password):
        """Validate password strength"""
        if len(password) < 8:
            return False, "Password must be at least 8 characters long"
        if not re.search(r'[A-Z]', password):
            return False, "Password must contain at least one uppercase letter"
        if not re.search(r'[a-z]', password):
            return False, "Password must contain at least one lowercase letter"
        if not re.search(r'\d', password):
            return False, "Password must contain at least one digit"
        return True, "Password is valid"
    
    def load_users(self):
        """Load users from JSON file"""
        try:
            with open(USER_DATA_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    
    def save_users(self, users):
        """Save users to JSON file"""
        with open(USER_DATA_FILE, 'w') as f:
            json.dump(users, f, indent=2, default=str)
    
    def register_user(self, username, email, password, full_name, company=""):
        """Register a new user"""
        users = self.load_users()
        
        # Check if username or email already exists
        if username in users:
            return False, "Username already exists"
        
        for user_data in users.values():
            if user_data.get('email') == email:
                return False, "Email already registered"
        
        # Validate email
        if not self.validate_email(email):
            return False, "Invalid email format"
        
        # Validate password
        is_valid, message = self.validate_password(password)
        if not is_valid:
            return False, message
        
        # Create new user
        users[username] = {
            'email': email,
            'password': self.hash_password(password),
            'full_name': full_name,
            'company': company,
            'registration_date': datetime.now().isoformat(),
            'last_login': None,
            'is_active': True,
            'role': 'user',
            'created_at': datetime.now().isoformat(),
            'permissions': ['read', 'write']
        }
        
        self.save_users(users)
        return True, "User registered successfully"
    
    def authenticate_user(self, username, password):
        """Authenticate user login"""
        users = self.load_users()
        
        if username not in users:
            return False, "Username not found"
        
        user_data = users[username]
        if not user_data.get('is_active', True):
            return False, "Account is deactivated"
        
        if self.verify_password(password, user_data['password']):
            # Update last login
            users[username]['last_login'] = datetime.now().isoformat()
            self.save_users(users)
            return True, "Login successful"
        else:
            return False, "Invalid password"
    
    def get_user_info(self, username):
        """Get user information"""
        users = self.load_users()
        user_data = users.get(username, None)
        if user_data:
            # Remove password from returned data
            user_info = user_data.copy()
            user_info.pop('password', None)
            return user_info
        return None
    
    def update_user_profile(self, username, full_name=None, company=None, email=None):
        """Update user profile"""
        users = self.load_users()
        
        if username not in users:
            return False, "User not found"
        
        if email and email != users[username]['email']:
            # Check if new email is already taken
            for user_data in users.values():
                if user_data.get('email') == email:
                    return False, "Email already in use"
            
            if not self.validate_email(email):
                return False, "Invalid email format"
            
            users[username]['email'] = email
        
        if full_name:
            users[username]['full_name'] = full_name
        
        if company is not None:  # Allow empty string
            users[username]['company'] = company
        
        self.save_users(users)
        return True, "Profile updated successfully"
    
    def change_password(self, username, old_password, new_password):
        """Change user password"""
        users = self.load_users()
        
        if username not in users:
            return False, "User not found"
        
        # Verify old password
        if not self.verify_password(old_password, users[username]['password']):
            return False, "Current password is incorrect"
        
        # Validate new password
        is_valid, message = self.validate_password(new_password)
        if not is_valid:
            return False, message
        
        # Update password
        users[username]['password'] = self.hash_password(new_password)
        self.save_users(users)
        return True, "Password changed successfully"
    
    def save_campaign_metrics(self, username, campaign_data):
        """Save campaign performance metrics (fallback implementation)"""
        try:
            metrics_file = "data/metrics.json"
            
            # Load existing metrics
            if os.path.exists(metrics_file):
                with open(metrics_file, 'r') as f:
                    all_metrics = json.load(f)
            else:
                all_metrics = []
            
            # Add new metric
            metric_data = {
                'username': username,
                'timestamp': datetime.now().isoformat(),
                'campaign_data': campaign_data
            }
            
            all_metrics.append(metric_data)
            
            # Save back to file
            with open(metrics_file, 'w') as f:
                json.dump(all_metrics, f, indent=2, default=str)
            
            return True
            
        except Exception as e:
            print(f"Error saving metrics: {e}")
            return False
    
    def get_user_metrics(self, username, limit=100):
        """Get user's campaign metrics (fallback implementation)"""
        try:
            metrics_file = "data/metrics.json"
            
            if not os.path.exists(metrics_file):
                return []
            
            with open(metrics_file, 'r') as f:
                all_metrics = json.load(f)
            
            # Filter by username and limit results
            user_metrics = [m for m in all_metrics if m.get('username') == username]
            user_metrics.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
            
            return user_metrics[:limit]
            
        except Exception as e:
            print(f"Error getting metrics: {e}")
            return []
    
    def update_user_role(self, username, role):
        """Update user role (admin function)"""
        users = self.load_users()
        
        if username not in users:
            return False, "User not found"
        
        users[username]['role'] = role
        self.save_users(users)
        return True, f"User role updated to {role}"
    
    def toggle_user_status(self, username, is_active):
        """Toggle user active status (admin function)"""
        users = self.load_users()
        
        if username not in users:
            return False, "User not found"
        
        users[username]['is_active'] = is_active
        self.save_users(users)
        status = "activated" if is_active else "deactivated"
        return True, f"User {status} successfully"
    
    def get_all_users(self):
        """Get all users (admin function)"""
        users = self.load_users()
        
        # Remove passwords from all user data
        safe_users = []
        for username, user_data in users.items():
            safe_user = user_data.copy()
            safe_user.pop('password', None)
            safe_user['username'] = username
            safe_users.append(safe_user)
        
        # Sort by registration date (newest first)
        safe_users.sort(key=lambda x: x.get('registration_date', ''), reverse=True)
        return safe_users
    
    def get_user_statistics(self):
        """Get user statistics (admin function)"""
        users = self.load_users()
        
        total_users = len(users)
        active_users = sum(1 for user in users.values() if user.get('is_active', True))
        admin_users = sum(1 for user in users.values() if user.get('role') == 'admin')
        
        # Users registered in last 30 days
        thirty_days_ago = datetime.now() - timedelta(days=30)
        recent_users = 0
        for user in users.values():
            reg_date_str = user.get('registration_date')
            if reg_date_str:
                try:
                    reg_date = datetime.fromisoformat(reg_date_str.replace('Z', '+00:00'))
                    if reg_date >= thirty_days_ago:
                        recent_users += 1
                except:
                    pass
        
        return {
            "total_users": total_users,
            "active_users": active_users,
            "admin_users": admin_users,
            "recent_users": recent_users
        }
    
    def create_admin_user(self, username, email, password, full_name):
        """Create an admin user"""
        users = self.load_users()
        
        # Check if username or email already exists
        if username in users:
            return False, "Username already exists"
        
        for user_data in users.values():
            if user_data.get('email') == email:
                return False, "Email already registered"
        
        # Validate email
        if not self.validate_email(email):
            return False, "Invalid email format"
        
        # Validate password
        is_valid, message = self.validate_password(password)
        if not is_valid:
            return False, message
        
        # Create admin user
        users[username] = {
            'email': email,
            'password': self.hash_password(password),
            'full_name': full_name,
            'company': 'System Administrator',
            'registration_date': datetime.now().isoformat(),
            'last_login': None,
            'is_active': True,
            'role': 'admin',
            'created_at': datetime.now().isoformat(),
            'permissions': ['read', 'write', 'admin']
        }
        
        self.save_users(users)
        return True, "Admin user created successfully"
    
    def delete_user(self, username):
        """Delete a user (admin function)"""
        users = self.load_users()
        
        if username not in users:
            return False, "User not found"
        
        del users[username]
        self.save_users(users)
        return True, f"User '{username}' deleted successfully"
