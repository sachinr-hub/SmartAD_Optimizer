import os
import pymongo
from pymongo import MongoClient
import streamlit as st
from datetime import datetime, timedelta
import bcrypt
import re
import secrets
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class DatabaseManager:
    def __init__(self):
        self.client = None
        self.db = None
        self.connect()
    
    def connect(self):
        """Connect to MongoDB database"""
        try:
            # Try to get MongoDB URI from environment variable first
            mongo_uri = os.getenv('MONGODB_URI')
            
            if not mongo_uri:
                # Fallback to local MongoDB
                mongo_uri = "mongodb://localhost:27017/"
            
            self.client = MongoClient(mongo_uri)
            self.db = self.client['smartad_optimizer']
            
            # Test the connection
            self.client.admin.command('ping')
            print("Successfully connected to MongoDB!")
            
        except Exception as e:
            # Fail fast: MongoDB is mandatory
            raise RuntimeError(f"Failed to connect to MongoDB: {e}")
    
    def get_collection(self, collection_name):
        """Get a collection from the database"""
        if self.db is not None:
            return self.db[collection_name]
        return None
    
    def close_connection(self):
        """Close database connection"""
        if self.client:
            self.client.close()

class UserAuthDB:
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.users_collection = self.db_manager.get_collection('users')
        self.campaigns_collection = self.db_manager.get_collection('campaigns')
        self.metrics_collection = self.db_manager.get_collection('metrics')
        
        # Create indexes for better performance
        if self.users_collection is not None:
            try:
                self.users_collection.create_index("username", unique=True)
                self.users_collection.create_index("email", unique=True)
            except Exception as e:
                print(f"Index creation warning: {e}")
    
    def hash_password(self, password):
        """Hash password using bcrypt"""
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    
    def verify_password(self, password, hashed):
        """Verify password against hash"""
        return bcrypt.checkpw(password.encode('utf-8'), hashed)
    
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
    
    def register_user(self, username, email, password, full_name, company=""):
        """Register a new user"""
        if self.users_collection is None:
            return False, "Database connection failed"
        
        try:
            # Check if username or email already exists
            if self.users_collection.find_one({"username": username}):
                return False, "Username already exists"
            
            if self.users_collection.find_one({"email": email}):
                return False, "Email already registered"
            
            # Validate email
            if not self.validate_email(email):
                return False, "Invalid email format"
            
            # Validate password
            is_valid, message = self.validate_password(password)
            if not is_valid:
                return False, message
            
            # Create new user
            user_data = {
                'username': username,
                'email': email,
                'password': self.hash_password(password),
                'full_name': full_name,
                'company': company,
                'registration_date': datetime.now(),
                'last_login': None,
                'is_active': True,
                'role': 'user',
                'created_at': datetime.now().isoformat(),
                'permissions': ['read', 'write']
            }
            
            result = self.users_collection.insert_one(user_data)
            if result.inserted_id:
                return True, "User registered successfully"
            else:
                return False, "Failed to register user"
                
        except pymongo.errors.DuplicateKeyError:
            return False, "Username or email already exists"
        except Exception as e:
            return False, f"Registration failed: {str(e)}"
    
    def authenticate_user(self, username, password):
        """Authenticate user login"""
        if self.users_collection is None:
            return False, "Database connection failed"
        
        try:
            user = self.users_collection.find_one({"username": username})
            
            if not user:
                return False, "Username not found"
            
            if not user.get('is_active', True):
                return False, "Account is deactivated"
            
            if self.verify_password(password, user['password']):
                # Update last login
                self.users_collection.update_one(
                    {"username": username},
                    {"$set": {"last_login": datetime.now()}}
                )
                return True, "Login successful"
            else:
                return False, "Invalid password"
                
        except Exception as e:
            return False, f"Authentication failed: {str(e)}"
    
    def get_user_info(self, username):
        """Get user information"""
        if self.users_collection is None:
            return None
        
        try:
            user = self.users_collection.find_one(
                {"username": username},
                {"password": 0}  # Exclude password from result
            )
            return user
        except Exception as e:
            print(f"Error getting user info: {e}")
            return None
    
    def update_user_role(self, username, role):
        """Update user role (admin function)"""
        if self.users_collection is None:
            return False, "Database connection failed"
        
        try:
            result = self.users_collection.update_one(
                {"username": username},
                {"$set": {"role": role}}
            )
            
            if result.modified_count > 0:
                return True, f"User role updated to {role}"
            else:
                return False, "User not found or no changes made"
                
        except Exception as e:
            return False, f"Role update failed: {str(e)}"
    
    def toggle_user_status(self, username, is_active):
        """Toggle user active status (admin function)"""
        if self.users_collection is None:
            return False, "Database connection failed"
        
        try:
            result = self.users_collection.update_one(
                {"username": username},
                {"$set": {"is_active": is_active}}
            )
            
            if result.modified_count > 0:
                status = "activated" if is_active else "deactivated"
                return True, f"User {status} successfully"
            else:
                return False, "User not found or no changes made"
                
        except Exception as e:
            return False, f"Status update failed: {str(e)}"
    
    def get_all_users(self):
        """Get all users (admin function)"""
        if self.users_collection is None:
            return []
        
        try:
            users = list(self.users_collection.find(
                {},
                {"password": 0}  # Exclude passwords
            ).sort("registration_date", -1))
            
            return users
            
        except Exception as e:
            print(f"Error getting all users: {e}")
            return []
    
    def get_user_statistics(self):
        """Get user statistics (admin function)"""
        if self.users_collection is None:
            return {}
        
        try:
            total_users = self.users_collection.count_documents({})
            active_users = self.users_collection.count_documents({"is_active": True})
            admin_users = self.users_collection.count_documents({"role": "admin"})
            
            # Users registered in last 30 days
            thirty_days_ago = datetime.now() - timedelta(days=30)
            recent_users = self.users_collection.count_documents({
                "registration_date": {"$gte": thirty_days_ago}
            })
            
            return {
                "total_users": total_users,
                "active_users": active_users,
                "admin_users": admin_users,
                "recent_users": recent_users
            }
            
        except Exception as e:
            print(f"Error getting user statistics: {e}")
            return {}
    
    def create_admin_user(self, username, email, password, full_name):
        """Create an admin user"""
        if self.users_collection is None:
            return False, "Database connection failed"
        
        try:
            # Check if username or email already exists
            if self.users_collection.find_one({"username": username}):
                return False, "Username already exists"
            
            if self.users_collection.find_one({"email": email}):
                return False, "Email already registered"
            
            # Validate email
            if not self.validate_email(email):
                return False, "Invalid email format"
            
            # Validate password
            is_valid, message = self.validate_password(password)
            if not is_valid:
                return False, message
            
            # Create admin user
            user_data = {
                'username': username,
                'email': email,
                'password': self.hash_password(password),
                'full_name': full_name,
                'company': 'System Administrator',
                'registration_date': datetime.now(),
                'last_login': None,
                'is_active': True,
                'role': 'admin',
                'created_at': datetime.now().isoformat(),
                'permissions': ['read', 'write', 'admin']
            }
            
            result = self.users_collection.insert_one(user_data)
            if result.inserted_id:
                return True, "Admin user created successfully"
            else:
                return False, "Failed to create admin user"
                
        except Exception as e:
            return False, f"Admin creation failed: {str(e)}"
    
    def delete_user(self, username):
        """Delete a user (admin function)"""
        if self.users_collection is None:
            return False, "Database connection failed"
        
        try:
            result = self.users_collection.delete_one({"username": username})
            
            if result.deleted_count > 0:
                return True, f"User '{username}' deleted successfully"
            else:
                return False, "User not found"
                
        except Exception as e:
            return False, f"User deletion failed: {str(e)}"
    
    def update_user_profile(self, username, full_name=None, company=None, email=None):
        """Update user profile"""
        if self.users_collection is None:
            return False, "Database connection failed"
        
        try:
            update_data = {}
            
            if email and email != self.get_user_info(username).get('email'):
                # Check if new email is already taken
                if self.users_collection.find_one({"email": email, "username": {"$ne": username}}):
                    return False, "Email already in use"
                
                if not self.validate_email(email):
                    return False, "Invalid email format"
                
                update_data['email'] = email
            
            if full_name:
                update_data['full_name'] = full_name
            
            if company is not None:  # Allow empty string
                update_data['company'] = company
            
            if update_data:
                result = self.users_collection.update_one(
                    {"username": username},
                    {"$set": update_data}
                )
                
                if result.modified_count > 0:
                    return True, "Profile updated successfully"
                else:
                    return False, "No changes made"
            
            return True, "No changes to update"
            
        except Exception as e:
            return False, f"Update failed: {str(e)}"
    
    def change_password(self, username, old_password, new_password):
        """Change user password"""
        if self.users_collection is None:
            return False, "Database connection failed"
        
        try:
            user = self.users_collection.find_one({"username": username})
            
            if not user:
                return False, "User not found"
            
            # Verify old password
            if not self.verify_password(old_password, user['password']):
                return False, "Current password is incorrect"
            
            # Validate new password
            is_valid, message = self.validate_password(new_password)
            if not is_valid:
                return False, message
            
            # Update password
            result = self.users_collection.update_one(
                {"username": username},
                {"$set": {"password": self.hash_password(new_password)}}
            )
            
            if result.modified_count > 0:
                return True, "Password changed successfully"
            else:
                return False, "Failed to change password"
                
        except Exception as e:
            return False, f"Password change failed: {str(e)}"

    # ---------------- Password Reset (Forgot Password) ----------------
    def get_user_by_email(self, email: str):
        """Fetch a user document by email."""
        try:
            return self.users_collection.find_one({"email": email}) if self.users_collection else None
        except Exception:
            return None

    def set_password(self, username: str, new_password: str):
        """Set a new password for the user without verifying old password (used in reset)."""
        if self.users_collection is None:
            return False, "Database connection failed"
        try:
            # Validate new password
            is_valid, message = self.validate_password(new_password)
            if not is_valid:
                return False, message

            result = self.users_collection.update_one(
                {"username": username},
                {"$set": {"password": self.hash_password(new_password)},
                 "$unset": {"reset_token_hash": "", "reset_token_expiry": ""}}
            )
            if result.modified_count > 0:
                return True, "Password updated successfully"
            return False, "Failed to update password"
        except Exception as e:
            return False, f"Password update failed: {str(e)}"

    def create_password_reset_token(self, email: str, expiry_minutes: int = 60):
        """Create a password reset token for the user identified by email.

        Stores a bcrypt-hashed token and expiry timestamp on the user document.
        Returns (success, message, token) where token is the plain token to be emailed if success.
        """
        if self.users_collection is None:
            return False, "Database connection failed", None
        try:
            user = self.get_user_by_email(email)
            if not user:
                # Do not disclose user existence; return success-like message
                return True, "If the email exists, a reset link has been sent.", None

            token = secrets.token_urlsafe(32)
            token_hash = bcrypt.hashpw(token.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
            expiry = datetime.utcnow() + timedelta(minutes=expiry_minutes)

            self.users_collection.update_one(
                {"_id": user["_id"]},
                {"$set": {"reset_token_hash": token_hash, "reset_token_expiry": expiry}}
            )

            return True, "Password reset link generated.", token
        except Exception as e:
            return False, f"Failed to create reset token: {str(e)}", None

    def verify_password_reset_token(self, token: str):
        """Verify the provided reset token. Returns the username if valid, else None."""
        if self.users_collection is None:
            return None
        try:
            # Find users that have a reset token set and not expired
            now = datetime.utcnow()
            candidates = self.users_collection.find({
                "reset_token_hash": {"$exists": True},
                "reset_token_expiry": {"$gte": now}
            })
            for user in candidates:
                token_hash = user.get("reset_token_hash")
                if token_hash and bcrypt.checkpw(token.encode("utf-8"), token_hash.encode("utf-8")):
                    return user.get("username")
            return None
        except Exception:
            return None

    def reset_password_with_token(self, token: str, new_password: str):
        """Reset password using a valid token."""
        username = self.verify_password_reset_token(token)
        if not username:
            return False, "Invalid or expired reset link"
        return self.set_password(username, new_password)
    
    def save_campaign_metrics(self, username, campaign_data):
        """Save campaign performance metrics"""
        if self.metrics_collection is None:
            return False
        
        try:
            metric_data = {
                'username': username,
                'timestamp': datetime.now(),
                'campaign_data': campaign_data
            }
            
            result = self.metrics_collection.insert_one(metric_data)
            return result.inserted_id is not None
            
        except Exception as e:
            print(f"Error saving metrics: {e}")
            return False
    
    def get_user_metrics(self, username, limit=100):
        """Get user's campaign metrics"""
        if self.metrics_collection is None:
            return []
        
        try:
            metrics = list(self.metrics_collection.find(
                {"username": username}
            ).sort("timestamp", -1).limit(limit))
            
            return metrics
            
        except Exception as e:
            print(f"Error getting metrics: {e}")
            return []

# Global database instance (MongoDB required)
db_auth = UserAuthDB()
if db_auth.users_collection is None:
    raise RuntimeError(
        "MongoDB connection failed: 'users' collection unavailable. "
        "Set MONGODB_URI or ensure MongoDB is running at mongodb://localhost:27017/."
    )
