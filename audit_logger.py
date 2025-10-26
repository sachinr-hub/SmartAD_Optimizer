import json
import os
from datetime import datetime
from typing import Dict, Any, List

class AuditLogger:
    def __init__(self, log_file: str = "data/audit_log.json"):
        self.log_file = log_file
        self.ensure_log_file_exists()
    
    def ensure_log_file_exists(self):
        """Ensure the audit log file exists"""
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w') as f:
                json.dump([], f)
    
    def log_action(self, 
                   username: str, 
                   action: str, 
                   resource: str = None, 
                   details: Dict[str, Any] = None,
                   ip_address: str = None,
                   user_agent: str = None):
        """Log an audit action"""
        try:
            # Load existing logs
            with open(self.log_file, 'r') as f:
                logs = json.load(f)
            
            # Create new log entry
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'username': username,
                'action': action,
                'resource': resource,
                'details': details or {},
                'ip_address': ip_address,
                'user_agent': user_agent,
                'session_id': self._get_session_id()
            }
            
            logs.append(log_entry)
            
            # Keep only last 10000 entries to prevent file from growing too large
            if len(logs) > 10000:
                logs = logs[-10000:]
            
            # Save back to file
            with open(self.log_file, 'w') as f:
                json.dump(logs, f, indent=2, default=str)
            
            return True
            
        except Exception as e:
            print(f"Error logging audit action: {e}")
            return False
    
    def get_logs(self, 
                 username: str = None, 
                 action: str = None, 
                 limit: int = 100,
                 start_date: str = None,
                 end_date: str = None) -> List[Dict[str, Any]]:
        """Get audit logs with optional filtering"""
        try:
            with open(self.log_file, 'r') as f:
                logs = json.load(f)
            
            # Apply filters
            filtered_logs = logs
            
            if username:
                filtered_logs = [log for log in filtered_logs if log.get('username') == username]
            
            if action:
                filtered_logs = [log for log in filtered_logs if log.get('action') == action]
            
            if start_date:
                filtered_logs = [log for log in filtered_logs if log.get('timestamp', '') >= start_date]
            
            if end_date:
                filtered_logs = [log for log in filtered_logs if log.get('timestamp', '') <= end_date]
            
            # Sort by timestamp (newest first) and limit
            filtered_logs.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
            
            return filtered_logs[:limit]
            
        except Exception as e:
            print(f"Error getting audit logs: {e}")
            return []
    
    def get_user_activity_summary(self, username: str, days: int = 30) -> Dict[str, Any]:
        """Get user activity summary for the last N days"""
        try:
            from datetime import timedelta
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            logs = self.get_logs(
                username=username,
                start_date=start_date.isoformat(),
                end_date=end_date.isoformat(),
                limit=1000
            )
            
            # Count actions by type
            action_counts = {}
            for log in logs:
                action = log.get('action', 'unknown')
                action_counts[action] = action_counts.get(action, 0) + 1
            
            # Get unique days with activity
            active_days = set()
            for log in logs:
                timestamp = log.get('timestamp', '')
                if timestamp:
                    try:
                        date = datetime.fromisoformat(timestamp.replace('Z', '+00:00')).date()
                        active_days.add(date.isoformat())
                    except:
                        pass
            
            return {
                'total_actions': len(logs),
                'action_counts': action_counts,
                'active_days': len(active_days),
                'period_days': days
            }
            
        except Exception as e:
            print(f"Error getting user activity summary: {e}")
            return {}
    
    def get_system_activity_summary(self, days: int = 7) -> Dict[str, Any]:
        """Get system-wide activity summary"""
        try:
            from datetime import timedelta
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            logs = self.get_logs(
                start_date=start_date.isoformat(),
                end_date=end_date.isoformat(),
                limit=5000
            )
            
            # Count by action type
            action_counts = {}
            # Count by user
            user_counts = {}
            # Count by day
            daily_counts = {}
            
            for log in logs:
                action = log.get('action', 'unknown')
                username = log.get('username', 'unknown')
                timestamp = log.get('timestamp', '')
                
                action_counts[action] = action_counts.get(action, 0) + 1
                user_counts[username] = user_counts.get(username, 0) + 1
                
                if timestamp:
                    try:
                        date = datetime.fromisoformat(timestamp.replace('Z', '+00:00')).date()
                        date_str = date.isoformat()
                        daily_counts[date_str] = daily_counts.get(date_str, 0) + 1
                    except:
                        pass
            
            # Get top users and actions
            top_users = sorted(user_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            top_actions = sorted(action_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            
            return {
                'total_actions': len(logs),
                'unique_users': len(user_counts),
                'top_users': top_users,
                'top_actions': top_actions,
                'daily_counts': daily_counts,
                'period_days': days
            }
            
        except Exception as e:
            print(f"Error getting system activity summary: {e}")
            return {}
    
    def export_logs(self, 
                    filename: str = None, 
                    format: str = 'json',
                    **filters) -> str:
        """Export audit logs to file"""
        try:
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"audit_export_{timestamp}.{format}"
            
            logs = self.get_logs(limit=10000, **filters)
            
            if format.lower() == 'json':
                with open(filename, 'w') as f:
                    json.dump(logs, f, indent=2, default=str)
            elif format.lower() == 'csv':
                import csv
                with open(filename, 'w', newline='') as f:
                    if logs:
                        writer = csv.DictWriter(f, fieldnames=logs[0].keys())
                        writer.writeheader()
                        writer.writerows(logs)
            
            return filename
            
        except Exception as e:
            print(f"Error exporting logs: {e}")
            return None
    
    def _get_session_id(self) -> str:
        """Get current session ID (simplified implementation)"""
        try:
            import streamlit as st
            return getattr(st.session_state, 'session_id', 'unknown')
        except:
            return 'unknown'

# Global audit logger instance
audit_logger = AuditLogger()

# Convenience functions for common audit actions
def log_login(username: str, success: bool = True):
    """Log user login attempt"""
    action = "login_success" if success else "login_failed"
    audit_logger.log_action(username, action, "authentication")

def log_logout(username: str):
    """Log user logout"""
    audit_logger.log_action(username, "logout", "authentication")

def log_admin_action(username: str, action: str, target_user: str = None, details: dict = None):
    """Log admin action"""
    resource = f"user:{target_user}" if target_user else "system"
    audit_logger.log_action(username, f"admin_{action}", resource, details)

def log_model_action(username: str, action: str, model_name: str = None, details: dict = None):
    """Log model-related action"""
    resource = f"model:{model_name}" if model_name else "model"
    audit_logger.log_action(username, f"model_{action}", resource, details)

def log_data_action(username: str, action: str, data_type: str = None, details: dict = None):
    """Log data-related action"""
    resource = f"data:{data_type}" if data_type else "data"
    audit_logger.log_action(username, f"data_{action}", resource, details)

def log_config_action(username: str, action: str, config_section: str = None, details: dict = None):
    """Log configuration action"""
    resource = f"config:{config_section}" if config_section else "config"
    audit_logger.log_action(username, f"config_{action}", resource, details)
