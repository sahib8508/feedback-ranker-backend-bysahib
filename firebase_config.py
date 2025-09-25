# firebase_config.py
import os
import firebase_admin
from firebase_admin import credentials, firestore
import json

def initialize_firebase():
    """Initialize Firebase with environment variables"""
    try:
        # Create service account dict from environment variables
        service_account_info = {
            "type": "service_account",
            "project_id": os.getenv('FIREBASE_PROJECT_ID'),
            "private_key_id": os.getenv('FIREBASE_PRIVATE_KEY_ID'),
            "private_key": os.getenv('FIREBASE_PRIVATE_KEY').replace('\\n', '\n'),
            "client_email": os.getenv('FIREBASE_CLIENT_EMAIL'),
            "client_id": os.getenv('FIREBASE_CLIENT_ID'),
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "client_x509_cert_url": os.getenv('FIREBASE_CLIENT_CERT_URL'),
            "universe_domain": "googleapis.com"
        }
        
        # Initialize Firebase only if not already initialized
        if not firebase_admin._apps:
            cred = credentials.Certificate(service_account_info)
            firebase_admin.initialize_app(cred)
        
        # Return Firestore client
        return firestore.client()
        
    except Exception as e:
        print(f"Error initializing Firebase: {e}")
        return None