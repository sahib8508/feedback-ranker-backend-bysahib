import os
import firebase_admin
from firebase_admin import credentials, firestore
import json
import logging

def initialize_firebase():
    """Initialize Firebase with environment variables"""
    try:
        # Check if Firebase is already initialized
        if firebase_admin._apps:
            return firestore.client()
        
        # Get private key and handle newlines properly
        private_key = os.getenv('FIREBASE_PRIVATE_KEY')
        if private_key:
            private_key = private_key.replace('\\n', '\n')
        
        # Create service account dict from environment variables
        service_account_info = {
            "type": "service_account",
            "project_id": os.getenv('FIREBASE_PROJECT_ID'),
            "private_key_id": os.getenv('FIREBASE_PRIVATE_KEY_ID'),
            "private_key": private_key,
            "client_email": os.getenv('FIREBASE_CLIENT_EMAIL'),
            "client_id": os.getenv('FIREBASE_CLIENT_ID'),
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "client_x509_cert_url": os.getenv('FIREBASE_CLIENT_CERT_URL'),
            "universe_domain": "googleapis.com"
        }
        
        # Validate that we have all required fields
        required_fields = ['project_id', 'private_key', 'client_email']
        missing_fields = [field for field in required_fields if not service_account_info.get(field)]
        
        if missing_fields:
            logging.error(f"Missing Firebase configuration fields: {missing_fields}")
            return None
        
        # Initialize Firebase
        cred = credentials.Certificate(service_account_info)
        firebase_admin.initialize_app(cred)
        
        logging.info("Firebase initialized successfully")
        return firestore.client()
        
    except Exception as e:
        logging.error(f"Error initializing Firebase: {str(e)}")
        return None