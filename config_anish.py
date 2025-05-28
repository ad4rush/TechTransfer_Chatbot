from dotenv import load_dotenv
import os
import random

load_dotenv()

# Collect all keys automatically
GOOGLE_API_KEYS = [
    "AIzaSyBerFBw6-9lW5YLTTNsDxkyZ8a87HtlXRw"
]

def get_random_google_api_key():
    if not GOOGLE_API_KEYS:
        raise ValueError("No Google API keys found in environment variables.")
    key = random.choice(GOOGLE_API_KEYS)
    print(f"Using API Key: {key[:10]}...")   # <-- only print first 10 characters for safety
    return key
