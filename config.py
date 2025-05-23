# config.py
from dotenv import load_dotenv
import os
import random
import time # For sleep in case of key exhaustion

load_dotenv()

# Using the list of API keys you provided
GOOGLE_API_KEYS = [
    "AIzaSyBJQ0Bx9lLrI-frjqccL6bAoIbgAKndL3w", 
    "AIzaSyBetDvBrQ5nyyeTU17LnNoMKaIZg1WUnns",  
    "AIzaSyAljRo1Om__f_ON8ZJqgp3xuiN25IdDHNE", 
    "AIzaSyAUJvzUo2A-M85vAuJWepw814-Mo8wV2Bo", 
    "AIzaSyDrbzIEKkVKDHGBbVtlhY1dl8UBh6UHeaQ", 
    "AIzaSyDUsD2zNZG7DbgR5aQuYXjYU9yZVQhVrzM",
    "AIzaSyBQSCVT0MczCIyvTMCyrldjyA-mrtY2olg",
    "AIzaSyAh-o5bV5I451_OrJxnWrZLsNdQMdg4nzU",
    "AIzaSyBerFBw6-9lW5YLTTNsDxkyZ8a87HtlXRw"
]
# Ensure no duplicates if any were present in the original list
GOOGLE_API_KEYS = sorted(list(set(GOOGLE_API_KEYS)))

# To manage key rotation and avoid immediate reuse in rapid succession by threads
# This is a simple approach; more sophisticated locking could be used if contention is high.
# For now, random choice from a larger pool is the primary strategy.
# A more advanced key manager could track last used time or failure counts.

USED_KEYS_TRACKER = {key: 0 for key in GOOGLE_API_KEYS} # Tracks how many times a key is picked

def get_random_google_api_key(exclude_keys=None):
    """
    Gets a random Google API key, optionally excluding some.
    Tries to pick less recently used keys if possible (simple heuristic).
    """
    if not GOOGLE_API_KEYS:
        raise ValueError("No Google API keys found in GOOGLE_API_KEYS list in config.py.")

    available_keys = GOOGLE_API_KEYS
    if exclude_keys:
        available_keys = [key for key in GOOGLE_API_KEYS if key not in exclude_keys]

    if not available_keys:
        # This might happen if all keys are excluded, or if the initial list was empty.
        # Could implement a wait or raise a more specific error.
        # For now, let's fall back to the full list if exclusion leads to empty.
        if exclude_keys and GOOGLE_API_KEYS:
            print(f"Warning: All keys were in exclude_keys list, or no non-excluded keys available. Picking from full list again.")
            available_keys = GOOGLE_API_KEYS
        else: # Should not happen if GOOGLE_API_KEYS is populated.
             raise ValueError("No available API keys to choose from after exclusion.")


    # Simple heuristic: pick the key that has been "used" the fewest times
    # This is a basic attempt to distribute load if keys have different quotas
    # or if some get rate-limited.
    # For true parallel safety and more even distribution, a proper queue/pool manager for keys would be better.
    key = min(available_keys, key=lambda k: USED_KEYS_TRACKER.get(k, 0))
    USED_KEYS_TRACKER[key] = USED_KEYS_TRACKER.get(key, 0) + 1
    
    print(f"Using API Key: {key[:10]}... (Usage count: {USED_KEYS_TRACKER[key]})")
    return key