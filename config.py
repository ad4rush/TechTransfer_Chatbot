# config.py
from dotenv import load_dotenv
import os
import random
import time
import logging
from collections import defaultdict
from threading import RLock # Re-entrant lock for managing key state

load_dotenv()
# Configure logging at the root level if not already done by other modules
# This ensures that if config.py is imported first, logging is set up.
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - [%(module)s.%(funcName)s:%(lineno)d] - %(message)s')

# Use the keys provided by the user
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
GOOGLE_API_KEYS = sorted(list(set(GOOGLE_API_KEYS))) # Remove duplicates and sort

_API_KEYS_STATE = {
    key: {"usage_count": 0, "resting_until": 0.0, "last_used_success": 0.0} 
    for key in GOOGLE_API_KEYS
}
_API_CALL_HISTORY = defaultdict(list)
_KEY_MANAGER_LOCK = RLock() 
GLOBAL_COOLDOWN_SECONDS = 5
KEY_RESTING_DURATION_SECONDS = 60 # Rest a key for 2 minutes after a 429

# Add this near the top of config.py with other constants
MAX_IMAGE_WORKERS = 3  # Maximum number of concurrent workers for image processing

CALLS_PER_MINUTE = 4  # Maximum calls allowed per minute per key
RATE_WINDOW = 60  # Window size in seconds (1 minute)
KEY_COOLDOWN = 15  # Cooldown time in seconds after hitting rate limit

def mark_key_as_rate_limited(api_key):
    """Marks a key as resting due to a rate limit error."""
    with _KEY_MANAGER_LOCK:
        if api_key in _API_KEYS_STATE:
            cooldown_expiry = time.time() + KEY_RESTING_DURATION_SECONDS
            _API_KEYS_STATE[api_key]["resting_until"] = cooldown_expiry
            # Do not increment usage_count on failure for rate limit, as it wasn't a "successful" use for quota.
            logging.warning(f"Key ...{api_key[-6:]} marked as resting until {time.ctime(cooldown_expiry)} (Rate Limit).")
        else:
            logging.warning(f"Attempted to mark unknown key ...{api_key[-6:]} as rate limited.")

def track_api_call(api_key):
    """Track API call and return True if rate limit exceeded"""
    current_time = time.time()
    
    with _KEY_MANAGER_LOCK:
        # Remove calls older than the rate window
        _API_CALL_HISTORY[api_key] = [
            call_time for call_time in _API_CALL_HISTORY[api_key]
            if current_time - call_time < RATE_WINDOW
        ]
        
        # Check if we've exceeded rate limit
        if len(_API_CALL_HISTORY[api_key]) >= CALLS_PER_MINUTE:
            return True
        
        # Add new call
        _API_CALL_HISTORY[api_key].append(current_time)
        return False

def get_available_api_key(exclude_keys=None):
    """
    Gets an available API key.
    Prioritizes keys not resting and then by least total usage and then least recent successful use.
    If all keys are resting, waits for GLOBAL_COOLDOWN_SECONDS and tries again up to max_global_waits.
    Raises ValueError if GOOGLE_API_KEYS is empty or if no key can be found after waiting.
    """
    if not GOOGLE_API_KEYS:
        logging.critical("CRITICAL: No Google API keys configured.")
        raise ValueError("No Google API keys found in config.py.")

    excluded_set = set(exclude_keys or [])
    current_time = time.time()
    
    with _KEY_MANAGER_LOCK:
        # Find a key that hasn't exceeded rate limits
        for key in GOOGLE_API_KEYS:
            if key in excluded_set:
                continue
                
            # Clean up old calls
            _API_CALL_HISTORY[key] = [
                call_time for call_time in _API_CALL_HISTORY[key]
                if current_time - call_time < RATE_WINDOW
            ]
            
            # Check if key has capacity
            if len(_API_CALL_HISTORY[key]) < CALLS_PER_MINUTE:
                return key
                
        # If no keys available, find the one that will be available soonest
        soonest_available = None
        min_wait_time = float('inf')
        
        for key in GOOGLE_API_KEYS:
            if key in excluded_set:
                continue
                
            if _API_CALL_HISTORY[key]:
                wait_time = _API_CALL_HISTORY[key][0] + RATE_WINDOW - current_time
                if wait_time < min_wait_time:
                    min_wait_time = wait_time
                    soonest_available = key
        
        if soonest_available:
            time.sleep(min_wait_time + 1)  # Add 1 second buffer
            return soonest_available
            
        raise ValueError("All API keys are excluded or rate limited")

def record_key_success(api_key):
    """To be called after a successful API call with a key."""
    if track_api_call(api_key):
        mark_key_as_rate_limited(api_key)
    with _KEY_MANAGER_LOCK:
        if api_key in _API_KEYS_STATE:
            _API_KEYS_STATE[api_key]["usage_count"] += 1
            _API_KEYS_STATE[api_key]["last_used_success"] = time.time()
            # If a key was resting but succeeded, we can optionally reset its resting_until early.
            # For now, let it naturally expire to avoid immediate re-use if it was borderline.
            # _API_KEYS_STATE[api_key]["resting_until"] = 0.0 
            logging.debug(f"Key ...{api_key[-6:]} recorded as successful. New usage count: {_API_KEYS_STATE[api_key]['usage_count']}.")

# Kept for compatibility, now redirects to the new smarter function.
def get_random_google_api_key(exclude_keys=None):
    return get_available_api_key(exclude_keys=exclude_keys)