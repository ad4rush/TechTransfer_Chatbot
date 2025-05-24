# FILE: config.py
# Manages API keys, rate limits, and cool-downs.
# get_available_api_key fails fast if no key is immediately ready,
# and records usage internally before returning a key.

from dotenv import load_dotenv
import os
import random
import time
import logging
from collections import defaultdict
from threading import RLock
import re # For parsing retry_delay

load_dotenv()
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - [%(filename)s.%(funcName)s:%(lineno)d] - %(message)s')

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
GOOGLE_API_KEYS = sorted(list(set(GOOGLE_API_KEYS)))

_API_KEYS_STATE = {
    key: {"usage_count": 0, "resting_until": 0.0, "last_used_success": 0.0}
    for key in GOOGLE_API_KEYS
}
# Stores timestamps of when a key's usage was recorded for an API call
_API_CALL_HISTORY = defaultdict(list)
_KEY_MANAGER_LOCK = RLock() # Re-entrant lock

CALLS_PER_MINUTE = 4  # As per your API limit
RATE_WINDOW = 60  # seconds
DEFAULT_KEY_RESTING_DURATION_SECONDS = 60 # Default if API doesn't specify retry_delay

def parse_retry_delay(error_message_str):
    """Parses the retry_delay from a Google API error message string if present."""
    if error_message_str:
        match = re.search(r"retry_delay {\s*seconds: (\d+)\s*}", error_message_str)
        if match:
            return int(match.group(1))
    return None

def mark_key_as_rate_limited(api_key, error_message_str=None):
    """
    Marks a key as resting due to a 429 rate limit error.
    Uses API's retry_delay if available in the error_message_str.
    """
    with _KEY_MANAGER_LOCK:
        if api_key in _API_KEYS_STATE:
            retry_after_seconds = DEFAULT_KEY_RESTING_DURATION_SECONDS
            parsed_delay = parse_retry_delay(error_message_str)

            if parsed_delay is not None:
                retry_after_seconds = parsed_delay
                logging.info(f"Key ...{api_key[-6:]} got 429. API suggests retry after {retry_after_seconds}s.")
            else:
                logging.info(f"Key ...{api_key[-6:]} got 429. No specific retry_delay found in error, using default {retry_after_seconds}s.")

            # Add a small buffer (e.g., 1-2 seconds) to the API's suggested delay or default
            actual_rest_duration = retry_after_seconds + 2
            cooldown_expiry = time.time() + actual_rest_duration
            
            _API_KEYS_STATE[api_key]["resting_until"] = cooldown_expiry
            _API_CALL_HISTORY[api_key] = [] # Clear call history as it's resting due to external limit
            logging.warning(f"Key ...{api_key[-6:]} marked as resting until {time.ctime(cooldown_expiry)} (total rest: {actual_rest_duration}s). Call history cleared.")
        else:
            logging.warning(f"Attempted to mark unknown key ...{api_key[-6:]} as rate limited.")

def _record_key_usage_internal(api_key):
    """
    Internal helper to record key usage. Called when a key is dispensed.
    Assumes _KEY_MANAGER_LOCK is already held by the caller (get_available_api_key).
    """
    current_time = time.time()
    _API_CALL_HISTORY[api_key].append(current_time)
    # Clean up old calls from history immediately
    _API_CALL_HISTORY[api_key] = [
        call_time for call_time in _API_CALL_HISTORY[api_key]
        if current_time - call_time < RATE_WINDOW
    ]
    # This debug log can be verbose, adjust level if needed
    # logging.debug(f"Key ...{api_key[-6:]} usage internally recorded. Call history size for key: {len(_API_CALL_HISTORY[api_key])}")

def record_key_success(api_key):
    """Records metadata for a successful API call (e.g., usage count)."""
    with _KEY_MANAGER_LOCK:
        if api_key in _API_KEYS_STATE:
            _API_KEYS_STATE[api_key]["usage_count"] += 1
            _API_KEYS_STATE[api_key]["last_used_success"] = time.time()
            logging.debug(f"Key ...{api_key[-6:]} successful call processed. Total usage: {_API_KEYS_STATE[api_key]['usage_count']}.")

def get_available_api_key():
    """
    Gets an available API key that is:
    1. Not currently in a 429-induced resting period (`resting_until`).
    2. Has made fewer than CALLS_PER_MINUTE in the last RATE_WINDOW (based on `_API_CALL_HISTORY`).
    If no key is immediately available, raises ValueError.
    Usage is recorded internally before returning the key.
    """
    if not GOOGLE_API_KEYS:
        logging.critical("CRITICAL: No Google API keys configured.")
        raise ValueError("No Google API keys found in config.py.")

    with _KEY_MANAGER_LOCK: # Ensure atomicity of checking and recording usage
        current_time = time.time()
        eligible_keys = []

        for key in GOOGLE_API_KEYS:
            # Check 1: Is the key 429-resting?
            if _API_KEYS_STATE[key]["resting_until"] > current_time:
                # logging.debug(f"Key ...{key[-6:]} is 429-resting until {time.ctime(_API_KEYS_STATE[key]['resting_until'])}. Skipping.")
                continue

            # Check 2: Local rate limit (CALLS_PER_MINUTE)
            # Clean history for this key before checking (already done when recording usage, but good for robustness)
            _API_CALL_HISTORY[key] = [
                call_time for call_time in _API_CALL_HISTORY[key]
                if current_time - call_time < RATE_WINDOW
            ]
            if len(_API_CALL_HISTORY[key]) < CALLS_PER_MINUTE:
                eligible_keys.append(key)
            else:
                # logging.debug(f"Key ...{key[-6:]} has {len(_API_CALL_HISTORY[key])} calls in window (limit {CALLS_PER_MINUTE}). Skipping.")
                continue
        
        if eligible_keys:
            # Prioritize keys with fewest calls in history, then by least recent successful use
            eligible_keys.sort(key=lambda k: (len(_API_CALL_HISTORY[k]), _API_KEYS_STATE[k]["last_used_success"]))
            chosen_key = eligible_keys[0]
            
            # Record usage for the chosen key *before* returning it
            _record_key_usage_internal(chosen_key) 
            
            logging.info(f"Dispensing API Key ...{chosen_key[-6:]} (Call history size now: {len(_API_CALL_HISTORY[chosen_key])})")
            return chosen_key

    # If no key was found after checking all (eligible_keys is empty)
    logging.warning("get_available_api_key: No API key immediately available (all are resting or locally rate-limited).")
    raise ValueError("All API keys are currently busy or resting.")
