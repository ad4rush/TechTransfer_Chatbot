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

# GOOGLE_API_KEYS = [
#     "AIzaSyBJQ0Bx9lLrI-frjqccL6bAoIbgAKndL3w",
#     "AIzaSyBetDvBrQ5nyyeTU17LnNoMKaIZg1WUnns",
#     "AIzaSyAljRo1Om__f_ON8ZJqgp3xuiN25IdDHNE",
#     "AIzaSyAUJvzUo2A-M85vAuJWepw814-Mo8wV2Bo",
#     "AIzaSyDrbzIEKkVKDHGBbVtlhY1dl8UBh6UHeaQ",
#     "AIzaSyDUsD2zNZG7DbgR5aQuYXjYU9yZVQhVrzM",
#     "AIzaSyBQSCVT0MczCIyvTMCyrldjyA-mrtY2olg",
#     "AIzaSyAh-o5bV5I451_OrJxnWrZLsNdQMdg4nzU",
#     "AIzaSyBerFBw6-9lW5YLTTNsDxkyZ8a87HtlXRw",
#     "AIzaSyAgjY3GDWkg_31dYY68FNUAgHGI_UHyJG0",
#     "AIzaSyCz2FWb8XzJnCq4954GWpElJpUtN6_OZ18",
#     "AIzaSyA4jzNRcKs5TMrGiCEaJDbOlYysSn5JMlI",
#     "AIzaSyAVk_fFczjPs_LwU0J-Bhx8owEinovrvXY",
#     "AIzaSyCTXd751eeJes830BvEjQRl4J-oNwS-_VQ",
#     "AIzaSyDo3maj8vcb_pH8G0WLlTM5BZnHR2-xbI0",
#     "AIzaSyBvLQSL2eZtlViSO5hTDBACmLWHRx5QzgI",
#     "AIzaSyCH7IwiZoikXnxQHCPKiAzAkesWWV6VAC4",
#     "AIzaSyBbFPAhY0FY7_hZr9vdSoYORUQx3Pt9YtM",
#     "AIzaSyDPqKVRsKsoDrurLDOk2H5x87Sw7RrhVa0",
#     "AIzaSyDuYVtVkvcVzL3ag186Awuclv38yhjfdZ4",
#     "AIzaSyCHRjJBxqIpw-sFsMycr9LnvxrkpfD3IAA"
# ]


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

# --- Kaggle Secrets Integration ---
try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    # Assuming the secret is named 'GOOGLE_KEYS' and contains comma-separated keys
    google_keys_secret = user_secrets.get_secret("GOOGLE_KEYS")
    GOOGLE_API_KEYS = [key.strip() for key in google_keys_secret.split(',')]
except ImportError:
    print("Could not import UserSecretsClient. Assuming local development.")
    # Fallback for local development
    GOOGLE_API_KEYS = [
    "AIzaSyBJQ0Bx9lLrI-frjqccL6bAoIbgAKndL3w",
    "AIzaSyBetDvBrQ5nyyeTU17LnNoMKaIZg1WUnns",
    "AIzaSyAljRo1Om__f_ON8ZJqgp3xuiN25IdDHNE",
    "AIzaSyAUJvzUo2A-M85vAuJWepw814-Mo8wV2Bo",
    "AIzaSyDrbzIEKkVKDHGBbVtlhY1dl8UBh6UHeaQ",
    "AIzaSyDUsD2zNZG7DbgR5aQuYXjYU9yZVQhVrzM",
    "AIzaSyBQSCVT0MczCIyvTMCyrldjyA-mrtY2olg",
    "AIzaSyAh-o5bV5I451_OrJxnWrZLsNdQMdg4nzU",
    "AIzaSyBerFBw6-9lW5YLTTNsDxkyZ8a87HtlXRw",
    "AIzaSyAgjY3GDWkg_31dYY68FNUAgHGI_UHyJG0",
    "AIzaSyCz2FWb8XzJnCq4954GWpElJpUtN6_OZ18",
    "AIzaSyA4jzNRcKs5TMrGiCEaJDbOlYysSn5JMlI",
    "AIzaSyAVk_fFczjPs_LwU0J-Bhx8owEinovrvXY",
    "AIzaSyCTXd751eeJes830BvEjQRl4J-oNwS-_VQ",
    "AIzaSyDo3maj8vcb_pH8G0WLlTM5BZnHR2-xbI0",
    "AIzaSyBvLQSL2eZtlViSO5hTDBACmLWHRx5QzgI",
    "AIzaSyCH7IwiZoikXnxQHCPKiAzAkesWWV6VAC4",
    "AIzaSyBbFPAhY0FY7_hZr9vdSoYORUQx3Pt9YtM",
    "AIzaSyDPqKVRsKsoDrurLDOk2H5x87Sw7RrhVa0",
    "AIzaSyDuYVtVkvcVzL3ag186Awuclv38yhjfdZ4",
    "AIzaSyCHRjJBxqIpw-sFsMycr9LnvxrkpfD3IAA",
    "AIzaSyBtoCXXNIA3f6hqEXIHkxWKbj-QFuO0ly4",
    "AIzaSyDS0yBG34uzmnwXKNl_vo4Mvd4rT4tmDzw",
    "AIzaSyD0FlijzRAhziYJmuHnw91YUSAdizyPyVY",
    "AIzaSyB8o84Jj8uGVH28YQoaFD1uG0EdROQTv14",
    "AIzaSyBhSTnySkX3i-RzvlFvQnyWWo481HHg4RY",
    "AIzaSyBGTzC--Sir8HQs28nmKNQbghRh3PWbphA",
    "AIzaSyDi7VgW9f0CvZB7vsqixFseDGtpmwmxh2w",
    "AIzaSyBGXUgC4mmcWPBb-_wK7fhlwq0JfNLc5Do",
    "AIzaSyArA36LYFSgSoE3PBkFb-MGM8wdMEnKtps",
    "AIzaSyBzcDGAcg1ARQiFyl48WHOSs9Vx9yw8iCM",
    "AIzaSyC9NKmnZzshXBRddxwaz2SEHBI5K0JG9p4",
    "AIzaSyD0VE9DuSp5uEJLQ3Qvkybrl1yzmm4F0QY",
    "AIzaSyCVuD8VP44bK3jlKBurcRmleix-Z2I2pXk",
    "AIzaSyB1IUIR8GTJ94pnXlK-QVq7P43m7QBjnWI",
    "AIzaSyCKwbWhh2Yp3La5GAnsR2LPYVIYgqGXjCQ",
    "AIzaSyAal8PgdWThvIvj-33d5jkpNq4HGY7AeSI",
    "AIzaSyCLmbTGu5HMsdj-i0rnFoBjBt0dIndnmdQ",
    "AIzaSyC8i0MHHVAMUfO8FLMDXW3fHu9KQKDqnV0",
    "AIzaSyA0qXJjV5G4DJaHQxMzaaTg7CZ0RGS2GX8",
    "AIzaSyCLnvR6M3W-lTPcsjn0A-jtH3o1E8SS1fk",
    "AIzaSyBtP-6yk_q8Wq3o4HUQAgoiy8d5BSp7wwU",
    "AIzaSyCR0gN3lrpBC6nrB8rzG5IA1O_OFpIgFkM",
    "AIzaSyD3yZr4T2YSUGsSFR9LvtiZLiG4yu8ZLGc",
    "AIzaSyAMzHNDZkJ0Cj0l-FrjUyTI1EKQ73e39tY",
    "AIzaSyAw-WZeqqDHgcRCSvaBD_LNqPIB3VQ-Fx4",
    "AIzaSyDBL3mZRlnqtsm8CnZod_PqqZnf7uxZOzg",
    "AIzaSyD4QcmoeXGiVtKY0Pm0R0ZSSYhr8oclc3Y",
    "AIzaSyCb74hJgfDw5mN-BITSZNwbKM8bgB2W_9w",
    "AIzaSyAwWoTdLd_zQvWAg86nNWwHs73ATax9C4w",
    "AIzaSyAaTWjodfhTfGYDRHBjPAix6LJlSVDy0Nk",
    "AIzaSyCsDp6IPnydUVvE0eGwsbbslw44vdQ4zY4",
    "AIzaSyDsRqz9cVkzq7bxMQlSz2QBwKivP8NPwM8",
    "AIzaSyBvfquHSbHpLIMGYSc_iySlVhmaO44WqsY",
    "AIzaSyA50LiYS3wM150TJym1TsDCP7voYRl0ZGw",
    "AIzaSyBUfDc7dFelcPylLY18SS4LUKWGFy9Y3hE",
    "AIzaSyC01jTdbDMlRhjC-KSUUFVwBn7eSBF0P9E",
    "AIzaSyBq2HhZJQTmqNvJKtf37KISFy_XMYAMDlQ",
    "AIzaSyDowlk2uu3sBldkWNRyQyIOBwNX6E3J3yQ",
    "AIzaSyBB1rUoMahp6gC-YNo6o-r9IuxdQtoNHcI",
    "AIzaSyBg8_a0QMLwq5f6ztjTan27sNZorYJvecA",
    "AIzaSyB1TgUmwIDOPkWf2PjRIhn8NoNDxkgb1uc",
    "AIzaSyARJYamt5iBIOlu5D7skEzIOQBKss4vtFQ",
    "AIzaSyBGX61_wM-Wl_DHDGKpNxYtq16q2n5sUDU",
    "AIzaSyBoEfYHw7Az23c4ls4YvLxOKdY4Pg3ZWsY",
    "AIzaSyAE97M6iVGPP7PlgAuFpOPR8Y9McXPVdPg",
    "AIzaSyDisPti0tjQ3UzZ5v1vo0eqQVZ4pksLB9Q",
    "AIzaSyCWATzVPxsrhZ6WYz7PS0AinaO_6jwEGZo",
    "AIzaSyCY26z6GYD6d3KbaRERSZQ973_gOFbeA6Y",
    "AIzaSyCGgqP5UbcjFBB1uG2INeRlrkYzN6ROTDA",
    "AIzaSyDzCL0r_EacSzXWJQnW-lx97hlkz6vR8ck",
    "AIzaSyDaRKCeGOKglRD6KNQ5qO9LFQpQu1CvKJw",
    "AIzaSyACuPziZkivR13TbFKC5sIHDTFmAp73E54",
    "AIzaSyCY7pWer1YjwDwacEcm-kfenuTt13xx81Y",
    "AIzaSyBFW_T64GnlSZMbjQGmq_1-L2RXqgXMtqg",
    "AIzaSyDY1yDTir2fvQrOSTxCG6cMxuD1E9In03c",
    "AIzaSyBTsrtIJelB8tL0-4SrwxviKtvAF-VD6Wk",
    "AIzaSyBJz57MQrBBtC8K3LR6ziM6uBovebr-uG0",
    "AIzaSyAdYLnh4dN7qijRXMQdfT5SXN4gnzAhVxo",
    "AIzaSyD_5TPGB7r1Sfwfc4HwNvsh8u-mncrQEgA",
    "AIzaSyDIvJmD3sjNv64m_RVks4IdmqxaFwzTRUQ",
    "AIzaSyD1BAJiV-MkKN0lwZrdmeEqKuiz4E0AZ6A",
    "AIzaSyBiFale6WScIZItScwPdNQXv5GbJk7cixE",
    "AIzaSyAWgj9YI50a7Pvmyk4KB5M7VGe4hmILG8M",
    "AIzaSyAO3LWh-g-IdxILumAI9AIq3X4IXySq8hM",
    "AIzaSyAsHArHOJOmSVb5T0Yo0ZDsaHhpYcdlduM",
    "AIzaSyDySxLphLH2jEQVKDAAv69-x_YBkxaw9KM",
    "AIzaSyC0iZ8Rnlrw84-yuS7WhnfNHC0Sl_mTybY"
]

except Exception as e:
    print(f"An error occurred while accessing Kaggle secrets: {e}")
    GOOGLE_API_KEYS = []


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
    Gets an available API key. This function will now WAIT indefinitely until a key is available.
    1. It tries to find a key that is not in a 429-induced resting period and has not hit its local rate limit.
    2. If no key is immediately available, it calculates the soonest a key will be ready and waits.
    3. If all keys are rate-limited (but not in a 429 cooldown), it waits for 60 seconds as requested.
    """
    if not GOOGLE_API_KEYS:
        logging.critical("CRITICAL: No Google API keys configured.")
        raise ValueError("No Google API keys found in config.py.")

    while True: # Keep looping until a key is found
        with _KEY_MANAGER_LOCK:
            current_time = time.time()
            eligible_keys = []

            # --- Find an eligible key ---
            for key in GOOGLE_API_KEYS:
                # Check 1: Is the key in a 429-induced cooldown?
                if _API_KEYS_STATE[key]["resting_until"] > current_time:
                    continue

                # Check 2: Has the key hit its local rate limit?
                _API_CALL_HISTORY[key] = [t for t in _API_CALL_HISTORY[key] if current_time - t < RATE_WINDOW]
                if len(_API_CALL_HISTORY[key]) < CALLS_PER_MINUTE:
                    eligible_keys.append(key)
            
            # --- If a key is found, dispense it ---
            if eligible_keys:
                # Prioritize keys with fewest calls, then by least recent use
                eligible_keys.sort(key=lambda k: (len(_API_CALL_HISTORY[k]), _API_KEYS_STATE[k]["last_used_success"]))
                chosen_key = eligible_keys[0]
                
                # Record usage *before* returning the key
                _record_key_usage_internal(chosen_key) 
                
                logging.info(f"Dispensing API Key ...{chosen_key[-6:]} (Call history size now: {len(_API_CALL_HISTORY[chosen_key])})")
                return chosen_key

        # --- If no key was found, wait intelligently ---
        logging.warning("No API key immediately available. Entering wait state...")
        
        with _KEY_MANAGER_LOCK:
            resting_keys_expiry = [
                s["resting_until"] for s in _API_KEYS_STATE.values() 
                if s["resting_until"] > time.time()
            ]

        if resting_keys_expiry:
            # If at least one key is in a 429-cooldown, wait for the first one to expire.
            earliest_expiry = min(resting_keys_expiry)
            wait_duration = max(0, earliest_expiry - time.time()) + 1.5 # Add a 1.5s buffer
            logging.info(f"Waiting for {wait_duration:.2f} seconds for the earliest key to finish its cooldown.")
            time.sleep(wait_duration)
        else:
            # If no keys are in cooldown, it means they are all just locally rate-limited.
            # This is the "all keys are dead" scenario. Wait for 1 minute as requested.
            wait_duration = 60
            logging.warning(f"All keys are currently busy with local rate limits. Waiting for {wait_duration} seconds.")
            time.sleep(wait_duration)
