# api_key_checker.py
import google.generativeai as genai
import logging
import time

# Attempt to import GOOGLE_API_KEYS from the existing config.py
try:
    from config import GOOGLE_API_KEYS
except ImportError:
    print("Error: Could not import GOOGLE_API_KEYS from config.py.")
    print("Please ensure config.py is in the same directory or accessible in your PYTHONPATH,")
    print("and that it contains a list named GOOGLE_API_KEYS.")
    exit(1)
except AttributeError:
    print("Error: GOOGLE_API_KEYS list not found in config.py.")
    print("Please ensure config.py defines a list named GOOGLE_API_KEYS.")
    exit(1)


# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Store results
key_status = {}

def check_api_key(api_key):
    """
    Checks a single API key by trying to list models.
    Returns True if successful, False otherwise, along with a status message.
    """
    api_key_display = f"{api_key[:10]}...{api_key[-4:]}" if len(api_key) > 14 else api_key
    try:
        # Configure the genai library with the current key
        # This is a global configuration, so each check reconfigures it.
        genai.configure(api_key=api_key)
        
        # Attempt a lightweight API call (e.g., listing models)
        # This verifies authentication and basic API access.
        models = [m.name for m in genai.list_models()]
        
        if models:
            # Check if a common model like 'gemini-pro' or 'gemini-flash' (or their variants) is available
            # This is a stronger check than just getting an empty list or a non-standard list.
            # 'gemini-2.0-flash-exp' is used in your chatbot. Let's check for something similar.
            # We'll check for a model that should generally be listable.
            # Example: 'models/gemini-pro' or a flash model.
            # For 'gemini-2.0-flash-exp', the actual model name in `list_models` might be different.
            # Let's check for 'gemini-pro' as a generally available one.
            if any('gemini-pro' in model_name for model_name in models) or \
               any('flash' in model_name for model_name in models):
                status_message = f"API Key {api_key_display}: WORKING (Successfully listed models)."
                logging.info(status_message)
                return True, status_message
            else:
                status_message = f"API Key {api_key_display}: POTENTIAL ISSUE (Listed models, but common models not found in list: {models[:3]}...). Key might have restricted access."
                logging.warning(status_message)
                return False, status_message # Treat as potential issue rather than hard fail for now

        else:
            status_message = f"API Key {api_key_display}: FAILED (Listed models but the list was empty)."
            logging.warning(status_message)
            return False, status_message
            
    except Exception as e:
        error_message = str(e)
        status_message = f"API Key {api_key_display}: FAILED. Error: {error_message}"
        logging.error(status_message)
        
        # Specific check for common issues
        if "API key not valid" in error_message or "PERMISSION_DENIED" in error_message:
            logging.error(f"   ㄴ Hint: Key {api_key_display} might be invalid, disabled, or lack permissions.")
        elif "quota" in error_message.lower() or "429" in error_message:
            logging.warning(f"   ㄴ Hint: Key {api_key_display} might have hit a quota or rate limit.")
            
        return False, status_message

if __name__ == "__main__":
    logging.info("Starting API Key Check...")

    if not GOOGLE_API_KEYS:
        logging.error("No API keys found in config.GOOGLE_API_KEYS. Exiting.")
    else:
        logging.info(f"Found {len(GOOGLE_API_KEYS)} API key(s) in config.py to check.")
        functional_keys = 0
        
        for i, key in enumerate(GOOGLE_API_KEYS):
            logging.info(f"--- Checking Key {i+1}/{len(GOOGLE_API_KEYS)} ---")
            is_working, status_msg = check_api_key(key)
            key_status[key] = {"status": "WORKING" if is_working else "FAILED", "message": status_msg}
            if is_working:
                functional_keys += 1
            if i < len(GOOGLE_API_KEYS) - 1:
                time.sleep(1) # Brief pause to avoid hitting any rapid connection limits if any

        logging.info("\n--- API Key Check Summary ---")
        for key, result in key_status.items():
            api_key_display_summary = f"{key[:10]}...{key[-4:]}" if len(key) > 14 else key
            logging.info(f"Key: {api_key_display_summary} - Status: {result['status']}")
            if result['status'] == "FAILED":
                logging.info(f"  └─ Details: {result['message'].split('Error: ', 1)[-1]}") # Show only error part

        logging.info(f"\nTotal Functional Keys: {functional_keys} out of {len(GOOGLE_API_KEYS)}")

        if functional_keys == 0 and GOOGLE_API_KEYS:
            logging.warning("WARNING: No functional API keys found. Your application will likely fail.")
        elif functional_keys < len(GOOGLE_API_KEYS):
            logging.warning("Warning: Some API keys are not fully functional. Check details above.")
        else:
            logging.info("All API keys appear to be functional.")