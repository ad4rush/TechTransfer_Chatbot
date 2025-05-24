# chatbot.py
import json
import google.generativeai as genai
import config
import re
import time
import random
import logging

if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(module)s.%(funcName)s:%(lineno)d] - %(message)s')

MAX_API_RETRIES_QNA = 3 # Max retries for a Q&A batch task (includes key rotations)
INITIAL_BACKOFF_SECONDS_QNA = 7 

def initialize_model_for_chat(api_key): 
    try:
        with config._KEY_MANAGER_LOCK:
            # REMOVED: if config.track_api_call(api_key):
            # REMOVED:     raise Exception("Rate limit exceeded for this key")
            # The API key is already vetted by get_available_api_key before this function is called.
            # Actual API calls (send_message) are tracked by record_key_success.
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-2.0-flash-exp')
        return model
    except Exception as e:
        logging.error(f"Failed to initialize chat model with key ...{api_key[-6:]}: {e}")
        return None

def get_questions_raw():
    # ... (get_questions_raw function remains the same) ...
    return [
        {"param": "Title", "question": "What is the title of the project or thesis?"},
        {"param": "Authors", "question": "Who are the authors of the project or thesis?"},
        {"param": "Institute and Department", "question": "Which institute and department is this associated with?"},
        {"param": "Date", "question": "What is the date or year of completion?"},
        {"param": "Advisor", "question": "Who is the advisor or supervisor of the project?"},
        {"param": "Field of Study", "question": "What is the field of study (e.g., CS, BioTech, etc.)?"},
        {"param": "Target Beneficiaries", "question": "Who are the intended users or customer groups for the proposed solution?"},
        {"param": "Project Goal", "question": "What is the main objective or problem the project aims to solve?"},
        {"param": "Scope and Limitations", "question": "What are the research boundaries, assumptions, and potential pitfalls?"},
        {"param": "Review of Existing Solutions", "question": "What are the current solutions and how does your project improve or differ from them?"},
        {"param": "Research Methodology", "question": "How was the research conducted (e.g., simulations, field tests)?"},
        {"param": "Data and Tools Used", "question": "What data, materials, and tools were used in the research?"},
        {"param": "Validation of Results", "question": "How were the results validated or tested?"},
        {"param": "Feasibility and Scalability", "question": "Is the solution technically and commercially scalable?"},
        {"param": "Uniqueness and Competitive Analysis", "question": "What makes the project unique, and who are the competitors or alternatives?"},
        {"param": "IP Potential", "question": "Does the project have any novel contributions with potential for intellectual property (patents, trade secrets, etc.)?"},
        {"param": "Market and Infrastructure Readiness", "question": "Is the market or infrastructure ready to adopt this technology?"},
        {"param": "Technology Readiness Level", "question": "What is the current Technology Readiness Level (TRL) from 1 to 9?"},
        {"param": "Commercial Viability", "question": "Is the product market-ready and financially viable?"},
        {"param": "Cost and Complexity", "question": "What are the development costs and technical complexities involved?"},
        {"param": "Prototype Stage", "question": "What is the current development status (proof-of-concept, prototype, theoretical)?"},
        {"param": "Cost Structure", "question": "How do costs (material, resource scarcity, production) affect pricing?"},
        {"param": "Scalability to Industry", "question": "Can the solution be scaled up for commercial or industrial use effectively?"},
        {"param": "Target Industry", "question": "What industries or sectors would benefit most from this solution?"},
        {"param": "Real-World Applications", "question": "What are the practical use cases where the solution can be implemented?"},
        {"param": "Market Size and Growth", "question": "What is the market size and growth potential if commercialized?"},
        {"param": "Certifications", "question": "Have any certifications been obtained or are any required for implementation?"}
    ]

def define_question_batches():
    # ... (define_question_batches function remains the same) ...
    questions = get_questions_raw()
    batches = {
        "Batch 1: Basic Information": questions[0:6],
        "Batch 2: Project Core & Context": questions[6:10],
        "Batch 3: Methodology & Validation": questions[10:13],
        "Batch 4: Innovation & Viability": questions[13:17],
        "Batch 5: Readiness & Development Stage": questions[17:22],
        "Batch 6: Market Application & Impact": questions[22:27]
    }
    all_batched_params = [q['param'] for batch in batches.values() for q in batch]
    all_raw_params = [q['param'] for q in questions]
    if set(all_batched_params) != set(all_raw_params) or len(all_batched_params) != len(all_raw_params) :
        logging.critical("CRITICAL ERROR: Mismatch in batched questions and raw questions! Check define_question_batches().")
        return {"CRITICAL_ERROR_IN_BATCHING": [{"param": "Error", "question": "Batch definition error"}]}
    return batches

def get_questions(): 
    return get_questions_raw()

def generate_answers_for_batch(initial_api_key, question_batch, processed_pdf_text, batch_name_for_logging="Unnamed Batch"):
    current_api_key_for_batch = initial_api_key
    chat_model = initialize_model_for_chat(current_api_key_for_batch) 
    keys_tried_this_batch_task = {initial_api_key}

    if not chat_model:
        logging.error(f"Batch '{batch_name_for_logging}': Failed to initialize chat model with key ...{initial_api_key[-6:]}.")
        return {q_data["param"]: "Error: Failed to initialize AI model for this batch." for q_data in question_batch}

    question_details = [f"  - Parameter: \"{q['param']}\"\n    Question: \"{q['question']}\"" for q in question_batch]
    current_batch_questions_string = "\n".join(question_details)
    prompt_for_batch = (
        "You are an expert research analyst. Your task is to meticulously analyze the provided 'Processed Research Paper Text' "
        "and provide **long, detailed, highly descriptive, accurate, and comprehensive answers** to the questions in this batch. "
        "These answers will serve as a high-quality dataset for fine-tuning. Precision, completeness, extensive detail, and direct relevance are paramount. "
        "Elaborate as much as possible based *solely* on the information in the 'Processed Research Paper Text'. Do not use external knowledge.\n\n"
        "Instructions for your response:\n"
        "1. Provide a direct, thorough, and **elaborate answer for each question in this batch, making it as long and detailed as the text supports**.\n"
        "2. Ensure each answer addresses all aspects of its corresponding question with significant detail.\n"
        "3. Avoid introductory phrases, conversational fluff, or disclaimers.\n"
        "4. Your output *must* be a single, valid JSON object. This JSON object should map each question's 'Parameter' name from this batch to its **long and detailed string answer**.\n\n"
        f"Questions to Answer in this Batch (use these 'Parameter' names as keys in your JSON output for this batch):\n{current_batch_questions_string}\n\n"
        f"Processed Research Paper Text:\n{processed_pdf_text}\n\n" 
        "JSON Response for this batch (ensure this is only the JSON object and nothing else):"
    )

    for attempt in range(MAX_API_RETRIES_QNA):
        try:
            logging.info(f"Q&A Batch '{batch_name_for_logging}' attempt {attempt + 1}. Key: ...{current_api_key_for_batch[-6:]}.")
            # Each attempt with a (potentially new) key should use a fresh chat session from the model
            chat_session_for_batch = chat_model.start_chat() 
            response = chat_session_for_batch.send_message(prompt_for_batch)
            config.record_key_success(current_api_key_for_batch) 
            
            response_text = response.text.strip()
            match = re.search(r"\{.*\}", response_text, re.DOTALL)
            json_str = match.group(0) if match else response_text
            if json_str.startswith("```json"): json_str = json_str[len("```json"):]
            if json_str.startswith("```"): json_str = json_str[len("```"):]
            if json_str.endswith("```"): json_str = json_str[:-len("```")]
            json_str = json_str.strip()

            batch_answers_dict = json.loads(json_str)
            final_batch_answers = {q["param"]: batch_answers_dict.get(q["param"], f"Info for '{q['param']}' not in AI JSON for batch.") for q in question_batch}
            logging.info(f"Successfully generated and parsed answers for batch '{batch_name_for_logging}'.")
            return final_batch_answers

        except json.JSONDecodeError as je:
            error_detail = f"Batch '{batch_name_for_logging}': Failed to decode JSON on attempt {attempt+1}. Error: {je}. Response: {response.text[:500] if 'response' in locals() else 'N/A'}"
            logging.warning(error_detail)
            if attempt < MAX_API_RETRIES_QNA - 1:
                time.sleep(INITIAL_BACKOFF_SECONDS_QNA * (2 ** attempt) + random.uniform(0,1)) 
            else: 
                logging.error(f"Max retries for JSON decoding for batch '{batch_name_for_logging}'.")
                return {q["param"]: f"Error: {error_detail}" for q in question_batch}
        
        except Exception as e: 
            logging.warning(f"Q&A Batch '{batch_name_for_logging}' attempt {attempt + 1} (key ...{current_api_key_for_batch[-6:]}) failed: {e}")
            is_rate_limit_error = "429" in str(e) or "rate limit" in str(e).lower() or "quota" in str(e).lower()
            
            if is_rate_limit_error:
                config.mark_key_as_rate_limited(current_api_key_for_batch)
                if attempt < MAX_API_RETRIES_QNA - 1:
                    try:
                        logging.info(f"Q&A Batch '{batch_name_for_logging}': Rate limit. Attempting to switch key.")
                        new_key_to_try = config.get_available_api_key(exclude_keys=list(keys_tried_this_batch_task))
                        keys_tried_this_batch_task.add(new_key_to_try)
                        current_api_key_for_batch = new_key_to_try
                        chat_model = initialize_model_for_chat(current_api_key_for_batch) # Re-init model with new key
                        if not chat_model:
                            logging.error(f"Q&A Batch '{batch_name_for_logging}': Failed to get new model with new key ...{current_api_key_for_batch[-6:]}. Ending retries.")
                            break 
                        logging.info(f"Q&A Batch '{batch_name_for_logging}': Switched to new key ...{current_api_key_for_batch[-6:]} for next attempt.")
                    except ValueError as e_get_key: # Raised by get_available_api_key if all keys are resting & global cooldown passed
                        logging.error(f"Q&A Batch '{batch_name_for_logging}': Could not get a new API key for retry ({e_get_key}). All keys might be exhausted. Ending retries.")
                        break  # Break from retry loop, will fall through to error return for batch
                    
                    backoff_time = INITIAL_BACKOFF_SECONDS_QNA * (2**attempt) + random.uniform(0,1)
                    logging.info(f"Q&A Batch '{batch_name_for_logging}': Retrying with new key ...{current_api_key_for_batch[-6:]} in {backoff_time:.2f}s...")
                    time.sleep(backoff_time)
                    # Continue to next attempt in the loop with the new key
                else: # Max retries for this task after a 429 on the last attempt
                    logging.error(f"Max retries ({MAX_API_RETRIES_QNA}) for Q&A batch '{batch_name_for_logging}' due to persistent rate limits.")
                    return {q["param"]: f"Error: API rate limit/quota after retries for batch. Original error: {e}" for q in question_batch}
            else: # Non-rate-limit API error
                logging.error(f"Non-retryable API error for Q&A batch '{batch_name_for_logging}' on attempt {attempt+1}: {e}", exc_info=True) # Log full traceback for unexpected errors
                if attempt == MAX_API_RETRIES_QNA - 1: # If this was the last retry
                    return {q["param"]: f"Error: API call failed for batch after retries. Original error: {e}" for q in question_batch}
                # Backoff for other potentially transient errors before final failure
                time.sleep(INITIAL_BACKOFF_SECONDS_QNA * (2**attempt) + random.uniform(0,1))

    logging.error(f"Q&A Batch '{batch_name_for_logging}': Failed to generate answers after all retries (loop exhausted).")
    return {q["param"]: "Failed to generate answer for this batch after all retries." for q in question_batch}


def generate_answer( # For Streamlit app
    initial_api_key, 
    chat_session_obj, # This is genai.ChatSession object
    prompt_text, 
    thesis_text, 
    max_retries_streamlit=2
):
    current_chat_session = chat_session_obj
    current_key = initial_api_key # The key originally used to create chat_session_obj
    keys_tried_this_streamlit_call = {initial_api_key}

    if not current_chat_session: # Should ideally not happen if main.py manages session state
        logging.warning("Streamlit generate_answer: Chat session object is None. Attempting to reinitialize.")
        try:
            current_key = config.get_available_api_key() 
            keys_tried_this_streamlit_call.add(current_key)
            model = initialize_model_for_chat(current_key)
            if model:
                current_chat_session = model.start_chat()
                # The calling main.py needs to update st.session_state.chat_session_object and current_chat_api_key
                logging.info(f"Streamlit generate_answer: Re-initialized chat with key ...{current_key[-6:]}. Caller must update session state.")
                # It's tricky to update Streamlit's session state from here directly.
                # This function will use the new session for its retries.
            else:
                return "Error: Failed to re-initialize chat session model for Streamlit."
        except ValueError as e_get_key:
             return f"Error: No API keys available to re-initialize chat: {e_get_key}"
        except Exception as e_reinit:
            return f"Error: Could not re-initialize chat session: {e_reinit}"

    modified_display_prompt = (
        "Provide a direct and long concise answer based on the provided text. Ensure it answers all points. It should be complete and detailed. "
        "Do not include any disclaimers or unnecessary information. Avoid 'I think' or 'In my opinion'. "
        "Do not include any introductory phrases like 'Based on the provided text...'. Just give the answer directly.\n\n"
        f"Instruction/Question: {prompt_text}\n\n"
        f"Full Reference Text:\n{thesis_text}\n"
    )
    
    for attempt in range(max_retries_streamlit):
        try:
            logging.info(f"Streamlit generate_answer attempt {attempt+1} with key ...{current_key[-6:]}")
            response = current_chat_session.send_message(modified_display_prompt)
            config.record_key_success(current_key)
            return response.text.strip()
        except Exception as e:
            logging.warning(f"Streamlit generate_answer attempt {attempt + 1} (key ...{current_key[-6:]}) failed: {e}")
            is_rate_limit_error = "429" in str(e) or "rate limit" in str(e).lower() or "quota" in str(e).lower()
            if is_rate_limit_error:
                config.mark_key_as_rate_limited(current_key)
                if attempt < max_retries_streamlit - 1:
                    try:
                        logging.info(f"Streamlit Q&A: Rate limit with key ...{current_key[-6:]}. Attempting to switch key.")
                        new_key_to_try = config.get_available_api_key(exclude_keys=list(keys_tried_this_streamlit_call))
                        keys_tried_this_streamlit_call.add(new_key_to_try)
                        current_key = new_key_to_try # Update current key for this task
                        
                        new_model = initialize_model_for_chat(current_key)
                        if not new_model:
                            logging.error("Streamlit Q&A: Failed to init model with new key. Ending retries for this call.")
                            break # Abort retries for this call
                        current_chat_session = new_model.start_chat() # Use a new chat session
                        logging.info(f"Streamlit Q&A: Switched to new key ...{current_key[-6:]} (new chat session created for this attempt).")
                        # The calling function (main.py) should ideally get this new session and key to update st.session_state
                    except ValueError as e_get_key:
                        logging.error(f"Streamlit Q&A: Could not get a new API key for retry ({e_get_key}). Ending retries for this call.")
                        break 
                    
                    backoff_time = INITIAL_BACKOFF_SECONDS_QNA * (0.5 * (2 ** attempt)) + random.uniform(0,0.5) 
                    logging.info(f"Streamlit Q&A: Retrying in {backoff_time:.2f}s...")
                    time.sleep(backoff_time)
                else: # Max retries for this call
                    logging.error(f"Streamlit Q&A: Max retries for rate limit with key ...{current_key[-6:]}.")
                    return f"Error: API rate limit/quota exceeded. Please try again shortly. (Key ...{current_key[-6:]} failed: {e})"
            else: # Non-rate-limit API error
                 logging.error(f"Streamlit Q&A: Non-retryable API error with key ...{current_key[-6:]}: {e}")
                 return f"Error: API call failed. (Key ...{current_key[-6:]}: {e})"
    return "Error: Failed to generate answer after multiple retries for Streamlit."