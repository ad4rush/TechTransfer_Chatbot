# FILE: chatbot.py
# CORRECTED: Removed explicit calls to config.record_key_usage()

import json
import google.generativeai as genai
import config # Uses updated config.py where record_key_usage is internal
import re
import time
import random
import logging

if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(module)s.%(funcName)s:%(lineno)d] - %(message)s')

MAX_API_RETRIES_QNA = 3
INITIAL_BACKOFF_SECONDS_QNA = 15 # General backoff for retries

def initialize_model_for_chat(api_key):
    try:
        with config._KEY_MANAGER_LOCK:
            # genai.configure is a global setting, ensure it's managed if multiple models/keys are used closely
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-2.0-flash-exp')
        return model
    except Exception as e:
        logging.error(f"Failed to initialize chat model with key ...{api_key[-6:]}: {e}")
        return None

def get_questions_raw():
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


def generate_answers_for_batch(key_for_first_attempt, question_batch, processed_pdf_text, batch_name_for_logging="Unnamed Batch"):
    current_api_key = key_for_first_attempt
    chat_model = None
    
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
            if attempt > 0: # For retries, always try to get a fresh key
                logging.info(f"Q&A Batch '{batch_name_for_logging}': Retry attempt {attempt + 1}. Attempting to get a new key.")
                current_api_key = config.get_available_api_key() # This will raise ValueError if no key is ready
                logging.info(f"Q&A Batch '{batch_name_for_logging}': Got key ...{current_api_key[-6:]} for retry attempt {attempt + 1}.")

            chat_model = initialize_model_for_chat(current_api_key)
            if not chat_model:
                logging.error(f"Q&A Batch '{batch_name_for_logging}': Model init failed with key ...{current_api_key[-6:]} on attempt {attempt + 1}.")
                if attempt < MAX_API_RETRIES_QNA - 1:
                    time.sleep(INITIAL_BACKOFF_SECONDS_QNA * (1.5**attempt) + random.uniform(0,1)) # General wait
                    continue # To next retry attempt, will try to get a key again
                else: # Max retries for model config
                    break # Break from retries

            logging.info(f"Q&A Batch '{batch_name_for_logging}' attempt {attempt + 1}. Key: ...{current_api_key[-6:]}.")
            chat_session_for_batch = chat_model.start_chat()
            
            # config.record_key_usage(current_api_key) # <--- REMOVED (handled by get_available_api_key)
            response = chat_session_for_batch.send_message(prompt_for_batch)
            config.record_key_success(current_api_key) # Still record success

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

        except ValueError as e_get_key: # Raised by get_available_api_key if no key is ready
            logging.warning(f"Q&A Batch '{batch_name_for_logging}': Attempt {attempt+1} failed to get an API key: {e_get_key}.")
            if attempt < MAX_API_RETRIES_QNA - 1:
                # Wait before retrying to get a key in the next loop iteration
                wait_duration = INITIAL_BACKOFF_SECONDS_QNA * (1.5**attempt) + random.uniform(0,1)
                logging.info(f"Q&A Batch '{batch_name_for_logging}': Waiting {wait_duration:.2f}s before retrying to get a key.")
                time.sleep(wait_duration)
            else: # Max retries for getting a key
                logging.error(f"Q&A Batch '{batch_name_for_logging}': Max retries ({MAX_API_RETRIES_QNA}) reached, failed to get an available key.")
                return {q["param"]: f"Error: Failed to get an API key after max retries for batch." for q in question_batch}

        except json.JSONDecodeError as je:
            error_detail = f"Batch '{batch_name_for_logging}': Failed to decode JSON on attempt {attempt+1}. Error: {je}. Response: {response.text[:500] if 'response' in locals() else 'N/A'}"
            logging.warning(error_detail)
            if attempt < MAX_API_RETRIES_QNA - 1:
                time.sleep(INITIAL_BACKOFF_SECONDS_QNA * (2 ** attempt) + random.uniform(0,1))
            else:
                logging.error(f"Max retries for JSON decoding for batch '{batch_name_for_logging}'.")
                return {q["param"]: f"Error: {error_detail}" for q in question_batch}
        
        except Exception as e: # Other API errors, including 429
            error_message_str = str(e)
            logging.warning(f"Q&A Batch '{batch_name_for_logging}' attempt {attempt + 1} (key ...{current_api_key[-6:]}) failed: {error_message_str}")
            is_rate_limit_error = "429" in error_message_str or "rate limit" in error_message_str.lower() or "quota" in error_message_str.lower()

            if is_rate_limit_error:
                config.mark_key_as_rate_limited(current_api_key, error_message_str=error_message_str)
            
            if attempt < MAX_API_RETRIES_QNA - 1:
                # General sleep before next attempt, which will try to get a fresh key.
                general_retry_sleep = INITIAL_BACKOFF_SECONDS_QNA / 2 * (1.5**attempt) + random.uniform(0,0.5)
                logging.info(f"Q&A Batch '{batch_name_for_logging}': API error. Sleeping {general_retry_sleep:.2f}s before next retry attempt.")
                time.sleep(general_retry_sleep)
            else: # Max retries for this task reached after API error
                logging.error(f"Max retries ({MAX_API_RETRIES_QNA}) for Q&A batch '{batch_name_for_logging}' after API error.")
                return {q["param"]: f"Error: API call failed for batch after retries. Original error: {error_message_str[:100]}..." for q in question_batch}

    logging.error(f"Q&A Batch '{batch_name_for_logging}': Failed to generate answers after all retries (loop exhausted).")
    return {q["param"]: "Failed to generate answer for this batch after all retries." for q in question_batch}


def generate_answer( 
    initial_api_key, 
    chat_session_obj, 
    prompt_text, 
    thesis_text, 
    max_retries_streamlit=2
):
    current_key_for_streamlit = initial_api_key
    current_chat_session = chat_session_obj

    modified_display_prompt = (
        "Provide a direct and long concise answer based on the provided text. Ensure it answers all points. It should be complete and detailed. "
        "Do not include any disclaimers or unnecessary information. Avoid 'I think' or 'In my opinion'. "
        "Do not include any introductory phrases like 'Based on the provided text...'. Just give the answer directly.\n\n"
        f"Instruction/Question: {prompt_text}\n\n"
        f"Full Reference Text:\n{thesis_text}\n"
    )
    
    for attempt in range(max_retries_streamlit):
        try:
            if attempt > 0 or not current_chat_session:
                logging.info(f"Streamlit generate_answer: Attempt {attempt+1}, getting fresh key.")
                current_key_for_streamlit = config.get_available_api_key()
                model = initialize_model_for_chat(current_key_for_streamlit)
                if model:
                    current_chat_session = model.start_chat()
                    logging.info(f"Streamlit generate_answer: Re-initialized chat with key ...{current_key_for_streamlit[-6:]} for attempt {attempt+1}.")
                else:
                    if attempt < max_retries_streamlit -1:
                        time.sleep(INITIAL_BACKOFF_SECONDS_QNA / 2 * (1.5**attempt))
                        continue
                    else:
                        return "Error: Failed to re-initialize chat session model for Streamlit after key refresh."
            
            if not current_chat_session:
                 return "Error: Chat session is not available for Streamlit."

            logging.info(f"Streamlit generate_answer attempt {attempt+1} with key ...{current_key_for_streamlit[-6:]}")
            # REMOVED: config.record_key_usage(current_key_for_streamlit) - Now handled by get_available_api_key
            response = current_chat_session.send_message(modified_display_prompt)
            config.record_key_success(current_key_for_streamlit)
            return response.text.strip()

        except ValueError as e_get_key:
            logging.warning(f"Streamlit generate_answer: Attempt {attempt+1} failed to get an API key: {e_get_key}.")
            if attempt < max_retries_streamlit - 1:
                wait_duration = INITIAL_BACKOFF_SECONDS_QNA / 2 * (1.5**attempt) + random.uniform(0,0.5)
                logging.info(f"Streamlit generate_answer: Waiting {wait_duration:.2f}s before retrying to get a key.")
                time.sleep(wait_duration)
            else:
                return f"Error: No API keys available for Streamlit Q&A after retries. ({e_get_key})"
        
        except Exception as e:
            error_message_str = str(e)
            logging.warning(f"Streamlit generate_answer attempt {attempt + 1} (key ...{current_key_for_streamlit[-6:]}) failed: {error_message_str}")
            is_rate_limit_error = "429" in error_message_str or "rate limit" in error_message_str.lower() or "quota" in error_message_str.lower()
            if is_rate_limit_error:
                config.mark_key_as_rate_limited(current_key_for_streamlit, error_message_str=error_message_str)
            
            if attempt < max_retries_streamlit - 1:
                general_retry_sleep = INITIAL_BACKOFF_SECONDS_QNA / 2 * (1.5**attempt) + random.uniform(0,0.5)
                logging.info(f"Streamlit Q&A: API error. Sleeping {general_retry_sleep:.2f}s before next retry attempt.")
                time.sleep(general_retry_sleep)
            else:
                return f"Error: API call failed for Streamlit. (Key ...{current_key_for_streamlit[-6:]} failed: {error_message_str[:100]}...)"
                
    return "Error: Failed to generate answer after multiple retries for Streamlit."

