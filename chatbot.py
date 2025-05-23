# chatbot.py
import json
import google.generativeai as genai
import config
import re
import time
import random
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s')

MAX_API_RETRIES = 3
INITIAL_BACKOFF_SECONDS = 5 # Increased for Q&A calls which might be larger

def initialize_chat_model(api_key): # Renamed for clarity, returns model
    """Configures and returns a Gemini Model instance for chat-like interactions."""
    # Similar to pdf_utils, ensuring the key is configured for this model instance.
    # This will be called per batch with a potentially different key.
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        return model
    except Exception as e:
        logging.error(f"Failed to initialize chat model with key {api_key[:10]}...: {e}")
        return None

def get_questions_raw(): # Renamed to avoid conflict if we have a batch getter
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
    """Groups questions into logical batches for processing."""
    questions = get_questions_raw()
    batches = {
        "Batch 1: Basic Information": questions[0:6],  # Title to Field of Study
        "Batch 2: Project Core & Context": questions[6:10], # Target Beneficiaries to Review of Existing Solutions
        "Batch 3: Methodology & Validation": questions[10:13],# Research Methodology to Validation of Results
        "Batch 4: Innovation & Viability": questions[13:17],# Feasibility to Market/Infra Readiness
        "Batch 5: Readiness & Development Stage": questions[17:22],# TRL to Cost Structure
        "Batch 6: Market Application & Impact": questions[22:27] # Scalability to Certifications
    }
    # Verify all questions are covered
    all_batched_params = [q['param'] for batch in batches.values() for q in batch]
    all_raw_params = [q['param'] for q in questions]
    if set(all_batched_params) != set(all_raw_params) or len(all_batched_params) != len(all_raw_params) :
        logging.error("Mismatch in batched questions and raw questions! Check define_question_batches().")
        # Fallback to a single batch if logic is flawed
        return {"Batch 1: All Questions": questions}
    return batches

def get_questions(): # This is the one main.py and batch_analyzer will use
    """Returns a flat list of all questions, compatible with existing logic."""
    return get_questions_raw()


def generate_answers_for_batch(api_key_for_batch, question_batch, processed_pdf_text, batch_name_for_logging="Unnamed Batch"):
    """
    Generates high-quality, long answers for a specific batch of questions.
    """
    chat_model = initialize_chat_model(api_key_for_batch)
    if not chat_model:
        logging.error(f"Failed to initialize chat model for batch {batch_name_for_logging}. API key: {api_key_for_batch[:10]}...")
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

    for attempt in range(MAX_API_RETRIES):
        try:
            logging.info(f"Attempt {attempt + 1} for batch '{batch_name_for_logging}'. Sending {len(prompt_for_batch)} chars. Key: {api_key_for_batch[:10]}...")
            # Using .start_chat() and .send_message() as per existing pattern,
            # though for a single turn, generate_content on the model directly might also work.
            # Sticking to chat_session pattern.
            chat_session_for_batch = chat_model.start_chat() # Fresh chat for each batch attempt if retrying on model level
            response = chat_session_for_batch.send_message(prompt_for_batch)
            
            response_text = response.text.strip()
            match = re.search(r"\{.*\}", response_text, re.DOTALL) # Try to find JSON block
            json_str = match.group(0) if match else response_text # Fallback to full text if no clear block
            
            # Further cleaning if markdown still present
            if json_str.startswith("```json"): json_str = json_str[len("```json"):]
            if json_str.startswith("```"): json_str = json_str[len("```"):]
            if json_str.endswith("```"): json_str = json_str[:-len("```")]
            json_str = json_str.strip()

            batch_answers_dict = json.loads(json_str)
            
            # Ensure all params from this batch are in the dict, even if value is from .get() default
            final_batch_answers = {}
            for q_data in question_batch:
                param = q_data["param"]
                final_batch_answers[param] = batch_answers_dict.get(param, f"Information for '{param}' not explicitly provided by AI for this batch.")
            
            logging.info(f"Successfully generated and parsed answers for batch '{batch_name_for_logging}'.")
            return final_batch_answers

        except json.JSONDecodeError as je:
            error_detail = f"Batch '{batch_name_for_logging}': Failed to decode JSON. Error: {je}. Response (first 500): {response.text[:500] if 'response' in locals() else 'N/A'}"
            logging.warning(f"[Attempt {attempt+1}] {error_detail}")
            if attempt < MAX_API_RETRIES - 1:
                time.sleep(INITIAL_BACKOFF_SECONDS * (2 ** attempt) + random.uniform(0,1)) # Exponential backoff
            else:
                logging.error(f"Max retries reached for JSON decoding for batch '{batch_name_for_logging}'.")
                return {q["param"]: f"Error: {error_detail}" for q in question_batch}
        
        except Exception as e: # Includes API errors like 429
            error_detail = f"Batch '{batch_name_for_logging}': API call or other error: {e}"
            logging.warning(f"[Attempt {attempt+1}] {error_detail}")
            if "429" in str(e) or "rate limit" in str(e).lower() or "quota" in str(e).lower():
                if attempt < MAX_API_RETRIES - 1:
                    backoff_time = INITIAL_BACKOFF_SECONDS * (2 ** attempt) + random.uniform(0,1)
                    logging.info(f"Rate limit/quota error for batch '{batch_name_for_logging}'. Retrying in {backoff_time:.2f}s...")
                    time.sleep(backoff_time)
                    # Key rotation for the next attempt of this batch is handled by the caller (batch_paper_analyzer)
                    # if it decides to retry the whole batch task with a new key.
                    # This function will just retry with its current `api_key_for_batch`.
                    # If this key is exhausted, the function will fail after retries, and the caller can try with another key.
                else:
                    logging.error(f"Max retries reached for batch '{batch_name_for_logging}' due to API errors (429/quota).")
                    return {q["param"]: f"Error: API rate limit/quota exceeded after retries for batch. Original error: {e}" for q in question_batch}
            else: # Non-rate-limit API error, fail faster
                logging.error(f"Non-retryable API error for batch '{batch_name_for_logging}': {e}", exc_info=True)
                return {q["param"]: f"Error: API call failed for batch. Original error: {e}" for q in question_batch}

    logging.error(f"Batch '{batch_name_for_logging}': Failed to generate answers after all retries.")
    return {q["param"]: "Failed to generate answer for this batch after all retries." for q in question_batch}


# generate_answer for Streamlit app (single question, kept for compatibility)
def generate_answer(chat_session, prompt, thesis_text, max_retries=3):
    attempt = 0
    modified_prompt = (
        "Provide a direct and long concise answer to the following question/instruction based on the provided text. Ensure it answers all points. It should be complete and detailed. "
        "Do not include any disclaimers or unnecessary information. "
        "Avoid using phrases like 'I think' or 'In my opinion'. "
        "Do not include any introductory phrases like 'Based on the provided text...' or similar conversational fluff. "
        "Just give the answer directly.\n\n"
        f"Instruction/Question: {prompt}\n\n"
        f"Full Reference Text:\n{thesis_text}\n" # Send full text
    )
    current_chat_session = chat_session # Use the passed one
    
    for attempt in range(max_retries):
        try:
            # logging.info(f"Streamlit generate_answer attempt {attempt+1}")
            response = current_chat_session.send_message(modified_prompt)
            return response.text.strip()
        except Exception as e:
            logging.warning(f"Streamlit generate_answer attempt {attempt + 1} failed: {e}")
            if "429" in str(e) or "rate limit" in str(e).lower() or "quota" in str(e).lower():
                if attempt < max_retries - 1:
                    backoff_time = INITIAL_BACKOFF_SECONDS * (2 ** attempt) + random.uniform(0,1)
                    logging.info(f"Streamlit Q&A: Rate limit. Retrying in {backoff_time:.2f}s...")
                    time.sleep(backoff_time)
                    # Re-initialize chat with a new key for the next attempt on rate limit
                    try:
                        new_key = config.get_random_google_api_key()
                        current_chat_session = initialize_chat_model(new_key).start_chat() # Use new model
                        if not current_chat_session: raise Exception("Failed to re-initialize chat for retry")
                    except Exception as e_reinit:
                        logging.error(f"Failed to re-initialize chat with new key during retry: {e_reinit}")
                        # Could continue with old session or just let it fail out
                else:
                    logging.error(f"Streamlit Q&A: Max retries for rate limit.")
                    return f"Error: API rate limit / quota exceeded after multiple retries. ({e})"
            else: # Non-rate-limit, non-key related API error
                 logging.error(f"Streamlit Q&A: Non-retryable API error: {e}")
                 return f"Error: API call failed. ({e})"

    logging.error(f"Streamlit Q&A: Failed after all retries.")
    return "Error: Failed to generate answer after multiple retries."