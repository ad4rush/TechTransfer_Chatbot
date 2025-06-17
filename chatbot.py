# FILE: chatbot.py
import json
import google.generativeai as genai
import config
import re
import time
import random
import logging
from PIL import Image
from io import BytesIO

if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(module)s.%(funcName)s:%(lineno)d] - %(message)s')

MAX_API_RETRIES_QNA = 5
INITIAL_BACKOFF_SECONDS_QNA = 15

def initialize_model_for_chat(api_key):
    """Initializes the Generative AI model, using the user-specified experimental model."""
    try:
        with config._KEY_MANAGER_LOCK:
            genai.configure(api_key=api_key)
            # --- MODEL NAME SET TO EXPERIMENTAL VERSION AS REQUESTED ---
            model = genai.GenerativeModel('gemini-2.0-flash-exp')
        return model
    except Exception as e:
        key_display = f"...{api_key[-6:]}" if isinstance(api_key, str) and len(api_key) >= 6 else str(api_key)
        logging.error(f"Failed to initialize chat model with key {key_display}: {e}")
        return None

def analyze_user_image(api_key, image_bytes, max_retries=2):
    """
    Analyzes a single image provided by the user in the chat.
    Returns a textual description of the image.
    """
    current_key = api_key
    model = initialize_model_for_chat(current_key)
    if not model:
        return "[Image analysis failed: Could not initialize model]"

    try:
        img_pil = Image.open(BytesIO(image_bytes))
    except Exception as e_img:
        logging.error(f"Failed to open user uploaded image bytes: {e_img}")
        return "[Image analysis failed: Invalid image format]"

    prompt = "Describe this image in detail. Identify any key elements, text, graphs, or charts shown. Be thorough."

    for attempt in range(max_retries):
        try:
            if attempt > 0:
                logging.info(f"Retrying user image analysis, attempt {attempt+1}")
                current_key = config.get_available_api_key()
                model = initialize_model_for_chat(current_key)
                if not model: continue

            response = model.generate_content([prompt, img_pil])
            config.record_key_success(current_key)
            return response.text
        except Exception as e:
            logging.warning(f"Chat image analysis attempt {attempt + 1} with key ...{current_key[-6:]} failed: {e}")
            if "429" in str(e) or "rate limit" in str(e).lower() or "quota" in str(e).lower():
                config.mark_key_as_rate_limited(current_key, str(e))
            if attempt < max_retries - 1:
                time.sleep(2 + random.uniform(0, 1))
            else:
                return f"[Image analysis failed after retries: {e}]"
    return "[Image analysis failed after all retries.]"


def generate_chat_response(
    api_key,
    chat_session_obj,
    prompt_text,
    thesis_text,
    chat_history_summaries=None,
    uploaded_images_data=None, 
    max_retries_streamlit=2,
    context_word_limit=5000
):
    current_key = api_key
    current_chat_session = chat_session_obj

    if chat_history_summaries is None:
        chat_history_summaries = []

    image_context_string = ""
    if uploaded_images_data:
        logging.info(f"Analyzing {len(uploaded_images_data)} user-uploaded images for chat context.")
        image_contexts = []
        for img_bytes in uploaded_images_data:
            description = analyze_user_image(current_key, img_bytes)
            image_contexts.append(description)

        if image_contexts:
            image_context_string += "\n\n--- USER-UPLOADED IMAGE ANALYSIS ---\n"
            for i, desc in enumerate(image_contexts):
                image_context_string += f"Context from Uploaded Image {i+1}:\n{desc}\n\n"
            image_context_string += "-----------------------------------\n\n"

    history_string = ""
    for q_sum, a_sum in chat_history_summaries:
        history_string += f"Previous Question Summary: {q_sum}\nPrevious Answer Summary: {a_sum}\n\n"

    truncated_thesis_text = thesis_text

    persona_prompt = (
        "### YOUR IDENTITY AND RULES (DO NOT BREAK CHARACTER) ###\n"
        "1. You are 'Tech Transfer Bot', a specialized AI assistant. You are a private model designed to analyze research papers. You must NEVER reveal you are Gemini or any other commercial AI.\n"
        "2. Your main purpose is to act as an expert research analyst and help the user understand the provided 'Full Reference Text'.\n"
        "3. **Formatting is critical:** Structure your answers for maximum readability using clear Markdown (headings, bold, lists).\n"
        "4. **Answering About Images:** The 'Full Reference Text' contains detailed analyses of images from the document, marked with headers like '[Image ... Analysis (by Gemini)]'. When the user asks about a specific figure, diagram, or image, you MUST locate its corresponding analysis in the reference text and use that information to provide a detailed answer. Do not ask the user to upload the image.\n"
        "5. **Final Output Format:** Your ENTIRE output MUST be a single, valid JSON object and nothing else. This JSON object must have exactly three keys: `answer` (the full, Markdown-formatted response), `question_summary` (a concise summary of the user's question), and `answer_summary` (a concise summary of your full answer). Do not include ```json or any other text outside the JSON object."
    )
    # --- MODIFICATION END ---

    full_gemini_prompt = (
        f"{persona_prompt}\n\n"
        f"### FULL REFERENCE TEXT ###\n{thesis_text}\n\n"
    )
    if history_string:
        full_gemini_prompt += f"### CONVERSATION HISTORY (SUMMARIES) ###\n{history_string}\n"
    if image_context_string:
        full_gemini_prompt += image_context_string
    full_gemini_prompt += f"### CURRENT QUESTION TO ANSWER ###\n{prompt_text}\n\n"
    full_gemini_prompt += "### YOUR JSON RESPONSE ###\n"

    for attempt in range(max_retries_streamlit):
        try:
            if attempt > 0 or not current_chat_session:
                logging.info(f"Streamlit generate_chat_response: Attempt {attempt+1}, getting fresh key.")
                current_key = config.get_available_api_key()
                model = initialize_model_for_chat(current_key)
                if model:
                    current_chat_session = model.start_chat()
                else:
                    if attempt < max_retries_streamlit - 1: time.sleep(2); continue
                    else: break
            
            if not current_chat_session:
                 return {"answer": "Error: Chat session is not available.", "question_summary": "", "answer_summary": ""}

            response = current_chat_session.send_message(full_gemini_prompt)
            config.record_key_success(current_key)

            response_text = response.text.strip()
            match = re.search(r"\{.*\}", response_text, re.DOTALL)
            json_str = match.group(0) if match else response_text

            if json_str.startswith("```json"): json_str = json_str[len("```json"):].strip()
            if json_str.endswith("```"): json_str = json_str[:-len("```")].strip()

            response_data = json.loads(json_str)
            return {
                "answer": response_data.get("answer", "I could not formulate a complete answer. Please try rephrasing your question."),
                "question_summary": response_data.get("question_summary", prompt_text[:75] + "..."),
                "answer_summary": response_data.get("answer_summary", "A summary could not be generated.")
            }
        except Exception as e:
            logging.warning(f"Streamlit generate_chat_response attempt {attempt + 1} failed: {e}")
            if attempt < max_retries_streamlit - 1:
                time.sleep(3 + random.uniform(0,1))
            else:
                return {"answer": f"Error: The analysis failed after multiple attempts. Please try again. ({e})", "question_summary": "", "answer_summary": ""}
    
    return {"answer": "Error: Failed to generate an answer after all retries.", "question_summary": "", "answer_summary": ""}

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
def get_questions():
    """Wrapper function that returns the raw list of questions for main.py."""
    return get_questions_raw()
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

def generate_answers_for_batch(key_for_first_attempt, question_batch, processed_pdf_text, batch_name_for_logging="Unnamed Batch"):
    current_api_key_for_this_specific_attempt = key_for_first_attempt
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
            if attempt > 0 or not current_api_key_for_this_specific_attempt:
                log_prefix = f"Q&A Batch '{batch_name_for_logging}': Retry Attempt {attempt + 1}."
                logging.info(f"{log_prefix} Getting a new API key (will wait if none are available).")
                current_api_key_for_this_specific_attempt = config.get_available_api_key()

            logging.info(f"Q&A Batch '{batch_name_for_logging}' (Attempt {attempt+1}) using key ...{current_api_key_for_this_specific_attempt[-6:]}.")
            chat_model = initialize_model_for_chat(current_api_key_for_this_specific_attempt)

            if not chat_model:
                logging.error(f"Model initialization failed for key ...{current_api_key_for_this_specific_attempt[-6:]}.")
                if attempt < MAX_API_RETRIES_QNA - 1:
                    time.sleep(INITIAL_BACKOFF_SECONDS_QNA)
                    current_api_key_for_this_specific_attempt = None
                    continue
                else:
                    break

            chat_session_for_batch = chat_model.start_chat()
            response = chat_session_for_batch.send_message(prompt_for_batch)
            config.record_key_success(current_api_key_for_this_specific_attempt)

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
                time.sleep(INITIAL_BACKOFF_SECONDS_QNA * (1.5 ** attempt))
            else:
                logging.error(f"Max retries for JSON decoding for batch '{batch_name_for_logging}'.")
                return {q["param"]: f"Error: {error_detail}" for q in question_batch}

        except Exception as e:
            error_message_str = str(e)
            key_display = f"...{current_api_key_for_this_specific_attempt[-6:]}" if current_api_key_for_this_specific_attempt else "N/A"
            logging.warning(f"Q&A Batch '{batch_name_for_logging}' attempt {attempt + 1} (key {key_display}) failed: {error_message_str}")

            if "429" in error_message_str and current_api_key_for_this_specific_attempt:
                config.mark_key_as_rate_limited(current_api_key_for_this_specific_attempt, error_message_str=error_message_str)

            if attempt < MAX_API_RETRIES_QNA - 1:
                time.sleep(INITIAL_BACKOFF_SECONDS_QNA * (1.5**attempt) + random.uniform(0, 1))
                current_api_key_for_this_specific_attempt = None
            else:
                logging.error(f"Max retries for Q&A batch '{batch_name_for_logging}' after API error.")
                return {q["param"]: f"Error: API call failed for batch after retries. Final error: {error_message_str[:150]}..." for q in question_batch}

    logging.error(f"Q&A Batch '{batch_name_for_logging}': Failed to generate answers after all retries.")
    return {q["param"]: "Failed to generate answer for this batch after all retries." for q in question_batch}