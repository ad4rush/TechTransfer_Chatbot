# chatbot.py
import json
import google.generativeai as genai
import config

def initialize_chat(api_key):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-2.0-flash-exp').start_chat()

def get_questions():
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


# def generate_answer(chat, prompt, thesis_text):
#     try:
#         response = chat.send_message(f"{prompt}\n\nThesis Text:\n{thesis_text[:3000]}...")
#         #response = chat.send_message(prompt)
#         return response.text.strip()
#     except Exception as e:
#         print(f"Initial API key failed. Retrying with another key. Error: {str(e)}")
        
#         # Retry with another key
#         new_key = config.get_random_google_api_key()
#         new_chat = initialize_chat(new_key)
        
#         try:
#             response = new_chat.send_message(f"{prompt}\n\nThesis Text:\n{thesis_text[:3000]}...")
#             return response.text.strip()
#         except Exception as e2:
#             print(f"Retry also failed: {str(e2)}")
#             raise e2  # raise final error

def generate_answer(chat, prompt, thesis_text, max_retries=5):
    """Generate answer using Gemini API with auto-retry on invalid API keys."""
    attempt = 0

    while attempt < max_retries:
        try:
            # Try sending the message
            response = chat.send_message(f"{prompt}\n\nThesis Text:\n{thesis_text[:3000]}...")
            return response.text.strip()
        
        except Exception as e:
            print(f"[Attempt {attempt+1}] API key failed. Retrying with another key... Error: {str(e)}")
            attempt += 1

            # Get a new key and new chat session
            new_key = config.get_random_google_api_key()
            chat = initialize_chat(new_key)

    # If all retries fail
    raise RuntimeError(f"❌ Failed to generate answer after {max_retries} retries.")
