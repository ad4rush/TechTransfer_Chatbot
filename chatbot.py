# chatbot.py
import json
import google.generativeai as genai

def initialize_chat(api_key):
    return genai.GenerativeModel('gemini-2.0-flash-exp').start_chat()

def get_questions():
    return [
        {"param": "Title of Thesis", "question": "What is the title of the thesis?"},
        {"param": "Author(s)", "question": "Who are the authors of the thesis?"},
        {"param": "Institution/Department", "question": "Which institution or department is associated with the thesis?"},
        {"param": "Year/Date of Completion", "question": "What is the year or date of completion of the thesis?"},
        {"param": "Supervisor/Advisor(s)", "question": "Who is the supervisor or advisor for the thesis?"},
        {"param": "Field of Study/Discipline", "question": "What is the field of study or discipline (e.g., Electrical Engineering, Computer Science)?"},
        {"param": "Problem Definition", "question": "What specific problem or gap does the thesis address?"},
        {"param": "Research Objectives/Goals", "question": "What are the research objectives or goals of the thesis?"},
        {"param": "Scope & Limitations", "question": "What are the scope and limitations of the research?"},
        {"param": "Technical Approach/Methodology", "question": "What technical approach or methodology was used in the thesis?"},
        {"param": "Data/Materials/Tools Used", "question": "What data, materials, or tools were used in the research?"},
        {"param": "Validation/Testing Methods", "question": "How were the results validated or tested?"},
        {"param": "Key Contributions/Innovations", "question": "What are the key contributions or innovations of the thesis?"},
        {"param": "Technology Readiness Level (TRL)", "question": "What is the stated or estimated Technology Readiness Level (TRL) for this project?"},
        {"param": "R2R or R2C", "question": "Is this thesis focused on Research-to-Research (R2R) or Research-to-Commercialization (R2C)?"},
        {"param": "Prototyping Status", "question": "What is the status of prototyping (working prototype, proof-of-concept, or theoretical)?"},
        {"param": "Technical Feasibility", "question": "What are the main points regarding technical feasibility of the project?"},
        {"param": "Scalability", "question": "Can the solution be scaled up for commercial or industrial use? How?"},
        {"param": "Operational Expertise", "question": "What operational expertise or management considerations are necessary for implementation?"}
    ]

def generate_answer(chat, prompt, thesis_text):
    response = chat.send_message(f"{prompt}\n\nThesis Text:\n{thesis_text[:3000]}...")
    return response.text.strip()