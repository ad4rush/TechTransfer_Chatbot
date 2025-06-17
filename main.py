import streamlit as st
import random
import os
import config
from chatbot import initialize_model_for_chat, get_questions, define_question_batches, generate_answers_for_batch, generate_chat_response
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from playwright.async_api import async_playwright
from pdf_utils import process_pdf_with_progress
import re
import platform
import logging
import json

# Configure logging
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(module)s.%(funcName)s:%(lineno)d] - %(message)s')

# --- Configuration Constants ---
OUTPUT_TEXT_FILE = "streamlit_extracted_text.txt"
IMAGES_FOLDER = "streamlit_extracted_images"
MAX_STREAMLIT_QA_WORKERS_CAP = 8

# --- Streamlit Session State Initialization ---
if 'processed' not in st.session_state:
    st.session_state.update({
        'processed': False, 'chat_session_object': None, 'current_chat_api_key': None,
        'questions': get_questions(), 'answers': {}, 'feedback_history': {}, 'report_ready': False,
        'pdf_text': None, 'initial_answers_generated': False, 'temp_pdf_path_for_cleanup': None,
        'chat_messages': [], 'chat_context_summaries': [],
        'extracted_images_data': [], 'uploaded_file_name': None
    })

# --- Helper Functions ---
def init_streamlit_chat_session():
    try:
        api_key = config.get_available_api_key()
        st.session_state.current_chat_api_key = api_key
        model = initialize_model_for_chat(api_key)
        if model:
            st.session_state.chat_session_object = model.start_chat()
            logging.info(f"Streamlit chat session initialized.")
            return True
        st.error("Failed to initialize chat model.")
        return False
    except Exception as e:
        st.error(f"Error initializing chat: {e}")
        return False

def generate_all_initial_answers():
    if not st.session_state.get('pdf_text') or st.session_state.get('initial_answers_generated'):
        return

    logging.info("Starting concurrent generation of initial answers.")
    st.session_state.answers = {}
    all_question_batches = define_question_batches()
    num_available_keys = len(config.GOOGLE_API_KEYS) if config.GOOGLE_API_KEYS else 0
    actual_workers = max(1, min(MAX_STREAMLIT_QA_WORKERS_CAP, len(all_question_batches), num_available_keys))

    with st.spinner(f"Generating initial analysis using up to {actual_workers} parallel processors..."):
        total_batches = len(all_question_batches)
        progress_bar_overall_placeholder = st.empty()
        status_text_area_placeholder = st.empty()
        progress_bar_overall_placeholder.progress(0.0, text="Initializing concurrent Q&A batch processing...")

        with ThreadPoolExecutor(max_workers=actual_workers) as executor:
            futures = {
                executor.submit(generate_answers_for_batch, config.get_available_api_key(), questions_in_batch, st.session_state.pdf_text, batch_name): (batch_name, questions_in_batch)
                for batch_name, questions_in_batch in all_question_batches.items()
            }
            for i, future in enumerate(as_completed(futures)):
                batch_name_completed, original_questions = futures[future]
                progress_percent = (i + 1) / total_batches
                try:
                    batch_answers_dict = future.result()
                    for q_data in original_questions:
                        param = q_data["param"]
                        answer_text = batch_answers_dict.get(param, f"Answer not found for {param}.")
                        st.session_state.answers[param] = {'answer': answer_text, 'versions': [answer_text]}
                except Exception as e:
                    logging.error(f"Error processing batch '{batch_name_completed}': {e}", exc_info=True)
                
                status_text_area_placeholder.info(f"Batch \"{batch_name_completed}\" processed. ({i + 1}/{total_batches} complete).")
                progress_bar_overall_placeholder.progress(progress_percent)

    status_text_area_placeholder.empty()
    progress_bar_overall_placeholder.empty()
    st.session_state.initial_answers_generated = True
    logging.info("Finished concurrent generation of initial answers.")
    st.success("Initial analysis complete!")


async def html_to_pdf(html_content, output_path):
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.set_content(html_content)
        await page.pdf(path=output_path)
        await browser.close()

def all_questions_answered():
    if not st.session_state.get('questions'): return False
    return all(
        (q_data["param"] in st.session_state.answers and
         st.session_state.answers[q_data["param"]].get('answer','').strip())
        for q_data in st.session_state.questions
    )

def load_extracted_images_data(pdf_file_name):
    pdf_name_without_ext = os.path.splitext(pdf_file_name)[0]
    images_pdf_specific_dir = os.path.join(IMAGES_FOLDER, pdf_name_without_ext)
    context_dir = os.path.join(images_pdf_specific_dir, "Extracted_images_context")
    image_data_list = []
    
    if not os.path.exists(images_pdf_specific_dir): return []

    image_files = sorted([f for f in os.listdir(images_pdf_specific_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    for img_file in image_files:
        analysis_file_path = os.path.join(context_dir, f"{os.path.splitext(img_file)[0]}_analysis.txt")
        analysis_content = ""
        if os.path.exists(analysis_file_path):
            with open(analysis_file_path, 'r', encoding='utf-8') as f:
                analysis_content = f.read()
        
        if "was not sent for vision analysis" not in analysis_content:
            image_data_list.append({
                "image_path": os.path.join(images_pdf_specific_dir, img_file),
                "analysis": analysis_content or "Analysis pending.", "filename": img_file
            })
    return image_data_list

# --- Page Rendering Functions ---
def render_main_analysis_page():
    st.header("Your Research Analysis!")

    if not st.session_state.initial_answers_generated:
        st.info("Initial answers are being generated. Please wait.")
        return

    all_question_batches = define_question_batches()
    for batch_name, questions_in_batch in all_question_batches.items():
        st.markdown(f"## {batch_name}")
        for q_data in questions_in_batch:
            param, question = q_data["param"], q_data["question"]
            if param not in st.session_state.answers:
                st.session_state.answers[param] = {'answer': 'Processing...', 'versions': []}

            st.markdown(f"### {param}")
            st.markdown(f"**Question:** *{question}*")
            edited_answer = st.text_area(f"Answer for {param}", value=st.session_state.answers[param].get('answer', ''), height=180, key=f"editor_{param}")
            
            if st.button(f"Save Changes", key=f"save_{param}"):
                st.session_state.answers[param]['answer'] = edited_answer
                st.session_state.answers[param].setdefault('versions', []).append(f"Manual edit: {edited_answer}")
                st.success(f"Saved answer for {param}"); st.rerun()

    if all_questions_answered() and not st.session_state.get('report_ready'):
        st.success("All questions have initial answers!")
        if st.button("Generate Final Report", use_container_width=True):
            st.session_state.report_ready = True
            st.rerun()

    if st.session_state.get('report_ready'):
        st.subheader("Download Your Final Report")
        uploaded_filename = st.session_state.get('uploaded_file_name', "Uploaded_File.pdf")
        uploaded_base = os.path.splitext(uploaded_filename)[0]

        toc_html, sections_html = "", ""
        def sanitize_filename(name): return "".join(c if c.isalnum() else "_" for c in name)[:50]

        for idx, q_data in enumerate(st.session_state.questions):
            param, question_text = q_data["param"], q_data["question"]
            anchor_id = f"q{idx+1}_{sanitize_filename(param)}"
            toc_html += f'<a href="#{anchor_id}">Q{idx+1}: {param}</a>'
            answer_info = st.session_state.answers.get(param, {})
            raw_answer = answer_info.get('answer', 'Not yet answered or answer unavailable.')

            if not isinstance(raw_answer, str): raw_answer = str(raw_answer)
            answer_html = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', raw_answer.replace('\n', '<br>'))
            sections_html += f"""<div class="section" id="{anchor_id}"><div class="box"><div class="param">{param}</div><div class="question">Q{idx+1}: {question_text}</div><div class="answer">{answer_html}</div></div></div>"""

        html_content = f"""<!DOCTYPE html><html><head><meta charset="UTF-8"><title>{uploaded_base} - Tech Transfer Report</title><style>body{{font-family:Arial,sans-serif;margin:40px;line-height:1.6;}}.header{{text-align:center;border-bottom:2px solid #333;margin-bottom:30px;}}.toc{{margin:20px 0;padding:10px;border:1px solid #eee;background-color:#f9f9f9;border-radius:5px;}}.toc h2{{margin-top:0;}}.toc a{{display:block;margin:5px 0;color:#2980b9;text-decoration:none;}}.toc a:hover{{text-decoration:underline;}}.section{{margin-bottom:30px;page-break-inside:avoid;}}.box{{border:1px solid #ccc;padding:15px 20px;border-radius:8px;background-color:#fdfdfd;box-shadow:2px 2px 5px rgba(0,0,0,0.05);}}.param{{font-weight:bold;color:#2c3e50;font-size:1.15em;margin-bottom:5px;}}.question{{color:#555;font-size:1em;margin-top:0px;font-style:italic;}}.answer{{margin:10px 0 10px 15px;color:#34495e;}}.footer{{text-align:center;margin-top:40px;color:#777;font-size:0.9em;}}</style></head><body><div class="header"><h1>{uploaded_base}</h1><p><em>Analyzed and compiled by Tech Transfer ChatBot</em></p></div><div class="toc"><h2>Table of Contents</h2>{toc_html}</div>{sections_html}<div class="footer"><p>Report generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p></div></body></html>"""
        
        safe_title = sanitize_filename(uploaded_base)
        html_file, pdf_file_report = f"Analyzed_{safe_title}.html", f"Analyzed_{safe_title}.pdf"
        with open(html_file, "w", encoding="utf-8") as f: f.write(html_content)

        try:
            loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)
            loop.run_until_complete(html_to_pdf(html_content, pdf_file_report))
        except Exception as e:
            st.error(f"PDF conversion failed: {e}"); pdf_file_report = None
        finally:
            loop.close()

        col1, col2 = st.columns(2)
        with col1:
            with open(html_file, "rb") as f: st.download_button("Download HTML", f.read(), file_name=html_file, mime="text/html", use_container_width=True)
        with col2:
            if pdf_file_report and os.path.exists(pdf_file_report):
                with open(pdf_file_report, "rb") as f: st.download_button("Download PDF", f.read(), file_name=pdf_file_report, mime="application/pdf", use_container_width=True)
            else:
                st.info("PDF report unavailable.")

def render_image_gallery_page():
    st.header("Extracted Images and Analysis")
    st.info("This gallery shows only the complex images that were sent to the vision model for detailed analysis.")
    if not st.session_state.extracted_images_data:
        st.info("No images from the document were identified as needing detailed vision analysis.")
        return
        
    for img_data in st.session_state.extracted_images_data:
        st.subheader(f"Image: {img_data['filename']}")
        if os.path.exists(img_data['image_path']):
            st.image(img_data['image_path'], use_container_width=True)
        
        with st.expander("View Analysis"):
            # --- NEW DISPLAY LOGIC START ---
            try:
                # Try to parse the analysis as JSON
                parsed_data = json.loads(img_data['analysis'])
                if isinstance(parsed_data, dict):
                    # If it's a dictionary, iterate and display all key-value pairs
                    for key, value in parsed_data.items():
                        if value and str(value).strip().lower() not in ['null', 'n/a', '']:
                            # Make key into a nice title, e.g. "explanation" -> "Explanation"
                            title = key.replace('_', ' ').title()
                            st.markdown(f"### {title}")
                            st.markdown(str(value))
                else:
                    # If it's valid JSON but not a dict (e.g., just a string), display it
                    st.markdown(str(parsed_data))
            except (json.JSONDecodeError, TypeError):
                # If it's not valid JSON, treat it as a single block of Markdown text
                st.markdown(img_data['analysis'])
            # --- NEW DISPLAY LOGIC END ---
        st.divider()


def render_chatbot_page():
    st.header("Chat with Your Document")
    
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about the document..."):
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response_data = generate_chat_response(
                    api_key=st.session_state.current_chat_api_key,
                    chat_session_obj=st.session_state.chat_session_object,
                    prompt_text=prompt,
                    thesis_text=st.session_state.pdf_text,
                    chat_history_summaries=st.session_state.chat_context_summaries,
                )
                response = response_data.get('answer', "I encountered an error. Please try again.")
                st.markdown(response)
        
        st.session_state.chat_messages.append({"role": "assistant", "content": response})
        new_q_summary = response_data.get("question_summary", "")
        new_a_summary = response_data.get("answer_summary", "")
        if new_q_summary and new_a_summary:
             st.session_state.chat_context_summaries.append((new_q_summary, new_a_summary))

# --- Main App Logic ---
st.title("Tech Transfer ChatBot")

with st.sidebar:
    st.header("Controls")
    if st.button("Start New Analysis", use_container_width=True):
        if path_to_clean := st.session_state.get('temp_pdf_path_for_cleanup'):
            if os.path.exists(path_to_clean): os.remove(path_to_clean)
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

if not st.session_state.get('processed'):
    st.info("Upload a PDF to begin analysis.")
    uploaded_file = st.file_uploader("Upload Research PDF", type=["pdf"])
    if uploaded_file:
        st.session_state.uploaded_file_name = uploaded_file.name
        os.makedirs(IMAGES_FOLDER, exist_ok=True)
        temp_pdf_path = f"temp_streamlit_{random.randint(1000,9999)}.pdf"
        st.session_state.temp_pdf_path_for_cleanup = temp_pdf_path
        with open(temp_pdf_path, "wb") as f: f.write(uploaded_file.getbuffer())
        
        with st.spinner("Processing PDF: extracting text, tables, and analyzing images..."):
            st.session_state.pdf_text = process_pdf_with_progress(
                temp_pdf_path, OUTPUT_TEXT_FILE, IMAGES_FOLDER,
                base_filename_for_output=st.session_state.uploaded_file_name
            )
        st.session_state.processed = True
        st.session_state.extracted_images_data = load_extracted_images_data(st.session_state.uploaded_file_name)
        
        if init_streamlit_chat_session():
            generate_all_initial_answers()
        
        st.rerun()
else:
    tab1, tab2, tab3 = st.tabs(["Main Analysis", "Image Gallery", "Chatbot"])
    with tab1:
        render_main_analysis_page()
    with tab2:
        render_image_gallery_page()
    with tab3:
        render_chatbot_page()
