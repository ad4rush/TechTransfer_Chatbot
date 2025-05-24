# main.py
import streamlit as st
import os
import config # Uses updated config.py
from chatbot import initialize_model_for_chat, get_questions, generate_answer # Uses updated chatbot.py
from datetime import datetime
import asyncio
from playwright.async_api import async_playwright
from pdf_utils import process_pdf_with_progress # Uses updated pdf_utils.py
import re
import logging
import random 

if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(module)s.%(funcName)s:%(lineno)d] - %(message)s')

OUTPUT_TEXT_FILE = "streamlit_extracted_text.txt" 
IMAGES_FOLDER = "streamlit_extracted_images"    

if 'processed' not in st.session_state:
    st.session_state.update({
        'processed': False,
        'chat_session_object': None, 
        'current_chat_api_key': None, 
        'questions': get_questions(), 
        'answers': {}, 
        'feedback_history': {},
        'manual_edits': {},
        'report_ready': False,
        'current_page': 0 
    })

st.title("Tech Transfer ChatBot")

uploaded_file = st.file_uploader("Upload Research PDF", type=["pdf"])

def init_streamlit_chat_session(force_new_key=False):
    """Initializes or re-initializes the Streamlit chat session, optionally forcing a new key."""
    current_key_to_exclude = st.session_state.current_chat_api_key if force_new_key else None
    try:
        # Get a new available key, excluding the current one if we're forcing a refresh due to its failure
        qna_api_key = config.get_available_api_key(exclude_keys=[current_key_to_exclude] if current_key_to_exclude else None)
        st.session_state.current_chat_api_key = qna_api_key 
        model_for_streamlit_chat = initialize_model_for_chat(qna_api_key) # From chatbot.py
        if model_for_streamlit_chat:
            st.session_state.chat_session_object = model_for_streamlit_chat.start_chat()
            logging.info(f"Streamlit chat session initialized/re-initialized with key ...{qna_api_key[-6:]}")
            return True
        else:
            st.error("Failed to initialize chat model for Q&A (model object was None).")
            st.session_state.chat_session_object = None
            return False
    except ValueError as e_get_key: # Handles all keys resting case from get_available_api_key
        st.error(f"Could not initialize chat: {e_get_key}. All API keys might be temporarily unavailable.")
        st.session_state.chat_session_object = None
        return False
    except Exception as e_chat_init:
        st.error(f"Generic error initializing chat for Q&A: {e_chat_init}")
        st.session_state.chat_session_object = None
        return False

if uploaded_file and not st.session_state.processed:
    os.makedirs(IMAGES_FOLDER, exist_ok=True)
    temp_file_id = uploaded_file.id if hasattr(uploaded_file, 'id') else random.randint(1000,9999)
    temp_pdf_path = f"temp_streamlit_{temp_file_id}.pdf"
    with open(temp_pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    with st.spinner("Processing PDF: extracting text and analyzing images... This may take some time."):
        # process_pdf_with_progress uses its own internal key management for image tasks
        processed_text_content = process_pdf_with_progress(
            temp_pdf_path, OUTPUT_TEXT_FILE, IMAGES_FOLDER, None 
        )
        st.session_state.pdf_text = processed_text_content 
        st.session_state.processed = True
        init_streamlit_chat_session() # Initialize chat after processing PDF
            
    st.success("PDF processed successfully!")
    
    if os.path.exists(OUTPUT_TEXT_FILE):
        with open(OUTPUT_TEXT_FILE, "rb") as fp:
            btn_data = fp.read()
        st.download_button(
            label="Download Extracted Text & Image Analysis",
            data=btn_data,
            file_name=f"{os.path.splitext(uploaded_file.name)[0]}_extracted_analysis.txt",
            mime="text/plain"
        )
    else:
        st.warning("Could not find the extracted text file for download.")

async def html_to_pdf(html_content, output_path):
    # ... (html_to_pdf remains the same) ...
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.set_content(html_content)
        await page.pdf(path=output_path)
        await browser.close()

def all_questions_answered():
    # ... (all_questions_answered remains the same) ...
    if not st.session_state.get('questions'): return False
    return all(
        (q_data["param"] in st.session_state.answers and 
         st.session_state.answers[q_data["param"]].get('answer','').strip())
        for q_data in st.session_state.questions
    )

if st.session_state.processed:
    st.divider()
    st.header("Your Research Analysis!")

    if st.session_state.report_ready:
        # ... (Report generation HTML logic - same as before) ...
        st.header("✅ Analysis Complete!")
        st.markdown("### Final Report")
        uploaded_filename = uploaded_file.name if uploaded_file else "Uploaded_File.pdf"
        uploaded_base = os.path.splitext(uploaded_filename)[0]
        
        toc_html = ""
        sections_html = ""
        for idx, q_data in enumerate(st.session_state.questions):
            param = q_data["param"]
            question_text = q_data["question"]
            toc_html += f'<a href="#q{idx+1}">Q{idx+1}: {param}</a>'
            answer_info = st.session_state.answers.get(param, {})
            raw_answer = answer_info.get('answer', 'Not yet answered or answer unavailable.')
            answer_html = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', raw_answer.replace('\n', '<br>')) 
            sections_html += f"""<div class="section" id="q{idx+1}"><div class="box"><div class="param">{param}</div><div class="question">Q{idx+1}: {question_text}</div><div class="answer">{answer_html}</div></div></div>"""

        html_content = f"""<!DOCTYPE html><html><head><meta charset="UTF-8"><title>{uploaded_base} - Tech Transfer Report</title><style>body{{font-family:Arial,sans-serif;margin:40px;line-height:1.6;}}.header{{text-align:center;border-bottom:2px solid #333;margin-bottom:30px;}}.toc{{margin:20px 0;padding:10px;border:1px solid #eee;background-color:#f9f9f9;border-radius:5px;}}.toc h2{{margin-top:0;}}.toc a{{display:block;margin:5px 0;color:#2980b9;text-decoration:none;}}.toc a:hover{{text-decoration:underline;}}.section{{margin-bottom:30px;}}.box{{border:1px solid #ccc;padding:15px 20px;border-radius:8px;background-color:#fdfdfd;box-shadow:2px 2px 5px rgba(0,0,0,0.05);}}.param{{font-weight:bold;color:#2c3e50;font-size:1.15em;margin-bottom:5px;}}.question{{color:#555;font-size:1em;margin-top:0px;font-style:italic;}}.answer{{margin:10px 0 10px 15px;color:#34495e;}}.footer{{text-align:center;margin-top:40px;color:#777;font-size:0.9em;}}</style></head><body><div class="header"><h1>{uploaded_base}</h1><p><em>Analyzed and compiled by Tech Transfer ChatBot</em></p></div><div class="toc"><h2>Table of Contents</h2>{toc_html}</div>{sections_html}<div class="footer"><p>Report generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p></div></body></html>"""
        def sanitize_filename(name): return "".join(c if c.isalnum() else "_" for c in name)[:50]
        safe_title = sanitize_filename(uploaded_base)
        html_file = f"Analyzed_{safe_title}.html"
        pdf_file_report = f"Analyzed_{safe_title}.pdf" 
        with open(html_file, "w", encoding="utf-8") as f_html_write: f_html_write.write(html_content)
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(html_to_pdf(html_content, pdf_file_report))
        except Exception as e_gen_pdf: 
            st.error(f"PDF conversion for report failed: {e_gen_pdf}")
            pdf_file_report = None
        
        col1,col2,col3=st.columns(3)
        # ... (Download button logic as before) ...
        with col1:
            with open(html_file,"rb") as f_html_read: st.download_button("Download Report (HTML)",f_html_read.read(),file_name=html_file,mime="text/html")
        with col2:
            if pdf_file_report and os.path.exists(pdf_file_report):
                with open(pdf_file_report,"rb") as f_pdf_read: st.download_button("Download Report (PDF)",f_pdf_read.read(),file_name=pdf_file_report,mime="application/pdf")
            else: st.info("PDF report was not generated or is unavailable.")
        with col3:
            if st.button("Start New Analysis"): # Renamed button for clarity
                keys_to_del=['processed','chat_session_object','current_chat_api_key','answers','feedback_history','manual_edits','report_ready','pdf_text','current_page']
                # temp_pdf_path was defined earlier in the script when a file is uploaded
                temp_pdf_path_to_delete = f"temp_streamlit_{uploaded_file.id if hasattr(uploaded_file, 'id') else ''}.pdf" # Reconstruct or store temp_pdf_path in session_state if needed across reruns before this point
                if 'temp_pdf_path' in locals() and os.path.exists(temp_pdf_path): # Use temp_pdf_path if it's in scope
                    try: os.remove(temp_pdf_path) 
                    except Exception as e_remove: logging.warning(f"Could not remove temp PDF: {e_remove}")

                if os.path.exists(OUTPUT_TEXT_FILE): 
                    try: os.remove(OUTPUT_TEXT_FILE)
                    except Exception as e_remove_txt: logging.warning(f"Could not remove temp text file: {e_remove_txt}")

                for k_del in keys_to_del: 
                    if k_del in st.session_state: del st.session_state[k_del]
                st.experimental_rerun()

    else: # Q&A Pagination Logic
        questions_per_page = 3
        total_questions = len(st.session_state.questions)
        total_pages = (total_questions + questions_per_page - 1) // questions_per_page
        start_idx = st.session_state.current_page * questions_per_page
        end_idx = min(start_idx + questions_per_page, total_questions)
        current_batch_qs = st.session_state.questions[start_idx:end_idx]
        st.subheader(f"Answering Questions {start_idx+1} to {end_idx} of {total_questions}")

        for i, q_data in enumerate(current_batch_qs):
            param, question = q_data["param"], q_data["question"]
            if param not in st.session_state.answers: st.session_state.answers[param] = {'answer': '', 'versions': []}
            if param not in st.session_state.feedback_history: st.session_state.feedback_history[param] = []

            if not st.session_state.answers[param].get('versions') and st.session_state.pdf_text:
                if not st.session_state.chat_session_object:
                    st.warning("Chat session not active. Trying to initialize...")
                    if not init_streamlit_chat_session(): 
                        st.error("Still unable to initialize chat session. Cannot generate new answers.")
                    else:
                        st.experimental_rerun() # Rerun to reflect new session state & try generating
                
                if st.session_state.chat_session_object: # Check again
                    with st.spinner(f"Generating answer for Q{start_idx + i + 1}: {param}..."):
                        response_text = generate_answer(
                            st.session_state.current_chat_api_key, 
                            st.session_state.chat_session_object, 
                            question, 
                            st.session_state.pdf_text
                        )
                        # If generate_answer had to use a new key and create a new session,
                        # st.session_state.chat_session_object might be stale.
                        # For true robustness, generate_answer could return the new session if it changed.
                        # For now, we assume the initial session or its internal retries manage.
                        st.session_state.answers[param]['answer'] = response_text
                        st.session_state.answers[param]['versions'] = [response_text] 
                        if "Error:" in response_text: # If error, show it
                            st.error(f"Could not generate answer for {param}: {response_text}")
                        st.experimental_rerun()
            
            elif not st.session_state.pdf_text:
                st.info("PDF not processed yet. Please upload and process a PDF.")

            st.markdown(f"### Q{start_idx + i + 1}")
            st.markdown(f"**Parameter:** `{param}`")
            st.markdown(f"**Q:** {question}")
            edited_answer = st.text_area(f"Answer for {param}",value=st.session_state.answers[param].get('answer', ''),height=180,key=f"editor_{param}")
            if st.button(f"💾 Save Changes to {param}", key=f"save_{param}"):
                st.session_state.answers[param]['answer'] = edited_answer
                if 'versions' not in st.session_state.answers[param]: st.session_state.answers[param]['versions'] = []
                st.session_state.answers[param]['versions'].append(f"Manual edit: {edited_answer}")
                st.success(f"Saved answer for {param}"); st.experimental_rerun()
            
            with st.expander(f"💬 Refine Answer for `{param}`"):
                feedback = st.text_area("Your feedback to improve the answer:", key=f"feedback_input_{param}")
                if st.button(f"🔁 Refine Answer for {param}", key=f"refine_{param}"):
                    if feedback.strip() and st.session_state.pdf_text and st.session_state.chat_session_object:
                        st.session_state.feedback_history[param].append(feedback)
                        with st.spinner("Refining answer..."):
                            refinement_prompt = (f"Original Question: {question}\n\n Current Answer: {edited_answer}\n\n User Feedback for Refinement: {feedback}\n\n Please provide a refined answer based on the feedback, using the Full Reference Text provided.")
                            refined_answer_text = generate_answer(
                                st.session_state.current_chat_api_key,
                                st.session_state.chat_session_object, 
                                refinement_prompt, 
                                st.session_state.pdf_text
                            )
                            st.session_state.answers[param]['answer'] = refined_answer_text
                            st.session_state.answers[param]['versions'].append(refined_answer_text)
                        st.success("Refinement complete!"); st.experimental_rerun()
                    elif not st.session_state.pdf_text: st.warning("PDF text not available.")
                    elif not st.session_state.chat_session_object: st.warning("Chat session not available.")
                    else: st.warning("Please provide feedback.")

            with st.expander("🕘 Edit History"):
                versions = st.session_state.answers[param].get('versions', [])
                if versions:
                    for ver_i, ver_txt in enumerate(versions[::-1],1): st.markdown(f"**Version {len(versions)-ver_i+1}:** {ver_txt}") 
                else: st.markdown("No edit history yet.")
        
        st.divider()
        cn1,_,cn2 = st.columns([1,3,1])
        with cn1:
            if st.button("⬅ Previous Set",disabled=(st.session_state.current_page==0),use_container_width=True):
                st.session_state.current_page-=1; st.experimental_rerun()
        with cn2:
            if st.button("Next Set ➡",disabled=(st.session_state.current_page>=total_pages-1),use_container_width=True):
                st.session_state.current_page+=1; st.experimental_rerun()
        if all_questions_answered() and not st.session_state.report_ready:
            st.success("🎉 All questions have initial answers!")
            if st.button("📥 Generate and Download Final Report", use_container_width=True):
                st.session_state.report_ready=True; st.experimental_rerun()