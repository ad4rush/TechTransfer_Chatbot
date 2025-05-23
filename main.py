# main.py
import streamlit as st
import os
import json # Not directly used for Q&A generation here, but good to have if needed elsewhere
import config
from chatbot import initialize_chat_model, get_questions, generate_answer # Using updated initialize_chat_model
from datetime import datetime
import asyncio
from playwright.async_api import async_playwright
from pdf_utils import process_pdf_with_progress # Uses the updated version
import re
import logging

# Configure logging for Streamlit app if desired, or rely on console output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s')


# Configuration
OUTPUT_TEXT_FILE = "streamlit_extracted_text.txt" 
IMAGES_FOLDER = "streamlit_extracted_images"    

# Session state initialization
if 'processed' not in st.session_state:
    st.session_state.update({
        'processed': False,
        'chat_session_object': None, # Will store the chat object
        'questions': get_questions(), # Uses the flat list from get_questions()
        'answers': {}, 
        'feedback_history': {},
        'manual_edits': {},
        'report_ready': False,
        'current_page': 0 
    })

st.title("Tech Transfer ChatBot")

# 1. PDF Processing Section
uploaded_file = st.file_uploader("Upload Research PDF", type=["pdf"])

if uploaded_file and not st.session_state.processed:
    os.makedirs(IMAGES_FOLDER, exist_ok=True)
    temp_pdf_path = os.path.join("temp_streamlit_upload.pdf")
    with open(temp_pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Use one key for the main PDF processing step; image tasks within will get diverse keys
    pdf_processing_main_api_key = config.get_random_google_api_key() 
    
    with st.spinner("Processing PDF: extracting text and analyzing images... This may take some time."):
        processed_text_content = process_pdf_with_progress(
            temp_pdf_path, 
            OUTPUT_TEXT_FILE, 
            IMAGES_FOLDER, 
            pdf_processing_main_api_key # This key is for the overall process_pdf_with_progress
                                        # Sub-tasks for images will now also attempt diverse keys via config
        )
        st.session_state.pdf_text = processed_text_content 
        st.session_state.processed = True
        
        # Initialize a chat session for the Q&A part of Streamlit app
        # This chat session will be used by generate_answer for single questions
        try:
            qna_api_key = config.get_random_google_api_key()
            # Assuming initialize_chat_model returns the model, and then we start_chat
            model_for_streamlit_chat = initialize_chat_model(qna_api_key)
            if model_for_streamlit_chat:
                st.session_state.chat_session_object = model_for_streamlit_chat.start_chat()
            else:
                st.error("Failed to initialize chat model for Q&A.")
                st.session_state.chat_session_object = None
        except Exception as e_chat_init:
            st.error(f"Error initializing chat for Q&A: {e_chat_init}")
            st.session_state.chat_session_object = None
            
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

if st.session_state.processed:
    st.divider()
    st.header("Your Research Analysis!")

    if st.session_state.report_ready:
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
        pdf_file = f"Analyzed_{safe_title}.pdf"
        with open(html_file, "w", encoding="utf-8") as f: f.write(html_content)
        try:
            asyncio.run(html_to_pdf(html_content, pdf_file))
        except RuntimeError as e:
            if "cannot be called from a running event loop" in str(e):
                loop = asyncio.get_event_loop()
                loop.run_until_complete(html_to_pdf(html_content, pdf_file))
            else:
                st.error(f"PDF conversion failed: {e}"); pdf_file = None
        except Exception as e: st.error(f"PDF conversion failed: {e}"); pdf_file = None
        col1,col2,col3=st.columns(3)
        with col1:
            with open(html_file,"rb") as f_html: st.download_button("Download Report (HTML)",f_html,file_name=html_file,mime="text/html")
        with col2:
            if pdf_file and os.path.exists(pdf_file):
                with open(pdf_file,"rb") as f_pdf: st.download_button("Download Report (PDF)",f_pdf,file_name=pdf_file,mime="application/pdf")
            else: st.info("PDF report could not be generated.")
        with col3:
            if st.button("Start New Analysis"):
                keys_to_del=['processed','chat_session_object','answers','feedback_history','manual_edits','report_ready','pdf_text','current_page']
                for k in keys_to_del:
                    if k in st.session_state: del st.session_state[k]
                if 'temp_pdf_path' in locals() and os.path.exists(temp_pdf_path): os.remove(temp_pdf_path) # Defined earlier
                if os.path.exists(OUTPUT_TEXT_FILE): os.remove(OUTPUT_TEXT_FILE)
                st.experimental_rerun()
    else:
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

            if not st.session_state.answers[param]['versions'] and st.session_state.pdf_text and st.session_state.chat_session_object:
                with st.spinner(f"Generating answer for Q{start_idx + i + 1}: {param}..."):
                    response_text = generate_answer(st.session_state.chat_session_object, question, st.session_state.pdf_text)
                    st.session_state.answers[param]['answer'] = response_text
                    st.session_state.answers[param]['versions'].append(response_text)
                    st.experimental_rerun()
            elif not st.session_state.chat_session_object and st.session_state.pdf_text:
                 st.warning("Chat session not initialized. Cannot generate answers.")


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
                            refined_answer_text = generate_answer(st.session_state.chat_session_object, refinement_prompt, st.session_state.pdf_text)
                            st.session_state.answers[param]['answer'] = refined_answer_text
                            st.session_state.answers[param]['versions'].append(refined_answer_text)
                        st.success("Refinement complete!"); st.experimental_rerun()
                    elif not st.session_state.pdf_text: st.warning("PDF text not available for refinement.")
                    elif not st.session_state.chat_session_object: st.warning("Chat session not available for refinement.")
                    else: st.warning("Please provide feedback to refine the answer.")
            with st.expander("🕘 Edit History"):
                versions = st.session_state.answers[param].get('versions', [])
                if versions:
                    for ver_i, ver_txt in enumerate(versions[::-1],1): st.markdown(f"**Version {len(versions)-ver_i}:** {ver_txt}")
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