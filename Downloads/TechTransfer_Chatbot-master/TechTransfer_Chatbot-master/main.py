# main.py
import streamlit as st
import os
import config # Uses updated config.py
from chatbot import initialize_model_for_chat, get_questions, generate_answer, define_question_batches, generate_answers_for_batch # Uses updated chatbot.py
from datetime import datetime
import asyncio
# main.py
# ... other imports like streamlit, os, config, etc. ...
from concurrent.futures import ThreadPoolExecutor, as_completed # <--- ADD THIS
# import platform # Already there from previous changes
# ...
from playwright.async_api import async_playwright
from pdf_utils import process_pdf_with_progress # Uses updated pdf_utils.py
import re
import platform 
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
        'questions': get_questions(), # This is a flat list of all questions
        'answers': {},
        'feedback_history': {},
        'report_ready': False,
        # 'current_page': 0, # Removed, no pagination
        'pdf_text': None,
        'initial_answers_generated': False,
        'temp_pdf_path_for_cleanup': None
    })

st.title("Tech Transfer ChatBot")

uploaded_file = st.file_uploader("Upload Research PDF", type=["pdf"])

def init_streamlit_chat_session(force_new_key=False):
    """Initializes or re-initializes the Streamlit chat session, optionally forcing a new key."""
    try:
        qna_api_key = config.get_available_api_key() #
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
    except ValueError as e_get_key: 
        st.error(f"Could not initialize chat: {e_get_key}. All API keys might be temporarily unavailable.")
        st.session_state.chat_session_object = None
        return False
    except Exception as e_chat_init:
        st.error(f"Generic error initializing chat for Q&A: {e_chat_init}")
        st.session_state.chat_session_object = None
        return False

# main.py
# ... other code ...

# main.py
# ... other code ...

# main.py

# You can define this near the top of your main.py or manage it in config.py if preferred
# This will determine how many Q&A batches are processed in parallel in the Streamlit app.
# It's good to base this on the number of API keys you have.
# With 19 keys (9 existing + 10 new), a value of 5-10 could be reasonable,
# depending on your CALLS_PER_MINUTE and how responsive you want the UI to be.
# Let's make it dynamically configurable based on available keys, up to a certain cap.
MAX_STREAMLIT_QA_WORKERS_CAP = 8  # Example: Cap at 8 parallel Q&A batches for UI responsiveness

def generate_all_initial_answers():
    """Generates answers for all questions in batches concurrently and stores them in session_state."""
    if not st.session_state.get('pdf_text') or st.session_state.get('initial_answers_generated'):
        logging.info("Skipping initial answer generation: PDF text not available or already generated.")
        return

    logging.info("Starting concurrent generation of all initial answers in batches.")
    st.session_state.answers = {} # Initialize/clear previous answers for this run

    all_question_batches = define_question_batches() # From chatbot.py
    if not all_question_batches:
        logging.warning("No question batches defined. Skipping initial answer generation.")
        st.session_state.initial_answers_generated = True
        return

    # Determine the number of workers for Q&A batch processing
    num_available_keys = len(config.GOOGLE_API_KEYS) if config.GOOGLE_API_KEYS else 0 #
    
    # Calculate effective workers: at least 1 if batches exist,
    # capped by MAX_STREAMLIT_QA_WORKERS_CAP, number of batches, and number of available keys.
    if num_available_keys == 0 and len(all_question_batches) > 0:
        logging.warning("No API keys configured, but attempting Q&A with 1 worker. This will likely fail if API calls are needed.")
        actual_workers = 1
    elif len(all_question_batches) == 0:
        actual_workers = 1 # No work to do, executor won't run.
    else:
        actual_workers = max(1, min(MAX_STREAMLIT_QA_WORKERS_CAP, len(all_question_batches), num_available_keys))

    with st.spinner(f"Generating initial analysis using up to {actual_workers} parallel batch processor(s)... This may take some time."):
        total_batches = len(all_question_batches)
        processed_batches_count = 0
        
        # Use st.empty() for elements that will be updated or removed
        progress_bar_overall_placeholder = st.empty()
        status_text_area_placeholder = st.empty()

        progress_bar_overall_placeholder.progress(0.0, text="Initializing concurrent Q&A batch processing...")

        futures = {}
        # Ensure executor runs with at least 1 worker if there's work to do.
        with ThreadPoolExecutor(max_workers=actual_workers) as executor:
            for batch_idx, (batch_name, questions_in_batch) in enumerate(all_question_batches.items()):
                logging.info(f"Submitting Q&A batch for concurrent processing: {batch_name} ({len(questions_in_batch)} questions)")
                # 'key_for_first_attempt' is None. chatbot.generate_answers_for_batch will acquire one.
                future = executor.submit(generate_answers_for_batch,
                                         None,  # key_for_first_attempt
                                         questions_in_batch,
                                         st.session_state.pdf_text,
                                         batch_name)
                futures[future] = (batch_name, questions_in_batch)

            for future_item in as_completed(futures):
                batch_name_completed, original_questions_in_batch = futures[future_item]
                processed_batches_count += 1
                progress_percent = processed_batches_count / total_batches if total_batches > 0 else 1.0
                
                try:
                    batch_answers_dict = future_item.result() # Get results from the completed future

                    for q_data in original_questions_in_batch:
                        param = q_data["param"]
                        answer_text = batch_answers_dict.get(param, f"Answer not found for {param} in batch '{batch_name_completed}'.")
                        
                        # Ensure answers dict structure is initialized correctly
                        if param not in st.session_state.answers:
                             st.session_state.answers[param] = {}
                        st.session_state.answers[param]['answer'] = answer_text
                        st.session_state.answers[param]['versions'] = [answer_text] 

                        if "Error:" in answer_text or "Failed:" in answer_text or "not found" in answer_text or "not in AI JSON" in answer_text :
                            logging.warning(f"Issue generating answer for {param} in batch {batch_name_completed}: {answer_text}")

                except Exception as e_batch_proc_concurrent: 
                    logging.error(f"Error processing batch '{batch_name_completed}' concurrently: {e_batch_proc_concurrent}", exc_info=True)
                    for q_data in original_questions_in_batch:
                        param = q_data["param"]
                        error_msg = f"Error generating answer for {param} due to batch processing failure: {str(e_batch_proc_concurrent)[:100]}..."
                        if param not in st.session_state.answers:
                             st.session_state.answers[param] = {}
                        st.session_state.answers[param]['answer'] = error_msg
                        st.session_state.answers[param]['versions'] = [error_msg]
                
                status_text_area_placeholder.info(f"Batch \"{batch_name_completed}\" processed. ({processed_batches_count}/{total_batches} batches complete).")
                progress_bar_overall_placeholder.progress(progress_percent)
        
        status_text_area_placeholder.empty()
        progress_bar_overall_placeholder.empty()

        st.session_state.initial_answers_generated = True
        logging.info("Finished concurrent generation of all initial answers.")
    st.success("Initial analysis for all questions complete!")
# ... rest of main.py ...
if uploaded_file and not st.session_state.processed:
    os.makedirs(IMAGES_FOLDER, exist_ok=True)
    temp_file_id = uploaded_file.id if hasattr(uploaded_file, 'id') else random.randint(1000,9999)
    temp_pdf_path = f"temp_streamlit_{temp_file_id}.pdf"
    st.session_state.temp_pdf_path_for_cleanup = temp_pdf_path 

    with open(temp_pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Main processing block
    processed_text_content = process_pdf_with_progress( #
        temp_pdf_path, OUTPUT_TEXT_FILE, IMAGES_FOLDER, None
    )
    st.session_state.pdf_text = processed_text_content
    st.session_state.processed = True
    
    init_ok = init_streamlit_chat_session() 
    if init_ok:
        generate_all_initial_answers() 
    else:
        st.error("Chat session could not be initialized. Some features like answer refinement might be affected.")
        if st.session_state.pdf_text:
             st.warning("Attempting to generate initial answers despite chat session init issue (refinement may fail later)...")
             generate_all_initial_answers()
        else:
             st.warning("PDF text not available, cannot generate initial answers.")
    
    # Success message moved here to appear after all initial processing
    st.success("PDF processed and initial analysis generated!")

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
        
        # Assuming 'uploaded_file' is in scope from st.file_uploader earlier in your script
        uploaded_filename = uploaded_file.name if uploaded_file else "Uploaded_File.pdf"
        uploaded_base = os.path.splitext(uploaded_filename)[0]

        toc_html = ""
        sections_html = ""
        def sanitize_filename(name): return "".join(c if c.isalnum() else "_" for c in name)[:50]

        for idx, q_data in enumerate(st.session_state.questions): # Report uses flat list
            param = q_data["param"]
            question_text = q_data["question"]
            anchor_id = f"q{idx+1}_{sanitize_filename(param)}" 
            toc_html += f'<a href="#{anchor_id}">Q{idx+1}: {param}</a>' #
            answer_info = st.session_state.answers.get(param, {})
            raw_answer = answer_info.get('answer', 'Not yet answered or answer unavailable.')
            
            # Ensure raw_answer is a string before string operations
            if not isinstance(raw_answer, str):
                raw_answer = str(raw_answer)
            
            answer_html = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', raw_answer.replace('\n', '<br>'))
            sections_html += f"""<div class="section" id="{anchor_id}"><div class="box"><div class="param">{param}</div><div class="question">Q{idx+1}: {question_text}</div><div class="answer">{answer_html}</div></div></div>"""

        html_content = f"""<!DOCTYPE html><html><head><meta charset="UTF-8"><title>{uploaded_base} - Tech Transfer Report</title><style>body{{font-family:Arial,sans-serif;margin:40px;line-height:1.6;}}.header{{text-align:center;border-bottom:2px solid #333;margin-bottom:30px;}}.toc{{margin:20px 0;padding:10px;border:1px solid #eee;background-color:#f9f9f9;border-radius:5px;}}.toc h2{{margin-top:0;}}.toc a{{display:block;margin:5px 0;color:#2980b9;text-decoration:none;}}.toc a:hover{{text-decoration:underline;}}.section{{margin-bottom:30px;page-break-inside:avoid;}}.box{{border:1px solid #ccc;padding:15px 20px;border-radius:8px;background-color:#fdfdfd;box-shadow:2px 2px 5px rgba(0,0,0,0.05);}}.param{{font-weight:bold;color:#2c3e50;font-size:1.15em;margin-bottom:5px;}}.question{{color:#555;font-size:1em;margin-top:0px;font-style:italic;}}.answer{{margin:10px 0 10px 15px;color:#34495e;}}.footer{{text-align:center;margin-top:40px;color:#777;font-size:0.9em;}}</style></head><body><div class="header"><h1>{uploaded_base}</h1><p><em>Analyzed and compiled by Tech Transfer ChatBot</em></p></div><div class="toc"><h2>Table of Contents</h2>{toc_html}</div>{sections_html}<div class="footer"><p>Report generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p></div></body></html>""" #
        
        safe_title = sanitize_filename(uploaded_base)
        html_file = f"Analyzed_{safe_title}.html"
        pdf_file_report = f"Analyzed_{safe_title}.pdf"
        
        with open(html_file, "w", encoding="utf-8") as f_html_write: f_html_write.write(html_content)
        
        # --- MODIFIED PDF GENERATION BLOCK for Windows asyncio/Playwright compatibility ---
        current_loop_for_pdf = None
        original_policy_for_pdf = None

        try:
            if platform.system() == "Windows":
                original_policy_for_pdf = asyncio.get_event_loop_policy()
                asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
            
            current_loop_for_pdf = asyncio.new_event_loop()
            asyncio.set_event_loop(current_loop_for_pdf)
            
            logging.info(f"Attempting PDF generation with event loop: {type(current_loop_for_pdf)}")
            # The html_to_pdf function is defined in your main.py
            current_loop_for_pdf.run_until_complete(html_to_pdf(html_content, pdf_file_report)) 
            logging.info("PDF generation task completed or handled within run_until_complete.")
            
        except Exception as e_gen_pdf:
            st.error(f"PDF conversion for report failed: {e_gen_pdf}")
            logging.error(f"PDF report generation failed: {e_gen_pdf}", exc_info=True)
            pdf_file_report = None 
        finally:
            if current_loop_for_pdf is not None:
                if not current_loop_for_pdf.is_closed():
                    current_loop_for_pdf.close()
                    logging.info("Dedicated event loop for PDF generation has been closed.")
            
            if original_policy_for_pdf is not None and platform.system() == "Windows":
                asyncio.set_event_loop_policy(original_policy_for_pdf)
                logging.info("Restored original asyncio event loop policy for Windows.")
        # --- END OF MODIFIED PDF GENERATION BLOCK ---

        col1,col2,col3=st.columns(3)
        with col1:
            with open(html_file,"rb") as f_html_read: st.download_button("Download Report (HTML)",f_html_read.read(),file_name=html_file,mime="text/html")
        with col2:
            if pdf_file_report and os.path.exists(pdf_file_report):
                with open(pdf_file_report,"rb") as f_pdf_read: st.download_button("Download Report (PDF)",f_pdf_read.read(),file_name=pdf_file_report,mime="application/pdf")
            else: st.info("PDF report was not generated or is unavailable.")
        with col3:
            if st.button("Start New Analysis"):
                keys_to_del=['processed','chat_session_object','current_chat_api_key','answers','feedback_history','report_ready','pdf_text', 'initial_answers_generated', 'temp_pdf_path_for_cleanup'] #
                
                path_to_clean = st.session_state.pop('temp_pdf_path_for_cleanup', None) 
                if path_to_clean and os.path.exists(path_to_clean):
                    try: 
                        os.remove(path_to_clean)
                        logging.info(f"Successfully removed temp PDF: {path_to_clean}")
                    except Exception as e_remove: 
                        logging.warning(f"Could not remove temp PDF {path_to_clean}: {e_remove}")
                else:
                    if path_to_clean: logging.warning(f"Temp PDF path {path_to_clean} was stored but file not found for cleanup.")

                # Assuming OUTPUT_TEXT_FILE is defined globally in your main.py
                if 'OUTPUT_TEXT_FILE' in globals() and os.path.exists(OUTPUT_TEXT_FILE):
                    try: os.remove(OUTPUT_TEXT_FILE)
                    except Exception as e_remove_txt: logging.warning(f"Could not remove temp text file {OUTPUT_TEXT_FILE}: {e_remove_txt}")
                elif 'OUTPUT_TEXT_FILE' not in globals():
                    logging.warning("OUTPUT_TEXT_FILE global variable not found, cannot clean up.")


                for k_del in keys_to_del:
                    if k_del in st.session_state: del st.session_state[k_del]
                st.rerun()

    else: # Q&A Display Logic - Modified for single page grouped display
        if not st.session_state.initial_answers_generated and st.session_state.pdf_text:
            st.warning("Initial answers not yet fully generated. Attempting now or please wait...")
            # Potentially add a button to trigger generation if it failed, or rely on automatic flow.
            # For now, assuming generate_all_initial_answers tried its best.
            # If it failed catastrophically, errors would be in the answers.
            if not any(st.session_state.answers): # If truly no answers at all
                 generate_all_initial_answers() 
                 st.rerun()


        all_question_batches = define_question_batches() # From chatbot.py

        for batch_idx, (batch_name, questions_in_batch) in enumerate(all_question_batches.items()):
            st.markdown(f"## {batch_name}") # Main heading for the group of questions

            for q_idx_in_batch, q_data in enumerate(questions_in_batch):
                param, question = q_data["param"], q_data["question"]
                
                if param not in st.session_state.answers: 
                    st.session_state.answers[param] = {'answer': 'Answer being processed or encountered an issue.', 'versions': []}
                if param not in st.session_state.feedback_history: 
                    st.session_state.feedback_history[param] = []
            
                if not st.session_state.pdf_text: 
                    st.info("PDF not processed yet. Please upload and process a PDF.")
                    break # Break from inner loop
            
                st.markdown(f"### {param}") # Subheading for the question parameter (title)
                st.markdown(f"**Question:** *{question}*") # Display the actual question
                
                current_answer_val = st.session_state.answers[param].get('answer', '')
                edited_answer = st.text_area(f"Your Answer for {param}",value=current_answer_val, height=180, key=f"editor_{param}")
                
                if st.button(f"💾 Save Changes to {param}", key=f"save_{param}"):
                    st.session_state.answers[param]['answer'] = edited_answer
                    if 'versions' not in st.session_state.answers[param] or not st.session_state.answers[param]['versions']: 
                        st.session_state.answers[param]['versions'] = [current_answer_val if current_answer_val else "Initial empty answer"]
                    st.session_state.answers[param]['versions'].append(f"Manual edit: {edited_answer}")
                    st.success(f"Saved answer for {param}"); st.rerun()
                
                with st.expander(f"💬 Refine Answer for `{param}`"):
                    feedback = st.text_area("Your feedback to improve the answer:", key=f"feedback_input_{param}")
                    if st.button(f"🔁 Refine Answer for {param}", key=f"refine_{param}"):
                        if not st.session_state.chat_session_object:
                             st.warning("Chat session not active for refinement. Trying to re-initialize...")
                             if not init_streamlit_chat_session():
                                  st.error("Still unable to initialize chat session. Cannot refine answer.")
                             else:
                                  st.info("Chat session re-initialized. Please try refining again.")
                                  st.rerun()

                        if feedback.strip() and st.session_state.pdf_text and st.session_state.chat_session_object:
                            st.session_state.feedback_history[param].append(feedback)
                            with st.spinner("Refining answer..."):
                                refinement_prompt = (f"Original Question: {question}\n\n Current Answer: {edited_answer}\n\n User Feedback for Refinement: {feedback}\n\n Please provide a refined answer based on the feedback, using the Full Reference Text provided.")
                                
                                refined_answer_text = generate_answer( #
                                    st.session_state.current_chat_api_key, 
                                    st.session_state.chat_session_object,
                                    refinement_prompt,
                                    st.session_state.pdf_text
                                )
                                st.session_state.answers[param]['answer'] = refined_answer_text
                                st.session_state.answers[param]['versions'].append(refined_answer_text)
                            st.success("Refinement complete!"); st.rerun()
                        elif not st.session_state.pdf_text: st.warning("PDF text not available.")
                        elif not st.session_state.chat_session_object: st.warning("Chat session not available for refinement.")
                        else: st.warning("Please provide feedback.")

                with st.expander("🕘 Edit History"):
                    versions = st.session_state.answers[param].get('versions', [])
                    if versions:
                        for ver_i, ver_txt in enumerate(versions[::-1],1): 
                            st.markdown(f"**Version {len(versions)-ver_i+1}:**")
                            st.markdown(f"> {ver_txt.replace(chr(10), chr(10) + '> ')}") 
                    else: st.markdown("No edit history yet.")
                st.divider() # Divider after each full question block
            
            if not st.session_state.pdf_text: # If PDF text became unavailable, break outer loop too
                break
        
        # Removed pagination buttons
        
        if all_questions_answered() and not st.session_state.report_ready:
            st.success("🎉 All questions have initial answers!") 
            if st.button("📥 Generate and Download Final Report", use_container_width=True):
                st.session_state.report_ready=True; st.rerun()