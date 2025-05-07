# main.py
import streamlit as st
import os
import json
import config
from chatbot import initialize_chat, get_questions, generate_answer
from datetime import datetime
import asyncio
from playwright.async_api import async_playwright
from pdf_utils import process_pdf_with_progress
import re


# Configuration
OUTPUT_TEXT_FILE = "extracted_text.txt"
IMAGES_FOLDER = "extracted_images"

# Session state initialization
if 'processed' not in st.session_state:
    st.session_state.update({
        'processed': False,
        'chat_initialized': False,
        'current_question': 0,
        'questions': get_questions(),
        'answers': {},
        'feedback_history': {},
        'manual_edits': {},
        'report_ready': False  # ✅ Added report_ready
    })

st.title("Tech Transfer ChatBot")

# 1. PDF Processing Section
uploaded_file = st.file_uploader("Upload Research PDF", type=["pdf"])
if uploaded_file and not st.session_state.processed:
    with st.spinner("Processing PDF..."):
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        fresh_api_key = config.get_random_google_api_key()
        #process_pdf("temp.pdf", OUTPUT_TEXT_FILE, IMAGES_FOLDER, fresh_api_key)
        process_pdf_with_progress("temp.pdf", OUTPUT_TEXT_FILE, IMAGES_FOLDER, fresh_api_key)
        st.session_state.pdf_text = open(OUTPUT_TEXT_FILE).read()
        st.session_state.processed = True
        st.session_state.chat = initialize_chat(fresh_api_key)
        
    st.success("PDF processed successfully!")
    st.download_button(
        label="Download Extracted Text",
        data=open(OUTPUT_TEXT_FILE, "rb").read(),
        file_name="extracted_text.txt",
        mime="text/plain"
    )

async def html_to_pdf(html_content, output_path):
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.set_content(html_content)
        await page.pdf(path=output_path)
        await browser.close()

# Helper to check all answered
def all_questions_answered():
    return all(
        (q_data["param"] in st.session_state.answers and st.session_state.answers[q_data["param"]]['answer'])
        for q_data in st.session_state.questions
    )

# 2. Q&A Section
if st.session_state.processed:
    st.divider()
    st.header("Your Research Analysis!")

    if st.session_state.report_ready:
        # FINAL REPORT PAGE
        st.header("✅ Analysis Complete!")
        st.markdown("### Final Report")

        # Personalized report with Table of Contents
        uploaded_filename = uploaded_file.name if uploaded_file else "Uploaded_File"
        uploaded_base = os.path.splitext(uploaded_filename)[0]
        

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>{uploaded_base} - Tech Transfer Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ text-align: center; border-bottom: 2px solid #333; margin-bottom: 30px; }}
                .toc {{ margin: 20px 0; }}
                .toc a {{ display: block; margin: 5px 0; color: #2980b9; text-decoration: none; }}
                .section {{ margin-bottom: 40px; }}
                .param {{ font-weight: bold; color: #2c3e50; font-size: 1.1em; }}
                .question {{ color: #555; font-size: 1em; margin-top: 10px; }}
                .answer {{ margin: 10px 0 20px 20px; color: #34495e; line-height: 1.6; }}
                .footer {{ text-align: center; margin-top: 40px; color: #666; }}
                .box {{
                    border: 1px solid #ccc;
                    padding: 20px;
                    border-radius: 10px;
                    background-color: #f9f9f9;
                    margin-top: 20px;
                }}

            </style>
        </head>
        <body>

        <div class="header">
            <h1>{uploaded_base}</h1>
            <p><em>Analyzed and compiled by Tech Transfer ChatBot</em></p>
        </div>

        <div class="toc">
            <h2>Table of Contents</h2>
        """

        # Table of Contents links
        for idx, q_data in enumerate(st.session_state.questions):
            param = q_data["param"]
            html_content += f'<a href="#q{idx+1}">Q{idx+1}: {param}</a>'

        html_content += """
        </div>
        """

        # Question and Answer sections
        for idx, q_data in enumerate(st.session_state.questions):
            param = q_data["param"]
            question = q_data["question"]
            raw_answer = st.session_state.answers[param]['answer']
            # Convert **text** into <b>text</b> and replace newlines
            answer_html = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', raw_answer.replace('\n', '<br>'))


            html_content += f"""
            <div class="section" id="q{idx+1}">
                <div class="box">
                    <div class="param">{param}</div>
                    <div class="question">Q{idx+1}: {question}</div>
                    <div class="answer">{answer_html}</div>
                </div>
            </div>
            """

        # Footer
        html_content += f"""
        <div class="footer">
            <p>Report generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        </div>

        </body>
        </html>
        """



        # Clean filename parts (remove spaces, special chars)
        def sanitize_filename(name):
            return "".join(c if c.isalnum() else "_" for c in name)[:50]  # Max 50 chars to be safe

        safe_title = sanitize_filename(uploaded_base)
       

        # Build file names
        html_file = f"Analyzed_{safe_title}.html"
        pdf_file = f"Analyzed_{safe_title}.pdf"

        # Write HTML file
        with open(html_file, "w", encoding="utf-8") as f:
            f.write(html_content)

        # Generate PDF
        try:
            asyncio.run(html_to_pdf(html_content, pdf_file))
        except Exception as e:
            st.error("PDF conversion failed.")
            pdf_file = None


        col1, col2, col3 = st.columns(3)
        with col1:
            with open(html_file, "rb") as f:
                st.download_button(
                    label="Download Full Report (HTML)",
                    data=f,
                    file_name=html_file,
                    mime="text/html",
                    help="Open in browser and use 'Print > Save as PDF' for PDF version"
                )
        with col2:
            if pdf_file and os.path.exists(pdf_file):
                with open(pdf_file, "rb") as f:
                    st.download_button(
                        label="Download Full Report (PDF)",
                        data=f,
                        file_name=pdf_file,
                        mime="application/pdf"
                    )
            else:
                st.write("PDF not available.")
        with col3:
            if st.button("Upload Another PDF"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()

    else:
        # QUESTIONS PAGE
        questions_per_page = 3
        total_questions = len(st.session_state.questions)
        total_pages = (total_questions + questions_per_page - 1) // questions_per_page

        if 'current_page' not in st.session_state:
            st.session_state.current_page = 0

        start_idx = st.session_state.current_page * questions_per_page
        end_idx = min(start_idx + questions_per_page, total_questions)
        current_batch = st.session_state.questions[start_idx:end_idx]

        st.subheader(f"Questions {start_idx+1} to {end_idx} of {total_questions}")

        for i, q_data in enumerate(current_batch):
            param = q_data["param"]
            question = q_data["question"]

            if param not in st.session_state.answers:
                st.session_state.answers[param] = {'answer': '', 'versions': []}
                st.session_state.feedback_history[param] = []

            if not st.session_state.answers[param]['versions']:
                with st.spinner(f"Generating answer for Q{start_idx + i + 1}..."):
                    fresh_api_key = config.get_random_google_api_key()
                    fresh_chat = initialize_chat(fresh_api_key)

                    response = generate_answer(
                        fresh_chat,
                        question,
                        st.session_state.pdf_text
                    )
                    st.session_state.answers[param]['versions'].append(response)
                    st.session_state.answers[param]['answer'] = response

            st.markdown(f"### Q{start_idx + i + 1}")
            st.markdown(f"**Parameter:** `{param}`")
            st.markdown(f"**Q:** {question}")

            edited_answer = st.text_area(
                f"Answer for {param}",
                value=st.session_state.answers[param]['answer'],
                height=180,
                key=f"editor_{param}"
            )

            if st.button(f"💾 Save Changes to {param}", key=f"save_{param}"):
                st.session_state.answers[param]['answer'] = edited_answer
                st.session_state.answers[param]['versions'].append(f"Manual edit: {edited_answer}")
                st.success(f"Saved answer for {param}")
                st.rerun()

            with st.expander(f"💬 Refine Answer for `{param}`"):
                feedback = st.text_area("Enter feedback", key=f"feedback_input_{param}")
                if st.button(f"🔁 Refine Answer for {param}", key=f"refine_{param}"):
                    if feedback.strip():
                        st.session_state.feedback_history[param].append(feedback)
                        with st.spinner("Refining..."):
                            prompt = f"{question}\n\nPrevious answer: {edited_answer}\nFeedback: {feedback}"

                            fresh_api_key = config.get_random_google_api_key()
                            fresh_chat = initialize_chat(fresh_api_key)

                            refined_answer = generate_answer(
                                fresh_chat,
                                prompt,
                                st.session_state.pdf_text
                            )
                            st.session_state.answers[param]['versions'].append(refined_answer)
                            st.session_state.answers[param]['answer'] = refined_answer
                        st.success("Refinement complete!")
                        st.rerun()
                    else:
                        st.warning("Please provide feedback.")

            with st.expander("🕘 Edit History"):
                for ver_i, version in enumerate(st.session_state.answers[param]['versions'][::-1], 1):
                    st.markdown(f"**Version {ver_i}:** {version}")

        # NAVIGATION
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            if st.button("⬅ Previous Set", disabled=(st.session_state.current_page == 0)):
                st.session_state.current_page -= 1
                st.rerun()
        with col2:
            if st.button("Next Set ➡", disabled=(st.session_state.current_page >= total_pages - 1)):
                st.session_state.current_page += 1
                st.rerun()

        # ✅ FINAL Download Button after last page
        if all_questions_answered() and not st.session_state.report_ready:
            st.success("🎉 All questions answered!")
            if st.button("📥 Download Final Report"):
                st.session_state.report_ready = True
                st.rerun()
