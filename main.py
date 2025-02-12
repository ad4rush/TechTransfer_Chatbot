# app.py
import streamlit as st
import os
import json
import config
from pdf_utils import process_pdf
from chatbot import initialize_chat, get_questions, generate_answer

# Configuration
GEMINI_API_KEY = config.GOOGLE_API_KEY
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
        'feedback_history': {}
    })

st.title("Tech Transfer ChatBot")

# 1. PDF Processing Section
uploaded_file = st.file_uploader("Upload Research PDF", type=["pdf"])
if uploaded_file and not st.session_state.processed:
    with st.spinner("Processing PDF..."):
        # Save and process PDF
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        process_pdf("temp.pdf", OUTPUT_TEXT_FILE, IMAGES_FOLDER, GEMINI_API_KEY)
        st.session_state.pdf_text = open(OUTPUT_TEXT_FILE).read()
        st.session_state.processed = True
        st.session_state.chat = initialize_chat(GEMINI_API_KEY)
        
    st.success("PDF processed successfully!")
    st.download_button(
        label="Download Extracted Text",
        data=open(OUTPUT_TEXT_FILE, "rb").read(),
        file_name="extracted_text.txt",
        mime="text/plain"
    )

# 2. Only show the Q&A section if PDF has been processed
if st.session_state.processed:
    st.divider()
    st.header("Your Research Analysis!")
    
    # Check if we've exhausted all questions
    if st.session_state.current_question >= len(st.session_state.questions):
        st.header("✅ Analysis Complete!")
        st.markdown("### Final Parameters")
        
        # Prepare final data
        final_data = {
            param: data['answer']
            for param, data in st.session_state.answers.items()
        }
        
        st.json(final_data)
        
        # Save and download results
        with open("confirmed_parameters.json", "w") as f:
            json.dump(final_data, f, indent=4)
        
        with open("confirmed_parameters.json", "rb") as f:
            st.download_button(
                label="Download Full Report",
                data=f,
                file_name="research_analysis.json",
                mime="application/json"
            )

    else:
        # 3. Current Question and Parameter
        current_q_data = st.session_state.questions[st.session_state.current_question]
        param = current_q_data["param"]
        
        # Initialize dict structure if needed
        if param not in st.session_state.answers:
            st.session_state.answers[param] = {
                'answer': '',
                'versions': []
            }
            st.session_state.feedback_history[param] = []
        
        st.subheader(f"Question {st.session_state.current_question + 1}/{len(st.session_state.questions)}")
        st.markdown(f"**Parameter:** {param}")
        st.markdown(f"**Question:** {current_q_data['question']}")
        
        # 4. Generate initial answer if we have none
        if not st.session_state.answers[param]['versions']:
            with st.spinner("Generating initial answer..."):
                response = generate_answer(
                    st.session_state.chat,
                    current_q_data["question"],
                    st.session_state.pdf_text
                )
                st.session_state.answers[param]['versions'].append(response)
                st.session_state.answers[param]['answer'] = response
        
        # 5. Display current answer
        st.markdown("### Current Answer")
        st.write(st.session_state.answers[param]['answer'])
        
        # 6. Feedback / Refinement Section
        st.markdown("### Refine the Answer")
        feedback = st.text_area("Provide feedback for refinement (optional).")
        
        if st.button("Refine Answer"):
            if feedback.strip():
                # Add feedback to history
                st.session_state.feedback_history[param].append(feedback)
                
                with st.spinner("Refining answer..."):
                    prompt = (
                        f"{current_q_data['question']}\n\n"
                        f"Previous answer: {st.session_state.answers[param]['answer']}\n"
                        f"Feedback: {feedback}\n"
                    )
                    refined_answer = generate_answer(
                        st.session_state.chat,
                        prompt,
                        st.session_state.pdf_text
                    )
                    st.session_state.answers[param]['versions'].append(refined_answer)
                    st.session_state.answers[param]['answer'] = refined_answer
                st.rerun()
            else:
                st.warning("No feedback provided. Please enter feedback or skip refinement.")
        
        # Show refinement history if any
        if st.session_state.feedback_history[param]:
            st.markdown("#### Refinement History")
            for i, fb in enumerate(st.session_state.feedback_history[param], 1):
                st.write(f"**{i}.** {fb}")
        
        # 7. Navigation
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Previous Question", disabled=(st.session_state.current_question == 0)):
                if st.session_state.current_question > 0:
                    st.session_state.current_question -= 1
                st.rerun()
                
        with col2:
            if st.button("Next Question →"):
                # Move to next question
                st.session_state.current_question += 1
                st.rerun()
