# pdf_utils.py
import os
import fitz  # PyMuPDF
import pdfplumber
from PIL import Image
from io import BytesIO
import google.generativeai as genai
from concurrent.futures import ThreadPoolExecutor
import time
import random # For jitter in backoff
import logging
import config # To get API keys

# Configure logging (if not already configured by a calling script)
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s')

MAX_IMAGE_WORKERS = 5 # Max parallel image processing tasks
MAX_API_RETRIES_IMG = 3 # Retries for image processing API calls
INITIAL_BACKOFF_SECONDS_IMG = 3 # Initial backoff for image processing
CONTEXT_WINDOW_PAGES = 1 # Number of pages before and after the current page to include in focused context for images

def configure_gemini_model_for_task(api_key_for_task):
    """Configures genai with a specific API key and returns a model instance."""
    try:
        # This global configure call is a known challenge with this SDK in multithreading.
        # Assuming GenerativeModel() uses the config state at its creation.
        # For truly robust key isolation per thread without potential races,
        # multiprocessing or a different SDK client management approach would be better.
        genai.configure(api_key=api_key_for_task)
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        return model
    except Exception as e:
        logging.error(f"Failed to configure Gemini model with key ...{api_key_for_task[-6:]}: {e}")
        return None

def extract_text_from_pdf(pdf_path) -> tuple[list[str], str]:
    """
    Extract text from PDF.
    Returns:
        - A list of strings, where each string is the text of a page.
        - A single string containing all text concatenated.
    """
    page_texts_list = []
    concatenated_text = ""
    
    try:
        # Try with pdfplumber first for its layout handling
        temp_plumber_pages = []
        with pdfplumber.open(pdf_path) as pdf:
            if not pdf.pages:
                logging.warning(f"No pages found in {pdf_path} by pdfplumber.")
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text() or ""
                temp_plumber_pages.append(page_text)
        
        if any(pt.strip() for pt in temp_plumber_pages): # If pdfplumber got substantial text
            page_texts_list = temp_plumber_pages
            logging.info(f"Extracted text using pdfplumber for {pdf_path}.")
        else:
            logging.warning(f"pdfplumber extracted no substantial text from {pdf_path}. Trying PyMuPDF page by page.")
            # Fallback to PyMuPDF if pdfplumber yielded empty or mostly whitespace results
            raise Exception("pdfplumber yielded empty content") # Trigger PyMuPDF fallback

    except Exception as e_plumber:
        logging.warning(f"pdfplumber failed for {pdf_path} ({e_plumber}). Using PyMuPDF for page-level text.")
        page_texts_list = [] # Reset if pdfplumber failed
        try:
            with fitz.open(pdf_path) as fitz_doc:
                if not len(fitz_doc):
                    logging.warning(f"No pages found in {pdf_path} by PyMuPDF fallback.")
                for page_num_fitz in range(len(fitz_doc)):
                    page_fitz = fitz_doc.load_page(page_num_fitz)
                    page_text_fitz = page_fitz.get_text("text") or ""
                    page_texts_list.append(page_text_fitz)
            logging.info(f"Extracted text using PyMuPDF (page-by-page) for {pdf_path}.")
        except Exception as e_fitz:
            logging.error(f"Both pdfplumber and PyMuPDF failed to extract text from {pdf_path}: {e_fitz}")
            return [], "" # Return empty if all fails

    concatenated_text = "\n".join(page_texts_list)
    if not concatenated_text.strip():
        logging.warning(f"No text content extracted from {pdf_path} by any method.")
        
    return page_texts_list, concatenated_text

def get_focused_context(all_page_texts: list[str], current_page_index: int, window_size: int = CONTEXT_WINDOW_PAGES) -> str:
    """
    Extracts text from the current page and N surrounding pages (window_size).
    """
    if not all_page_texts:
        return ""
    
    start_page = max(0, current_page_index - window_size)
    end_page = min(len(all_page_texts), current_page_index + window_size + 1)
    
    context_pages = all_page_texts[start_page:end_page]
    focused_context = "\n".join(context_pages).strip()
    # Add a note about the window if it's not the full document
    # if len(context_pages) < len(all_page_texts):
    #     return f"[Context from pages {start_page+1} to {end_page}]\n{focused_context}"
    return focused_context


def process_image_task(api_key_for_task, image_bytes, image_ext, page_num, img_index,
                       text_on_current_page, focused_contextual_text, # Changed from contextual_pdf_text
                       images_folder_for_this_pdf):
    img_name = f"page_{page_num+1}_image_{img_index}.{image_ext}"
    os.makedirs(images_folder_for_this_pdf, exist_ok=True)
    img_path = os.path.join(images_folder_for_this_pdf, img_name)
    
    model_for_task = configure_gemini_model_for_task(api_key_for_task)
    if not model_for_task:
        return f"\n[Image {img_name} Error]\nFailed to configure Gemini model for this task.\n"

    try:
        with open(img_path, "wb") as f:
            f.write(image_bytes)

        prompt = (
            "Analyze this image thoroughly from a research paper. "
            "Describe all visual elements (e.g., graphs, diagrams, photos, charts), their structure, and their significance. "
            "Extract and clearly state any text visible within the image (e.g., labels, captions directly on the image, data points). "
            "Explain the image's overall context and importance based on the text from the page it appears on and relevant surrounding pages. " # Modified instruction
            "Be detailed and encyclopedic in your description and analysis.\n\n"
            f"Text from the page where this image appears:\n{text_on_current_page}\n\n"
            f"Focused contextual text from surrounding pages (current page +/- {CONTEXT_WINDOW_PAGES} pages):\n{focused_contextual_text}\n" # Use focused context
            "Provide your analysis of the image (including any text you identified within it):"
        )
        img_pil = Image.open(BytesIO(image_bytes))

        for attempt in range(MAX_API_RETRIES_IMG):
            try:
                response = model_for_task.generate_content([prompt, img_pil])
                return f"\n[Image {img_name} Analysis (by Gemini)]\n{response.text}\n"
            except Exception as e_api:
                logging.warning(f"API call attempt {attempt + 1} for image {img_name} failed: {e_api}")
                if "429" in str(e_api) or "rate limit" in str(e_api).lower() or "quota" in str(e_api).lower():
                    if attempt < MAX_API_RETRIES_IMG - 1:
                        backoff_time = INITIAL_BACKOFF_SECONDS_IMG * (2 ** attempt) + random.uniform(0,1)
                        logging.info(f"Rate limit hit for image {img_name}. Retrying in {backoff_time:.2f} seconds...")
                        time.sleep(backoff_time)
                    else:
                        logging.error(f"Max retries reached for image {img_name} due to rate limits.")
                        raise # Re-raise to be caught by the outer try-except
                else: 
                    raise 
        return f"\n[Image {img_name} Error]\nDescription failed after multiple retries due to API issues.\n"

    except Exception as e:
        logging.error(f"Error processing image {img_name}: {str(e)}", exc_info=True)
        return f"\n[Image {img_name} Error]\nDescription failed: {str(e)}\n"


def process_pdf(pdf_path, output_text_file, images_folder_root):
    pdf_filename_without_ext = os.path.splitext(os.path.basename(pdf_path))[0]
    images_folder_for_this_pdf = os.path.join(images_folder_root, pdf_filename_without_ext)
    os.makedirs(images_folder_for_this_pdf, exist_ok=True)

    # Get page-separated texts and the full concatenated text
    list_of_page_texts, _ = extract_text_from_pdf(pdf_path) # We only need list_of_page_texts for per-page work here
                                                            # The full text for Q&A will be the result of this func.

    if not list_of_page_texts:
        logging.error(f"No text pages extracted from {pdf_path}. Cannot process for image context effectively.")
        with open(output_text_file, "w", encoding="utf-8") as output_file_handle:
            output_file_handle.write("Error: No text pages could be extracted from the PDF.")
        return "Error: No text pages could be extracted from the PDF."

    processed_pages_content_for_file = [] # This will store the final string for each page

    try:
        with fitz.open(pdf_path) as fitz_doc:
            total_pages_fitz = len(fitz_doc)
            # Use the number of pages from fitz, as it's used for image extraction
            logging.info(f"Processing {total_pages_fitz} pages for {os.path.basename(pdf_path)} images and layout...")

            for page_num in range(total_pages_fitz):
                logging.info(f"  Processing page {page_num + 1}/{total_pages_fitz}...")
                
                # Get text from the current page (from the pre-extracted list)
                text_on_current_page = list_of_page_texts[page_num] if page_num < len(list_of_page_texts) else ""
                
                # Get focused context for this image using the list of all page texts
                focused_context = get_focused_context(list_of_page_texts, page_num, window_size=CONTEXT_WINDOW_PAGES)

                fitz_page = fitz_doc.load_page(page_num)
                images_on_page_info = fitz_page.get_images(full=True)
                page_image_descriptions = [None] * len(images_on_page_info)

                if images_on_page_info:
                    logging.info(f"    Found {len(images_on_page_info)} images on page {page_num + 1}. Submitting for parallel processing...")
                    with ThreadPoolExecutor(max_workers=MAX_IMAGE_WORKERS) as executor:
                        future_to_img_idx = {}
                        active_api_keys_for_page = set()

                        for img_idx, img_info in enumerate(images_on_page_info):
                            api_key_for_image = config.get_random_google_api_key(exclude_keys=list(active_api_keys_for_page))
                            active_api_keys_for_page.add(api_key_for_image)
                            xref = img_info[0]
                            try:
                                base_image = fitz_doc.extract_image(xref)
                                if base_image and base_image.get("image"):
                                    future = executor.submit(process_image_task,
                                                             api_key_for_image,
                                                             base_image["image"], base_image["ext"],
                                                             page_num, img_idx,
                                                             text_on_current_page, # Text from current page
                                                             focused_context,     # Focused context from surrounding pages
                                                             images_folder_for_this_pdf)
                                    future_to_img_idx[future] = img_idx
                                else:
                                    page_image_descriptions[img_idx] = f"\n[Image page_{page_num+1}_image_{img_idx}.{img_info[1]} Error]\nInvalid image data.\n"
                            except Exception as e_extract:
                                 page_image_descriptions[img_idx] = f"\n[Image page_{page_num+1}_image_{img_idx}.{img_info[1]} Error]\nExtraction failed: {e_extract}\n"
                        
                        for future in future_to_img_idx:
                            idx = future_to_img_idx[future]
                            try:
                                description_result = future.result(timeout=300)
                                page_image_descriptions[idx] = description_result
                            except Exception as exc:
                                img_ext_fallback = images_on_page_info[idx][1] if idx < len(images_on_page_info) and len(images_on_page_info[idx]) > 1 else "unknown_ext"
                                page_image_descriptions[idx] = f"\n[Image page_{page_num+1}_image_{idx}.{img_ext_fallback} Error]\nProcessing failed in thread: {str(exc)}\n"
                    logging.info(f"    Finished processing images for page {page_num + 1}.")
                
                # Start with the text extracted for this page (from list_of_page_texts)
                current_page_full_content = text_on_current_page
                for desc in page_image_descriptions:
                    if desc:
                        current_page_full_content += f"\n{desc}" # Append image analysis
                
                processed_pages_content_for_file.append(f"=== Page {page_num+1} ===\n{current_page_full_content}\n\n")
    
    except Exception as e_main_processing:
        logging.error(f"Critical error during page processing loop for {pdf_path}: {e_main_processing}", exc_info=True)
        # Fallback to writing only the concatenated direct text if detailed processing failed
        # (though list_of_page_texts might be more reliable if concatenated_text was from an earlier failed step)
        fallback_text = "\n".join(list_of_page_texts) if list_of_page_texts else "Error: PDF processing failed critically, no text extracted."
        with open(output_text_file, "w", encoding="utf-8") as output_file_handle:
            output_file_handle.write(fallback_text)
        return fallback_text

    final_output_text = "".join(processed_pages_content_for_file)
    with open(output_text_file, "w", encoding="utf-8") as output_file_handle:
        output_file_handle.write(final_output_text)
    return final_output_text


def process_pdf_with_progress(pdf_path, output_text_file, images_folder_root, api_key_initial_placeholder):
    """
    For Streamlit. Implements focused context for images and multithreading.
    api_key_initial_placeholder is not directly used for model config here as tasks get their own.
    """
    import streamlit as st

    pdf_filename_without_ext = os.path.splitext(os.path.basename(pdf_path))[0]
    images_folder_for_this_pdf = os.path.join(images_folder_root, pdf_filename_without_ext)
    os.makedirs(images_folder_for_this_pdf, exist_ok=True)

    list_of_page_texts, _ = extract_text_from_pdf(pdf_path)
    if not list_of_page_texts:
        st.error("No text could be extracted from the PDF. Cannot proceed.")
        with open(output_text_file, "w", encoding="utf-8") as f: f.write("Error: No text extracted.")
        return "Error: No text extracted."

    all_pages_final_content_for_file = []

    try:
        with fitz.open(pdf_path) as fitz_doc, \
             open(output_text_file, "w", encoding="utf-8") as output_file_handle:

            total_pages = len(fitz_doc)
            if total_pages == 0:
                st.warning("The PDF appears to be empty or could not be read by PyMuPDF.")
                output_file_handle.write("Error: PDF is empty or unreadable by PyMuPDF.")
                return "Error: PDF is empty or unreadable by PyMuPDF."

            progress_bar = st.progress(0)
            status_text = st.empty()

            for page_num in range(total_pages):
                status_text.info(f"Processing Page {page_num+1} of {total_pages}...")
                
                text_on_current_page = list_of_page_texts[page_num] if page_num < len(list_of_page_texts) else ""
                focused_context = get_focused_context(list_of_page_texts, page_num, window_size=CONTEXT_WINDOW_PAGES)

                fitz_page = fitz_doc.load_page(page_num)
                images_on_page_info = fitz_page.get_images(full=True)
                page_image_descriptions = [None] * len(images_on_page_info)

                if images_on_page_info:
                    status_text.info(f"Page {page_num+1}: Found {len(images_on_page_info)} images. Analyzing concurrently...")
                    with ThreadPoolExecutor(max_workers=MAX_IMAGE_WORKERS) as executor:
                        future_to_img_idx = {}
                        active_api_keys_for_page = set()
                        for img_idx, img_info in enumerate(images_on_page_info):
                            api_key_for_image = config.get_random_google_api_key(exclude_keys=list(active_api_keys_for_page))
                            active_api_keys_for_page.add(api_key_for_image)
                            xref = img_info[0]
                            try:
                                base_image = fitz_doc.extract_image(xref)
                                if base_image and base_image.get("image"):
                                    future = executor.submit(process_image_task,
                                                             api_key_for_image,
                                                             base_image["image"], base_image["ext"],
                                                             page_num, img_idx,
                                                             text_on_current_page,
                                                             focused_context,
                                                             images_folder_for_this_pdf)
                                    future_to_img_idx[future] = img_idx
                                else:
                                    page_image_descriptions[img_idx] = f"\n[Image page_{page_num+1}_image_{img_idx}.{img_info[1]} Error]\nInvalid image data.\n"
                            except Exception as e_extract:
                                 page_image_descriptions[img_idx] = f"\n[Image page_{page_num+1}_image_{img_idx}.{img_info[1]} Error]\nExtraction failed: {e_extract}\n"
                        
                        for future in future_to_img_idx:
                            idx = future_to_img_idx[future]
                            try:
                                description_result = future.result(timeout=300)
                                page_image_descriptions[idx] = description_result
                            except Exception as exc:
                                img_ext_fallback = images_on_page_info[idx][1] if idx < len(images_on_page_info) and len(images_on_page_info[idx]) > 1 else "unknown_ext"
                                page_image_descriptions[idx] = f"\n[Image page_{page_num+1}_image_{idx}.{img_ext_fallback} Error]\nProcessing timed out or failed: {str(exc)}\n"
                
                current_page_final_content = text_on_current_page # Start with text from this page
                for desc in page_image_descriptions:
                    if desc:
                        current_page_final_content += f"\n{desc}"
                
                output_file_handle.write(f"=== Page {page_num+1} ===\n{current_page_final_content}\n\n")
                all_pages_final_content_for_file.append(current_page_final_content)
                progress_bar.progress((page_num + 1) / total_pages)

            status_text.success("✅ PDF processing completed!")
            progress_bar.empty()
    except Exception as e_main:
        st_error_msg = f"A critical error occurred during PDF processing: {e_main}"
        logging.error(st_error_msg, exc_info=True)
        st.error(st_error_msg)
        try:
            with open(output_text_file, "w", encoding="utf-8") as error_f: error_f.write(st_error_msg)
        except: pass # nosec
        return st_error_msg

    final_text_for_session_state = "".join([f"=== Page {i+1} ===\n{content}\n\n" for i, content in enumerate(all_pages_final_content_for_file)])
    return final_text_for_session_state