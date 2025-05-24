# FILE: pdf_utils.py
# (Showing new helper function and modifications to _process_pdf_core)

import os
import fitz  # PyMuPDF
import pdfplumber
from PIL import Image # Pillow, for image manipulation
from io import BytesIO
import google.generativeai as genai
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import random
import logging
import config
import pytesseract # <--- ADD THIS IMPORT for OCR

# --- (Your existing imports and constants like MAX_IMAGE_WORKERS etc.) ---
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(module)s.%(funcName)s:%(lineno)d] - %(message)s')

MAX_IMAGE_WORKERS = 3 # Keep this tuned based on previous discussions
MAX_API_RETRIES_IMG = 3
INITIAL_BACKOFF_SECONDS_IMG = 7
CONTEXT_WINDOW_PAGES = 1
OCR_TEXT_WORTHINESS_THRESHOLD = 20 # Min characters for an image to be "worthy" of Gemini analysis

# --- (OPTIONAL: Configure Tesseract path if not in system PATH) ---
# try:
#     pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe' # Example for Windows
# except Exception as e_tesseract_path:
#     logging.warning(f"Could not set tesseract_cmd path, ensure Tesseract is in your PATH. Error: {e_tesseract_path}")


# --- (extract_text_from_pdf, get_focused_context, _configure_gemini_model_for_task_internal, process_image_task
#      remain the same as in your corrected version from previous steps) ---

def _configure_gemini_model_for_task_internal(api_key_for_task):
    try:
        with config._KEY_MANAGER_LOCK:
            genai.configure(api_key=api_key_for_task)
            model = genai.GenerativeModel('gemini-2.0-flash-exp')
        return model
    except Exception as e:
        logging.error(f"Failed to configure Gemini model with key ...{api_key_for_task[-6:]}: {e}")
        return None

def extract_text_from_pdf(pdf_path) -> tuple[list[str], str]:
    page_texts_list = []
    try:
        temp_plumber_pages = []
        with pdfplumber.open(pdf_path) as pdf:
            if not pdf.pages: logging.warning(f"No pages found in {pdf_path} by pdfplumber.")
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text() or ""
                temp_plumber_pages.append(page_text)
        if any(pt.strip() for pt in temp_plumber_pages):
            page_texts_list = temp_plumber_pages
        else:
            raise Exception("pdfplumber yielded empty content for all pages")
    except Exception as e_plumber:
        logging.warning(f"pdfplumber failed or yielded empty for {os.path.basename(pdf_path)} ({e_plumber}). Using PyMuPDF.")
        page_texts_list = []
        try:
            with fitz.open(pdf_path) as fitz_doc:
                if not len(fitz_doc): logging.warning(f"No pages found in {pdf_path} by PyMuPDF fallback.")
                for page_num_fitz in range(len(fitz_doc)):
                    page_fitz = fitz_doc.load_page(page_num_fitz)
                    page_text_fitz = page_fitz.get_text("text") or ""
                    page_texts_list.append(page_text_fitz)
        except Exception as e_fitz:
            logging.error(f"Both pdfplumber and PyMuPDF failed to extract text from {os.path.basename(pdf_path)}: {e_fitz}", exc_info=True)
            return [], ""
    concatenated_text = "\n".join(page_texts_list)
    if not concatenated_text.strip(): logging.warning(f"No text content extracted from {os.path.basename(pdf_path)} by any method.")
    return page_texts_list, concatenated_text

def get_focused_context(all_page_texts: list[str], current_page_index: int, window_size: int = CONTEXT_WINDOW_PAGES) -> str:
    if not all_page_texts: return ""
    start_page = max(0, current_page_index - window_size)
    end_page = min(len(all_page_texts), current_page_index + window_size + 1)
    context_pages = all_page_texts[start_page:end_page]
    return "\n".join(context_pages).strip()

def process_image_task(initial_api_key_for_submission, image_bytes, image_ext, page_num, img_idx,
                       text_on_current_page, focused_contextual_text,
                       images_folder_for_this_pdf, original_img_info):
    # This function remains largely the same from your corrected version
    # (where it calls config.record_key_usage before the API call)
    img_name = f"page_{page_num+1}_image_{img_idx}.{image_ext}"
    img_path = os.path.join(images_folder_for_this_pdf, img_name)
    os.makedirs(os.path.dirname(img_path), exist_ok=True)

    current_api_key_for_task = initial_api_key_for_submission
    model_for_task = None
    keys_tried_this_specific_task = {current_api_key_for_task}

    try:
        with open(img_path, "wb") as f: f.write(image_bytes)
        img_pil = Image.open(BytesIO(image_bytes))
        prompt = (
            "Analyze this image thoroughly from a research paper. "
            "Describe all visual elements (e.g., graphs, diagrams, photos, charts), their structure, and their significance. "
            "Extract and clearly state any text visible within the image (e.g., labels, captions, data points). "
            "Explain the image's overall context and importance based on the text from its page and relevant surrounding pages.\n\n"
            f"Text from the page where this image appears (page {page_num+1}):\n{text_on_current_page}\n\n"
            f"Focused contextual text from surrounding pages (current page +/- {CONTEXT_WINDOW_PAGES} pages):\n{focused_contextual_text}\n"
            "Provide your analysis of the image (including any text you identified within it):"
        )
        for attempt in range(MAX_API_RETRIES_IMG):
            try:
                model_for_task = _configure_gemini_model_for_task_internal(current_api_key_for_task)
                if not model_for_task:
                    logging.error(f"Image {img_name}: Model config failed with key ...{current_api_key_for_task[-6:]} on attempt {attempt+1}.")
                    if attempt < MAX_API_RETRIES_IMG - 1:
                        try:
                            current_api_key_for_task = config.get_available_api_key()
                            keys_tried_this_specific_task.add(current_api_key_for_task)
                            logging.info(f"Image {img_name}: Switched to new key ...{current_api_key_for_task[-6:]} for model config retry.")
                            time.sleep(INITIAL_BACKOFF_SECONDS_IMG / 2)
                            continue
                        except ValueError as e_get_key:
                            logging.error(f"Image {img_name}: Could not get new key after model config failure: {e_get_key}")
                            break
                    else:
                        break
                logging.info(f"Image task (p{page_num+1},i{img_idx}) API call attempt {attempt+1} with key ...{current_api_key_for_task[-6:]}")
                config.record_key_usage(current_api_key_for_task)
                response = model_for_task.generate_content([prompt, img_pil])
                config.record_key_success(current_api_key_for_task)
                return (page_num, img_idx, f"\n[Image {img_name} Analysis (by Gemini)]\n{response.text}\n")
            except Exception as e_api:
                logging.warning(f"API call attempt {attempt + 1} for image {img_name} (key ...{current_api_key_for_task[-6:]}) failed: {e_api}")
                is_rate_limit_error = "429" in str(e_api) or "rate limit" in str(e_api).lower() or "quota" in str(e_api).lower()
                if is_rate_limit_error:
                    config.mark_key_as_rate_limited(current_api_key_for_task)
                    if attempt < MAX_API_RETRIES_IMG - 1:
                        try:
                            new_key_to_try = config.get_available_api_key()
                            keys_tried_this_specific_task.add(new_key_to_try)
                            current_api_key_for_task = new_key_to_try
                            logging.info(f"Image {img_name}: Switched to new key ...{current_api_key_for_task[-6:]} for next attempt.")
                        except ValueError as e_get_key:
                            logging.error(f"Image {img_name}: Could not get a new API key for retry after 429 ({e_get_key}). Ending retries.")
                            break
                        backoff_time = INITIAL_BACKOFF_SECONDS_IMG * (2**attempt) + random.uniform(0,1)
                        logging.info(f"Image {img_name}: Retrying in {backoff_time:.2f}s with key ...{current_api_key_for_task[-6:]}...")
                        time.sleep(backoff_time)
                    else:
                        logging.error(f"Max retries ({MAX_API_RETRIES_IMG}) reached for image {img_name} due to persistent rate limits.")
                        return (page_num, img_idx, f"\n[Image {img_name} Error]\nFailed after max retries (persistent rate limits).\n")
                else:
                    logging.error(f"Image {img_name}: Non-rate-limit API error on attempt {attempt+1}: {e_api}", exc_info=False)
                    if attempt < MAX_API_RETRIES_IMG - 1:
                        time.sleep(INITIAL_BACKOFF_SECONDS_IMG * (1.5**attempt) + random.uniform(0,0.5))
                    else:
                        return (page_num, img_idx, f"\n[Image {img_name} Error]\nFailed after max retries (non-rate-limit error: {e_api}).\n")
        return (page_num, img_idx, f"\n[Image {img_name} Error]\nAll API call attempts failed for this image.\n")
    except Exception as e:
        logging.error(f"Outer error in process_image_task for {img_name} (page {page_num+1}): {e}", exc_info=True)
        return (page_num, img_idx, f"\n[Image {img_name} Error]\nProcessing failed: {str(e)}\n")


def is_image_worthy_for_gemini(image_bytes, min_text_len=OCR_TEXT_WORTHINESS_THRESHOLD):
    """
    Performs OCR on the image and checks if the amount of text found meets a threshold.
    Returns: True if worthy, False otherwise.
    """
    try:
        img_pil = Image.open(BytesIO(image_bytes))
        # Perform OCR - you might want to specify language e.g., lang='eng'
        # Tesseract configuration options can also be passed via config argument
        # e.g., custom_config = r'--oem 3 --psm 6'
        ocr_text = pytesseract.image_to_string(img_pil, timeout=10) # Added timeout
        ocr_text_stripped = ocr_text.strip()
        
        if len(ocr_text_stripped) >= min_text_len:
            logging.debug(f"OCR found {len(ocr_text_stripped)} chars (>= threshold {min_text_len}). Image is worthy. Text: '{ocr_text_stripped[:100]}...'")
            return True, ocr_text_stripped # Return text for potential logging or basic inclusion
        else:
            logging.debug(f"OCR found {len(ocr_text_stripped)} chars (< threshold {min_text_len}). Image deemed not worthy for Gemini. Text: '{ocr_text_stripped[:100]}...'")
            return False, ocr_text_stripped
    except pytesseract.TesseractNotFoundError:
        logging.error("Tesseract is not installed or not in your PATH. OCR pre-check skipped. Sending image to Gemini.")
        return True, "[OCR skipped: Tesseract not found]" # Default to worthy if OCR fails
    except RuntimeError as e_timeout: # Catches Tesseract timeout
        logging.warning(f"Tesseract OCR timed out for an image: {e_timeout}. Pre-check skipped. Sending image to Gemini.")
        return True, "[OCR skipped: Tesseract timeout]"
    except Exception as e:
        logging.warning(f"Error during OCR pre-check for an image: {e}. Pre-check skipped. Sending image to Gemini.")
        return True, f"[OCR skipped: Error - {str(e)}]" # Default to worthy if OCR fails for other reasons


def _process_pdf_core(pdf_path, output_text_file, images_folder_root, is_streamlit_run=False, st_progress_objects=None):
    pdf_filename = os.path.basename(pdf_path)
    pdf_filename_without_ext = os.path.splitext(pdf_filename)[0]
    images_folder_for_this_pdf = os.path.join(images_folder_root, pdf_filename_without_ext)
    os.makedirs(images_folder_for_this_pdf, exist_ok=True)

    list_of_page_texts, _ = extract_text_from_pdf(pdf_path)
    if not list_of_page_texts:
        # ... (error handling as before)
        error_msg = f"Error: No text pages could be extracted from '{pdf_filename}'."
        logging.error(error_msg)
        with open(output_text_file, "w", encoding="utf-8") as f: f.write(error_msg)
        if is_streamlit_run and st_progress_objects: st_progress_objects['status_text'].error(error_msg)
        return error_msg

    all_image_processing_tasks_data = [] # Tasks for Gemini
    page_image_counts = {}
    total_pages_fitz = 0
    # This dictionary will store results directly for images skipped by OCR or if Gemini fails for worthy ones
    processed_image_results = {} 

    logging.info(f"Starting PDF Processing for: {pdf_filename}")
    try:
        with fitz.open(pdf_path) as fitz_doc:
            total_pages_fitz = len(fitz_doc)
            logging.info(f"Phase 1: Scanning {total_pages_fitz} pages in '{pdf_filename}' to collect and OCR-filter image tasks...")
            # ... (streamlit progress update if needed) ...

            for page_num in range(total_pages_fitz):
                # ... (streamlit progress update if needed) ...
                
                text_on_current_page = list_of_page_texts[page_num] if page_num < len(list_of_page_texts) else ""
                focused_context = get_focused_context(list_of_page_texts, page_num, window_size=CONTEXT_WINDOW_PAGES)
                fitz_page = fitz_doc.load_page(page_num)
                images_on_page_info = fitz_page.get_images(full=True)
                page_image_counts[page_num] = len(images_on_page_info)

                for img_idx, img_info in enumerate(images_on_page_info):
                    xref = img_info[0]
                    try:
                        base_image = fitz_doc.extract_image(xref)
                        if base_image and base_image.get("image"):
                            image_bytes = base_image["image"]
                            image_ext = base_image["ext"]
                            img_name_for_log = f"page_{page_num+1}_image_{img_idx}.{image_ext}"

                            # Perform OCR check
                            is_worthy, ocr_text = is_image_worthy_for_gemini(image_bytes)
                            
                            if is_worthy:
                                logging.debug(f"Image {img_name_for_log} is worthy for Gemini. Adding to processing queue.")
                                all_image_processing_tasks_data.append({
                                    "image_bytes": image_bytes, "image_ext": image_ext,
                                    "page_num": page_num, "img_idx": img_idx,
                                    "text_on_current_page": text_on_current_page,
                                    "focused_contextual_text": focused_context,
                                    "images_folder_for_this_pdf": images_folder_for_this_pdf,
                                    "original_img_info": img_info
                                })
                            else:
                                # Image not worthy, add placeholder directly to results
                                logging.info(f"Image {img_name_for_log} skipped for Gemini analysis based on OCR (Text: '{ocr_text[:50]}...').")
                                placeholder_text = f"\n[Image {img_name_for_log} OCR: Text found: '{ocr_text.strip()[:100].replacechr(10,' ') if ocr_text else 'None'}' - Full analysis via Gemini skipped.]\n"
                                processed_image_results[(page_num, img_idx)] = placeholder_text
                        else: 
                            logging.warning(f"Invalid image data extracted by Fitz for an image on page {page_num+1}, index {img_idx}.")
                            processed_image_results[(page_num, img_idx)] = f"\n[Image page_{page_num+1}_image_{img_idx}.{img_info[1] if len(img_info)>1 else 'err'} Error]\nInvalid image data from Fitz.\n"
                    except Exception as e_extract:
                        logging.error(f"Error extracting image on page {page_num+1}, index {img_idx}: {e_extract}")
                        processed_image_results[(page_num, img_idx)] = f"\n[Image page_{page_num+1}_image_{img_idx}.{img_info[1] if len(img_info)>1 else 'err'} Error]\nFitz extraction failed: {e_extract}\n"
            
            logging.info(f"Phase 1 Complete: Collected {len(all_image_processing_tasks_data)} image tasks worthy of Gemini analysis for '{pdf_filename}'.")
    except Exception as e_fitz_scan:
        # ... (error handling as before) ...
        error_msg = f"Critical error during PDF scan (Phase 1) for images in {pdf_filename}: {e_fitz_scan}"
        logging.error(error_msg, exc_info=True)
        with open(output_text_file, "w", encoding="utf-8") as f: f.write(error_msg)
        if is_streamlit_run and st_progress_objects: st_progress_objects['status_text'].error(error_msg)
        return error_msg

    # Phase 2: Process "worthy" image tasks in parallel (same as before, but `all_image_processing_tasks_data` is now pre-filtered)
    if all_image_processing_tasks_data: # Only proceed if there are tasks for Gemini
        active_tasks_for_submission = all_image_processing_tasks_data # Already filtered for 'error' during collection
        logging.info(f"Phase 2: Submitting {len(active_tasks_for_submission)} OCR-filtered image tasks to ThreadPoolExecutor (max_workers={MAX_IMAGE_WORKERS}) for '{pdf_filename}'.")
        # ... (rest of ThreadPoolExecutor logic for Gemini tasks remains the same as your corrected version)
        # It will populate `processed_image_results` for tasks it handles.
        actual_workers_for_images = min(MAX_IMAGE_WORKERS, len(active_tasks_for_submission)) if active_tasks_for_submission else 1
        with ThreadPoolExecutor(max_workers=actual_workers_for_images) as executor:
            future_to_task_meta = {}
            for task_data in active_tasks_for_submission:
                try:
                    key_for_this_img_task_submission = config.get_available_api_key()
                except ValueError as e_get_key_init:
                    logging.error(f"Could not get initial API key for image task (p{task_data['page_num']+1}, i{task_data['img_idx']}): {e_get_key_init}. Marking task as error.")
                    processed_image_results[(task_data['page_num'], task_data['img_idx'])] = f"\n[Image page_{task_data['page_num']+1}_image_{task_data['img_idx']}.{task_data.get('image_ext','png')} Error]\nNo API key available for start of task: {e_get_key_init}.\n"
                    continue
                future = executor.submit(
                    process_image_task,
                    key_for_this_img_task_submission,
                    task_data["image_bytes"], task_data["image_ext"], task_data["page_num"], task_data["img_idx"],
                    task_data["text_on_current_page"], task_data["focused_contextual_text"],
                    task_data["images_folder_for_this_pdf"], task_data["original_img_info"]
                )
                future_to_task_meta[future] = (task_data["page_num"], task_data["img_idx"])
            processed_count = 0
            total_to_process_api = len(future_to_task_meta)
            if total_to_process_api > 0:
                logging.info(f"Phase 2: Waiting for {total_to_process_api} Gemini image analysis API calls to complete for '{pdf_filename}'...")
                for future in as_completed(future_to_task_meta):
                    try:
                        page_num_res, img_idx_res, description_or_error_str = future.result()
                        processed_image_results[(page_num_res, img_idx_res)] = description_or_error_str
                    except Exception as e_future_res:
                        pg, idx = future_to_task_meta[future]
                        logging.error(f"Future for image (Page: {pg+1}, Img Index: {idx}) failed: {e_future_res}", exc_info=True)
                        processed_image_results[(pg,idx)] = f"\n[Image page_{pg+1}_image_{idx}.unknown Error]\nTask failed unexpectedly: {e_future_res}\n"
                    processed_count +=1
                    logging.info(f"  Gemini image analysis {processed_count}/{total_to_process_api} completed for '{pdf_filename}'...")
                    # ... (streamlit progress if needed) ...
            else:
                logging.info(f"Phase 2: No OCR-filtered image tasks were submitted to Gemini for '{pdf_filename}'.")
        logging.info(f"Phase 2 Complete: All submitted Gemini image tasks processed for '{pdf_filename}'.")
    else:
        logging.info(f"No images were deemed worthy of Gemini analysis for '{pdf_filename}' after OCR pre-check.")


    # Phase 3: Assemble final document text
    # This part remains the same, but `processed_image_results` now contains
    # results from Gemini for "worthy" images and placeholders for "unworthy" ones.
    logging.info(f"Phase 3: Assembling final document for '{pdf_filename}'...")
    # ... (streamlit progress update if needed) ...

    final_pages_content_strings = []
    # ... (rest of Phase 3 is identical, iterating through pages and images,
    #      and using `processed_image_results.get((page_num, img_idx))`
    #      to append descriptions or placeholders) ...
    num_pages_from_text_list = len(list_of_page_texts)
    total_doc_pages_for_assembly = total_pages_fitz if total_pages_fitz > 0 else num_pages_from_text_list

    for page_num in range(total_doc_pages_for_assembly):
        page_base_text = list_of_page_texts[page_num] if page_num < num_pages_from_text_list else ""
        current_page_full_content = page_base_text
        num_images_on_this_page = page_image_counts.get(page_num, 0) # From initial scan

        for img_idx in range(num_images_on_this_page):
            # This will get either Gemini's analysis or the OCR placeholder
            description = processed_image_results.get((page_num, img_idx))
            if description: # Append if not None
                current_page_full_content += f"\n{description}"
            # If an image was skipped by OCR and no placeholder was added, or if it failed extraction,
            # it simply won't have an entry in processed_image_results for that (page_num, img_idx)
            # and nothing will be appended for it. This is fine.
            
        final_pages_content_strings.append(f"=== Page {page_num+1} ===\n{current_page_full_content}\n\n")

    final_output_text = "".join(final_pages_content_strings)
    with open(output_text_file, "w", encoding="utf-8") as f:
        f.write(final_output_text)

    if is_streamlit_run and st_progress_objects:
        st_progress_objects['progress_bar'].progress(1.0)
        st_progress_objects['status_text'].success(f"✅ PDF '{pdf_filename}' fully processed!")

    logging.info(f"Phase 3 Complete: Finished processing '{pdf_filename}'. Output at {output_text_file}")
    return final_output_text


# process_pdf and process_pdf_with_progress remain wrappers around _process_pdf_core
def process_pdf(pdf_path, output_text_file, images_folder_root):
    return _process_pdf_core(pdf_path, output_text_file, images_folder_root, is_streamlit_run=False)

def process_pdf_with_progress(pdf_path, output_text_file, images_folder_root, api_key_initial_placeholder=None):
    import streamlit as st
    st_progress_objects = {
        'progress_bar': st.progress(0),
        'status_text': st.empty()
    }
    return _process_pdf_core(pdf_path, output_text_file, images_folder_root,
                             is_streamlit_run=True, st_progress_objects=st_progress_objects)

