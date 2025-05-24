# FILE: pdf_utils.py
# MAX_IMAGE_WORKERS set to 1 for conservative testing

import os
import fitz
import pdfplumber
from PIL import Image
from io import BytesIO
import google.generativeai as genai
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
import time
import random
import logging
import config 
import pytesseract

if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(module)s.%(funcName)s:%(lineno)d] - %(message)s')

MAX_IMAGE_WORKERS = 4 # <--- SET TO 1 FOR CONSERVATIVE TESTING
MAX_API_RETRIES_IMG = 3
INITIAL_BACKOFF_SECONDS_IMG = 12 
CONTEXT_WINDOW_PAGES = 1
OCR_TEXT_WORTHINESS_THRESHOLD = 20
IMAGE_PROCESSING_TIMEOUT = 20  # seconds

# --- (Rest of the file is the same as the version that fixed the AttributeError) ---
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

def is_image_worthy_for_gemini(image_bytes, min_text_len=OCR_TEXT_WORTHINESS_THRESHOLD):
    try:
        img_pil = Image.open(BytesIO(image_bytes))
        ocr_text = pytesseract.image_to_string(img_pil, timeout=10)
        ocr_text_stripped = ocr_text.strip()
        if len(ocr_text_stripped) >= min_text_len:
            logging.debug(f"OCR found {len(ocr_text_stripped)} chars (>= threshold {min_text_len}). Image is worthy. Text: '{ocr_text_stripped[:100]}...'")
            return True, ocr_text_stripped
        else:
            logging.debug(f"OCR found {len(ocr_text_stripped)} chars (< threshold {min_text_len}). Image deemed not worthy. Text: '{ocr_text_stripped[:100]}...'")
            return False, ocr_text_stripped
    except pytesseract.TesseractNotFoundError:
        logging.error("Tesseract is not installed or not in your PATH. OCR pre-check skipped. Sending image to Gemini.")
        return True, "[OCR skipped: Tesseract not found]"
    except RuntimeError as e_timeout: 
        logging.warning(f"Tesseract OCR timed out for an image: {e_timeout}. Pre-check skipped. Sending image to Gemini.")
        return True, "[OCR skipped: Tesseract timeout]"
    except Exception as e:
        logging.warning(f"Error during OCR pre-check for an image: {e}. Pre-check skipped. Sending image to Gemini.")
        return True, f"[OCR skipped: Error - {str(e)}]"


def process_image_task(key_for_first_attempt, image_bytes, image_ext, page_num, img_idx,
                       text_on_current_page, focused_contextual_text,
                       images_folder_for_this_pdf, original_img_info):
    img_name = f"page_{page_num+1}_image_{img_idx}.{image_ext}"
    img_path = os.path.join(images_folder_for_this_pdf, img_name)
    os.makedirs(os.path.dirname(img_path), exist_ok=True)

    # Create a future for this task with timeout handling
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_process_single_image,
            key_for_first_attempt, image_bytes, image_ext, page_num, img_idx,
            text_on_current_page, focused_contextual_text,
            images_folder_for_this_pdf, original_img_info, img_name, img_path
        )
        
        try:
            # Wait for the result with timeout
            return future.result(timeout=IMAGE_PROCESSING_TIMEOUT)
        except TimeoutError:
            logging.warning(f"Image {img_name} processing timed out after {IMAGE_PROCESSING_TIMEOUT} seconds. Skipping...")
            return (page_num, img_idx, f"\n[Image {img_name} Skipped]\nImage processing exceeded {IMAGE_PROCESSING_TIMEOUT} seconds timeout limit.\n")
        except Exception as e:
            logging.error(f"Error processing image {img_name}: {e}")
            return (page_num, img_idx, f"\n[Image {img_name} Error]\nProcessing failed: {str(e)}\n")


def _process_single_image(key_for_first_attempt, image_bytes, image_ext, page_num, img_idx,
                         text_on_current_page, focused_contextual_text,
                         images_folder_for_this_pdf, original_img_info, img_name, img_path):
    """Internal function to handle the actual image processing"""
    current_api_key = key_for_first_attempt
    model_for_task = None

    try:
        with open(img_path, "wb") as f: 
            f.write(image_bytes)
        
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
                if attempt > 0:
                    current_api_key = config.get_available_api_key()
                
                model_for_task = _configure_gemini_model_for_task_internal(current_api_key)
                if not model_for_task:
                    if attempt < MAX_API_RETRIES_IMG - 1:
                        time.sleep(INITIAL_BACKOFF_SECONDS_IMG * (1.5**attempt) + random.uniform(0,1))
                        continue
                    else:
                        break

                response = model_for_task.generate_content([prompt, img_pil])
                config.record_key_success(current_api_key)
                return (page_num, img_idx, f"\n[Image {img_name} Analysis (by Gemini)]\n{response.text}\n")

            except Exception as e:
                if attempt < MAX_API_RETRIES_IMG - 1:
                    time.sleep(INITIAL_BACKOFF_SECONDS_IMG * (1.5**attempt) + random.uniform(0,1))
                else:
                    return (page_num, img_idx, f"\n[Image {img_name} Error]\nAPI call failed after retries: {str(e)}\n")

        return (page_num, img_idx, f"\n[Image {img_name} Error]\nAll API call attempts failed.\n")

    except Exception as e:
        return (page_num, img_idx, f"\n[Image {img_name} Error]\nProcessing failed: {str(e)}\n")


def _process_pdf_core(pdf_path, output_text_file, images_folder_root, is_streamlit_run=False, st_progress_objects=None):
    pdf_filename = os.path.basename(pdf_path)
    pdf_filename_without_ext = os.path.splitext(pdf_filename)[0]
    images_folder_for_this_pdf = os.path.join(images_folder_root, pdf_filename_without_ext)
    os.makedirs(images_folder_for_this_pdf, exist_ok=True)

    list_of_page_texts, _ = extract_text_from_pdf(pdf_path)
    if not list_of_page_texts:
        error_msg = f"Error: No text pages could be extracted from '{pdf_filename}'."
        logging.error(error_msg)
        with open(output_text_file, "w", encoding="utf-8") as f: f.write(error_msg)
        if is_streamlit_run and st_progress_objects: st_progress_objects['status_text'].error(error_msg)
        return error_msg

    all_image_tasks_for_gemini = []
    page_image_counts = {}
    total_pages_fitz = 0
    processed_image_results = {}

    logging.info(f"Starting PDF Processing for: {pdf_filename}")
    try: 
        with fitz.open(pdf_path) as fitz_doc:
            total_pages_fitz = len(fitz_doc)
            logging.info(f"Phase 1: Scanning {total_pages_fitz} pages in '{pdf_filename}' to collect and OCR-filter image tasks...")
            if is_streamlit_run and st_progress_objects:
                st_progress_objects['status_text'].info(f"Scanning PDF ({total_pages_fitz} pages) for images...")
                st_progress_objects['progress_bar'].progress(0.01)

            for page_num in range(total_pages_fitz):
                if is_streamlit_run and st_progress_objects and (page_num % 5 == 0 or page_num == total_pages_fitz -1) :
                     st_progress_objects['progress_bar'].progress(min(0.1, 0.01 + 0.09 * (page_num / total_pages_fitz if total_pages_fitz > 0 else 1)))
                
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
                            is_worthy, ocr_text = is_image_worthy_for_gemini(image_bytes)
                            if is_worthy:
                                logging.debug(f"Image {img_name_for_log} is worthy. Adding to Gemini queue.")
                                all_image_tasks_for_gemini.append({
                                    "image_bytes": image_bytes, "image_ext": image_ext,
                                    "page_num": page_num, "img_idx": img_idx,
                                    "text_on_current_page": text_on_current_page,
                                    "focused_contextual_text": focused_context,
                                    "images_folder_for_this_pdf": images_folder_for_this_pdf,
                                    "original_img_info": img_info
                                })
                            else:
                                logging.info(f"Image {img_name_for_log} skipped for Gemini by OCR (Text: '{ocr_text[:50]}...').")
                                placeholder_text = f"\n[Image {img_name_for_log} OCR: Text found: '{ocr_text.strip()[:100].replace(chr(10),' ') if ocr_text else 'None'}' - Full analysis via Gemini skipped.]\n"
                                processed_image_results[(page_num, img_idx)] = placeholder_text
                        else:
                            processed_image_results[(page_num, img_idx)] = f"\n[Image page_{page_num+1}_image_{img_idx}.{img_info[1] if len(img_info)>1 else 'err'} Error]\nInvalid image data from Fitz.\n"
                    except Exception as e_extract:
                        processed_image_results[(page_num, img_idx)] = f"\n[Image page_{page_num+1}_image_{img_idx}.{img_info[1] if len(img_info)>1 else 'err'} Error]\nFitz extraction failed: {e_extract}\n"
            logging.info(f"Phase 1 Complete: Collected {len(all_image_tasks_for_gemini)} image tasks for Gemini for '{pdf_filename}'.")
    except Exception as e_fitz_scan:
        error_msg = f"Critical error during PDF scan (Phase 1) for images in {pdf_filename}: {e_fitz_scan}"
        logging.error(error_msg, exc_info=True)
        with open(output_text_file, "w", encoding="utf-8") as f: f.write(error_msg)
        if is_streamlit_run and st_progress_objects: st_progress_objects['status_text'].error(error_msg)
        return error_msg

    if all_image_tasks_for_gemini:
        logging.info(f"Phase 2: Submitting {len(all_image_tasks_for_gemini)} OCR-filtered image tasks to ThreadPoolExecutor (max_workers={MAX_IMAGE_WORKERS}) for '{pdf_filename}'.")
        if is_streamlit_run and st_progress_objects:
            st_progress_objects['status_text'].info(f"Analyzing {len(all_image_tasks_for_gemini)} images concurrently (up to {MAX_IMAGE_WORKERS} in parallel)...")
            st_progress_objects['progress_bar'].progress(0.1)
        
        actual_workers_for_images = min(MAX_IMAGE_WORKERS, len(all_image_tasks_for_gemini)) if all_image_tasks_for_gemini else 1
        with ThreadPoolExecutor(max_workers=actual_workers_for_images) as executor:
            future_to_task_meta = {}
            for task_data in all_image_tasks_for_gemini:
                try:
                    key_for_submission = config.get_available_api_key()
                except ValueError as e_get_key_init:
                    logging.error(f"Could not get initial API key for submitting image task (p{task_data['page_num']+1}, i{task_data['img_idx']}): {e_get_key_init}. Marking task as error for now.")
                    processed_image_results[(task_data['page_num'], task_data['img_idx'])] = f"\n[Image page_{task_data['page_num']+1}_image_{task_data['img_idx']}.{task_data.get('image_ext','png')} Error]\nNo API key available for submission: {e_get_key_init}.\n"
                    continue 
                future = executor.submit(
                    process_image_task,
                    key_for_submission,
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
                    if is_streamlit_run and st_progress_objects and total_to_process_api > 0:
                        progress = 0.1 + 0.8 * (processed_count / total_to_process_api)
                        st_progress_objects['progress_bar'].progress(progress)
                        st_progress_objects['status_text'].info(f"Analyzed image {processed_count}/{total_to_process_api}...")
            else:
                logging.info(f"Phase 2: No OCR-filtered image tasks were submitted to Gemini for '{pdf_filename}'.")
        logging.info(f"Phase 2 Complete: All submitted Gemini image tasks processed for '{pdf_filename}'.")
    else:
        logging.info(f"No images were deemed worthy of Gemini analysis for '{pdf_filename}' after OCR pre-check.")

    logging.info(f"Phase 3: Assembling final document for '{pdf_filename}'...")
    if is_streamlit_run and st_progress_objects:
        st_progress_objects['status_text'].info("Assembling final document content...")
        st_progress_objects['progress_bar'].progress(0.95)

    final_pages_content_strings = []
    num_pages_from_text_list = len(list_of_page_texts)
    total_doc_pages_for_assembly = total_pages_fitz if total_pages_fitz > 0 else num_pages_from_text_list

    for page_num in range(total_doc_pages_for_assembly):
        page_base_text = list_of_page_texts[page_num] if page_num < len(list_of_page_texts) else ""
        current_page_full_content = page_base_text
        num_images_on_this_page = page_image_counts.get(page_num, 0) 
        for img_idx in range(num_images_on_this_page):
            description = processed_image_results.get((page_num, img_idx))
            if description:
                current_page_full_content += f"\n{description}"
        final_pages_content_strings.append(f"=== Page {page_num+1} ===\n{current_page_full_content}\n\n")

    final_output_text = "".join(final_pages_content_strings)
    with open(output_text_file, "w", encoding="utf-8") as f:
        f.write(final_output_text)

    if is_streamlit_run and st_progress_objects:
        st_progress_objects['progress_bar'].progress(1.0)
        st_progress_objects['status_text'].success(f"✅ PDF '{pdf_filename}' fully processed!")

    logging.info(f"Phase 3 Complete: Finished processing '{pdf_filename}'. Output at {output_text_file}")
    return final_output_text

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
