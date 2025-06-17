# FILE: pdf_utils.py
from collections import defaultdict
import os
import fitz
import pdfplumber
from PIL import Image
from io import BytesIO
import google.generativeai as genai
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import random
import logging
import config
import pytesseract
import re
import json

if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(module)s.%(funcName)s:%(lineno)d] - %(message)s')

# --- Configuration for Individual Image Analysis ---
MAX_IMAGE_ANALYSIS_WORKERS = max(1, len(config.GOOGLE_API_KEYS))
MAX_OCR_WORKERS = os.cpu_count() or 4

# --- API Call Settings ---
MAX_API_RETRIES_IMG = 3
INITIAL_BACKOFF_SECONDS_IMG = 5
CONTEXT_WINDOW_PAGES = 1
OCR_TEXT_WORTHINESS_THRESHOLD = 25
IMAGE_PROCESSING_TIMEOUT = 180 
OCR_TIMEOUT_SECONDS = 60

def _format_table_as_markdown(table_data: list[list[str]]) -> str:
    if not table_data:
        return ""
    header = ["Sl.No."] + (table_data[0] if table_data else [])
    new_table_data = [header]
    for i, row in enumerate(table_data[1:], 1):
        new_table_data.append([str(i)] + row)
    cleaned_data = []
    for row in new_table_data:
        cleaned_row = []
        for cell in row:
            text = str(cell) if cell is not None else ""
            cleaned_text = text.replace('\n', ' ').strip().replace("|", "\\|")
            cleaned_row.append(cleaned_text)
        cleaned_data.append(cleaned_row)
    if not cleaned_data:
        return ""
    final_header = cleaned_data[0]
    header_len = len(final_header)
    if header_len == 0:
        return ""
    separator = ["---"] * header_len
    markdown_table = "| " + " | ".join(final_header) + " |\n"
    markdown_table += "| " + " | ".join(separator) + " |\n"
    for row in cleaned_data[1:]:
        while len(row) < header_len:
            row.append("")
        markdown_table += "| " + " | ".join(row[:header_len]) + " |\n"
    return markdown_table

def _configure_gemini_model_for_task_internal(api_key_for_task):
    """Initializes the Generative AI model for vision tasks."""
    try:
        genai.configure(api_key=api_key_for_task)
        # Using a standard, powerful vision model is best for this task
        model = genai.GenerativeModel('gemini-2.0-flash-exp') 
        return model
    except Exception as e:
        logging.error(f"Failed to configure Gemini model with key ...{api_key_for_task[-6:]}: {e}")
        return None

def extract_text_from_pdf(pdf_path) -> tuple[list[str], str]:
    page_texts_list = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if not pdf.pages:
                raise Exception("No pages found by pdfplumber.")
            page_texts_list = [p.extract_text(layout=True) or "" for p in pdf.pages]
            if not any(pt.strip() for pt in page_texts_list):
                raise Exception("pdfplumber yielded empty content for all pages.")
    except Exception as e_plumber:
        logging.warning(f"pdfplumber failed for {os.path.basename(pdf_path)} ({e_plumber}). Trying PyMuPDF fallback.")
        page_texts_list = []
        try:
            with fitz.open(pdf_path) as fitz_doc:
                if not len(fitz_doc):
                     logging.warning(f"No pages found in {pdf_path} by PyMuPDF.")
                page_texts_list = [page.get_text("text") or "" for page in fitz_doc]
        except Exception as e_fitz:
            logging.error(f"Both pdfplumber and PyMuPDF failed for {os.path.basename(pdf_path)}: {e_fitz}", exc_info=True)
            return [], ""

    concatenated_text = "\n".join(page_texts_list)
    if not concatenated_text.strip():
        logging.warning(f"No text content extracted from {os.path.basename(pdf_path)}.")
    return page_texts_list, concatenated_text

def get_focused_context(all_page_texts: list[str], current_page_index: int, window_size: int = CONTEXT_WINDOW_PAGES) -> str:
    if not all_page_texts:
        return ""
    start_page = max(0, current_page_index - window_size)
    end_page = min(len(all_page_texts), current_page_index + window_size + 1)
    context_pages = all_page_texts[start_page:end_page]
    return "\n".join(context_pages).strip()

def _ocr_image_with_options(img_pil: Image.Image, psm_config: str = None, timeout: int = OCR_TIMEOUT_SECONDS):
    try:
        ocr_text = pytesseract.image_to_string(img_pil, config=psm_config, timeout=timeout)
        return ocr_text
    except pytesseract.TesseractNotFoundError:
        logging.error("Tesseract is not installed or not in your PATH.")
        return "[OCR_ERROR: Tesseract not found]"
    except RuntimeError as e_timeout:
        logging.warning(f"Tesseract OCR timed out: {e_timeout}.")
        return "[OCR_ERROR: Tesseract timeout]"
    except Exception as e:
        logging.warning(f"Error during Tesseract OCR: {e}.")
        return f"[OCR_ERROR: {str(e)}]"

def is_image_worthy_for_gemini(image_bytes, min_text_len=OCR_TEXT_WORTHINESS_THRESHOLD, psm_config_override: str = None):
    try:
        custom_oem_psm_config = psm_config_override if psm_config_override else r'--oem 3 --psm 6'
        ocr_text = _ocr_image_with_options(Image.open(BytesIO(image_bytes)), psm_config=custom_oem_psm_config)
        ocr_text_stripped = ocr_text.strip()
        is_worthy = len(ocr_text_stripped) < min_text_len
        return is_worthy, ocr_text_stripped
    except Exception as e:
        logging.warning(f"Error during OCR worthiness check: {e}. Assuming worthy for vision analysis.", exc_info=True)
        return True, f"[OCR failed due to error: {str(e)}]"

def analyze_single_image_task(api_key, image_data):
    """
    Analyzes a single image with its surrounding text context.
    """
    page_num, img_idx = image_data['page_num'], image_data['img_idx']
    img_name, analysis_file_path = image_data['img_name'], image_data['analysis_file_path']
    key_for_task = api_key
    
    prompt = (
        "You are an expert research paper analyst. Your task is to analyze the following image in the context of the surrounding text from the research paper. "
        "Provide a complete, detailed, and comprehensive analysis. Your goal is to infer the knowledge and significance of the image based on the provided text.\n\n"
        "Your analysis must include:\n"
        "1.  **Detailed Visual Description:** Describe every element of the image, including graphs, diagrams, charts, and their labels.\n"
        "2.  **Text Transcription:** Accurately transcribe all visible text from the image.\n"
        "3.  **Table Formatting:** If the image contains a table, extract its data and format it as a clean Markdown table.\n"
        "4.  **Inferred Knowledge & Summary:** This is the most important part. Synthesize the information from the image and the context text. Explain what the image represents, what conclusions can be drawn, and what its overall significance is to the paper. This should be a detailed, multi-paragraph explanation.\n\n"
        "Do not just state what you see; explain what it means."
    )

    content_for_api_call = [
        prompt,
        "\n--- Surrounding Text Context ---\n",
        image_data['focused_contextual_text'],
        "\n--- Image for Analysis ---\n",
        Image.open(BytesIO(image_data['image_bytes']))
    ]

    for attempt in range(MAX_API_RETRIES_IMG):
        try:
            if attempt > 0:
                logging.info(f"Retrying analysis for {img_name}, attempt {attempt + 1}")
                key_for_task = config.get_available_api_key()
            
            model = _configure_gemini_model_for_task_internal(key_for_task)
            if not model:
                if attempt < MAX_API_RETRIES_IMG - 1: time.sleep(INITIAL_BACKOFF_SECONDS_IMG); continue
                else: break

            response = model.generate_content(content_for_api_call, request_options={'timeout': IMAGE_PROCESSING_TIMEOUT})
            config.record_key_success(key_for_task)
            
            analysis_text = response.text
            with open(analysis_file_path, "w", encoding="utf-8") as f_analysis:
                f_analysis.write(analysis_text)
            
            result_text = f"\n[Image {img_name} Analysis (by Gemini)]\n{analysis_text}\n"
            return (page_num, img_idx), result_text

        except Exception as e:
            error_message_str = str(e)
            logging.warning(f"Attempt {attempt + 1}/{MAX_API_RETRIES_IMG} for {img_name} failed: {error_message_str}")
            if "429" in error_message_str or "rate limit" in error_message_str.lower() or "quota" in error_message_str.lower():
                config.mark_key_as_rate_limited(key_for_task, error_message_str)

            if attempt < MAX_API_RETRIES_IMG - 1:
                time.sleep(INITIAL_BACKOFF_SECONDS_IMG * (1.5**attempt) + random.uniform(0, 1))
            else:
                error_text_final = f"\n[Image {img_name} Error]\nAPI call failed after retries: {error_message_str}\n"
                with open(analysis_file_path, "w", encoding="utf-8") as f_analysis:
                    f_analysis.write(error_text_final)
                return (page_num, img_idx), error_text_final

    fallback_error = f"\n[Image {img_name} Error]\nCould not be analyzed after {MAX_API_RETRIES_IMG} attempts.\n"
    return (page_num, img_idx), fallback_error

def _determine_pdf_layout_with_gemini(pdf_path, num_pages_to_sample=3) -> str:
    """
    Uses Gemini to determine the best Tesseract PSM.
    This version has been made robust against malformed JSON responses.
    """
    logging.info(f"Attempting to determine PDF layout for {os.path.basename(pdf_path)} using Gemini...")
    try:
        with fitz.open(pdf_path) as fitz_doc:
            if len(fitz_doc) == 0: return r'--oem 3 --psm 6'
            start_page_index = min(1, len(fitz_doc) - 1)
            end_page_index = min(len(fitz_doc), start_page_index + num_pages_to_sample)
            
            sampled_page_images = [
                Image.open(BytesIO(fitz_doc.load_page(i).get_pixmap(dpi=150).pil_tobytes(format="PNG")))
                for i in range(start_page_index, end_page_index)
            ]

        if not sampled_page_images: return r'--oem 3 --psm 6'

        key = config.get_available_api_key()
        model = _configure_gemini_model_for_task_internal(key)
        if not model: return r'--oem 3 --psm 6'

        prompt = (
            "You are a document layout analysis engine. Analyze the layout (e.g., single-column, two-column). "
            "Respond with ONLY a single, valid JSON object with one key: 'suggested_psm'. "
            "The value must be an integer: 1 (for multi-column), 3 (for standard single-column), or 6 (fallback)."
        )
        response = model.generate_content([prompt] + sampled_page_images)
        config.record_key_success(key)
        
        gemini_response_text = response.text
        logging.info(f"Gemini layout analysis raw response: {gemini_response_text}")

        # --- FIX: Robust parsing using regex instead of json.loads ---
        match = re.search(r'\b([136])\b', gemini_response_text)
        if match:
            psm_value = int(match.group(1))
            logging.info(f"Gemini successfully suggested PSM: {psm_value} via robust parsing.")
            return f'--oem 3 --psm {psm_value}'
        else:
            logging.warning(f"Could not find a valid PSM value (1, 3, or 6) in Gemini response. Defaulting to PSM 6.")
            return r'--oem 3 --psm 6'

    except Exception as e:
        logging.error(f"Critical error during Gemini PDF layout determination: {e}", exc_info=True)
        return r'--oem 3 --psm 6'

def _ocr_page_task(args):
    page_num, pdf_path, psm_config, dpi = args
    try:
        with fitz.open(pdf_path) as doc:
            page = doc.load_page(page_num)
            pix = page.get_pixmap(dpi=dpi)
            img_bytes = pix.pil_tobytes("png")
            _, ocr_text = is_image_worthy_for_gemini(img_bytes, psm_config_override=psm_config)
            return page_num, ocr_text
    except Exception as e:
        logging.error(f"Error during OCR for page {page_num + 1} of {os.path.basename(pdf_path)}: {e}", exc_info=True)
        return page_num, f"\n[ERROR: OCR failed for page {page_num + 1}. Error: {e}]\n"

def _process_pdf_core(pdf_path, output_text_file, images_folder_root, is_streamlit_run=False, st_progress_objects=None, base_filename_for_output=None):
    output_base_name = base_filename_for_output or os.path.basename(pdf_path)
    pdf_filename_without_ext = os.path.splitext(output_base_name)[0]
    images_folder_for_this_pdf = os.path.join(images_folder_root, pdf_filename_without_ext)
    image_context_folder_path = os.path.join(images_folder_for_this_pdf, "Extracted_images_context")
    os.makedirs(image_context_folder_path, exist_ok=True)

    list_of_page_texts, images_for_analysis, processed_image_results = [], [], {}
    page_tables_markdown, page_image_counts = defaultdict(list), {}
    
    if is_streamlit_run:
        st_progress_objects['status_text'].info(f"Analyzing layout of '{output_base_name}'...")
        st_progress_objects['progress_bar'].progress(0.01)

    determined_psm_config = _determine_pdf_layout_with_gemini(pdf_path)
    is_multi_column = 'psm 1' in determined_psm_config
    strategy = 'Multi-column (Full Page OCR)' if is_multi_column else 'Single-column (Direct Text Extraction)'
    
    try:
        with fitz.open(pdf_path) as fitz_doc:
            total_pages = len(fitz_doc)
            if is_streamlit_run:
                st_progress_objects['status_text'].info(f"Step 1/3: Extracting text ({strategy})...")
                st_progress_objects['progress_bar'].progress(0.05)

            if is_multi_column:
                with ThreadPoolExecutor(max_workers=MAX_OCR_WORKERS) as executor:
                    results = list(executor.map(_ocr_page_task, [(i, pdf_path, determined_psm_config, 200) for i in range(total_pages)]))
                list_of_page_texts = [res[1] for res in sorted(results, key=lambda x: x[0])]
            else:
                list_of_page_texts, _ = extract_text_from_pdf(pdf_path)
            
            if is_streamlit_run: st_progress_objects['progress_bar'].progress(0.40)
            if is_streamlit_run: st_progress_objects['status_text'].info(f"Step 2/3: Finding tables & preparing images...")
            
            with pdfplumber.open(pdf_path) as plumber_pdf:
                for page_num in range(min(total_pages, len(plumber_pdf.pages))):
                    for table in plumber_pdf.pages[page_num].find_tables():
                        if extracted_data := table.extract():
                            page_tables_markdown[page_num].append(_format_table_as_markdown(extracted_data))
                    
                    images_on_page_info = fitz_doc[page_num].get_images(full=True)
                    page_image_counts[page_num] = len(images_on_page_info)

                    for img_idx, img_info in enumerate(images_on_page_info):
                        if base_image := fitz_doc.extract_image(img_info[0]):
                            image_bytes = base_image.get("image")
                            if not image_bytes: continue
                            
                            img_name = f"page_{page_num+1}_image_{img_idx}.{base_image['ext']}"
                            img_path = os.path.join(images_folder_for_this_pdf, img_name)
                            analysis_filepath = os.path.join(image_context_folder_path, f"{os.path.splitext(img_name)[0]}_analysis.txt")

                            with open(img_path, "wb") as f_img: f_img.write(image_bytes)
                            
                            is_worthy, ocr_text = is_image_worthy_for_gemini(image_bytes, psm_config_override=determined_psm_config)
                            if is_worthy:
                                images_for_analysis.append({
                                    "image_bytes": image_bytes, "page_num": page_num, "img_idx": img_idx, "img_name": img_name,
                                    "focused_contextual_text": get_focused_context(list_of_page_texts, page_num),
                                    "analysis_file_path": analysis_filepath
                                })
                            else:
                                placeholder_text = f"This image was identified as primarily text-based and was not sent for vision analysis. The extracted text is:\n---\n{ocr_text.strip()}\n---"
                                with open(analysis_filepath, "w", encoding="utf-8") as f: f.write(placeholder_text)
                                processed_image_results[(page_num, img_idx)] = f"\n[INFO: Image '{img_name}' content was extracted via OCR.]\n"
    
    except Exception as e:
        error_msg = f"Critical error during PDF scan: {os.path.basename(pdf_path)}: {e}"
        logging.error(error_msg, exc_info=True)
        if is_streamlit_run: st_progress_objects['status_text'].error(error_msg)
        with open(output_text_file, "w", encoding="utf-8") as f: f.write(error_msg)
        return error_msg

    if images_for_analysis:
        if is_streamlit_run:
            st_progress_objects['status_text'].info(f"Step 3/3: Analyzing {len(images_for_analysis)} complex images in parallel...")
            st_progress_objects['progress_bar'].progress(0.45)
        
        with ThreadPoolExecutor(max_workers=MAX_IMAGE_ANALYSIS_WORKERS) as executor:
            futures = {executor.submit(analyze_single_image_task, config.get_available_api_key(), img_data): img_data['img_name'] for img_data in images_for_analysis}
            
            for i, future in enumerate(as_completed(futures), 1):
                try:
                    img_key, result_text = future.result()
                    processed_image_results[img_key] = result_text
                except Exception as e_future:
                    logging.error(f"Future for image '{futures[future]}' failed: {e_future}", exc_info=True)
                if is_streamlit_run:
                    st_progress_objects['progress_bar'].progress(0.45 + 0.50 * (i / len(futures)))

    if is_streamlit_run:
        st_progress_objects['status_text'].info("Assembling final document...")
        st_progress_objects['progress_bar'].progress(0.98)

    final_pages_content = []
    for page_num, page_text in enumerate(list_of_page_texts):
        page_content_parts = [page_text]
        for img_idx in range(page_image_counts.get(page_num, 0)):
            if content := processed_image_results.get((page_num, img_idx)): page_content_parts.append(content)
        
        if page_tables_markdown[page_num]:
            page_content_parts.append("\n\n" + "---" * 20 + "\n### Extracted Tables\n" + "".join(page_tables_markdown[page_num]))
        
        final_pages_content.append(f"=== Page {page_num + 1} ===\n{''.join(page_content_parts)}\n\n")

    final_output = "".join(final_pages_content)
    with open(output_text_file, "w", encoding="utf-8") as f: f.write(final_output)

    if is_streamlit_run:
        st_progress_objects['progress_bar'].progress(1.0)
        st_progress_objects['status_text'].success(f"✅ PDF '{output_base_name}' fully processed!")
    
    return final_output

def process_pdf(pdf_path, output_text_file, images_folder_root):
    return _process_pdf_core(pdf_path, output_text_file, images_folder_root, is_streamlit_run=False)

def process_pdf_with_progress(pdf_path, output_text_file, images_folder_root, base_filename_for_output=None):
    import streamlit as st
    st_progress_objects = {
        'progress_bar': st.progress(0),
        'status_text': st.empty()
    }
    return _process_pdf_core(
        pdf_path=pdf_path,
        output_text_file=output_text_file,
        images_folder_root=images_folder_root,
        is_streamlit_run=True,
        st_progress_objects=st_progress_objects,
        base_filename_for_output=base_filename_for_output
    )