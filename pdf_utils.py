# FILE: pdf_utils.py
# MAX_IMAGE_WORKERS set to 1 for conservative testing
from collections import defaultdict
import os
import fitz
import pdfplumber
from PIL import Image, ImageDraw
from io import BytesIO
import google.generativeai as genai
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
import time
import random
import logging
import config
import pytesseract
import re

if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(module)s.%(funcName)s:%(lineno)d] - %(message)s')

MAX_IMAGE_WORKERS = max(1, len(config.GOOGLE_API_KEYS) // 2)
MAX_OCR_WORKERS = os.cpu_count() or 4  # A sensible default for CPU-bound OCR tasks
print(MAX_IMAGE_WORKERS, MAX_OCR_WORKERS)
MAX_API_RETRIES_IMG = 5
INITIAL_BACKOFF_SECONDS_IMG = 5
CONTEXT_WINDOW_PAGES = 1
OCR_TEXT_WORTHINESS_THRESHOLD = 20
IMAGE_PROCESSING_TIMEOUT = 60  # seconds
OCR_TIMEOUT_SECONDS = 60  # Increased timeout for Tesseract

# --- Replace your helper function with this corrected version ---

# --- Replace your helper function with this enhanced version ---


def _format_table_as_markdown(table_data: list[list[str]]) -> str:
    """
    Converts a list of lists from pdfplumber into a clean Markdown table.
    ENHANCED: Automatically adds a 'Sl.No.' column for improved structure and clarity.
    """
    if not table_data:
        return ""

    # --- Step 1: Create a new data structure with the serial number column ---
    header = ["Sl.No."] + (table_data[0] if table_data else [])

    new_table_data = [header]
    # Start row enumeration from 1 for the serial number
    for i, row in enumerate(table_data[1:], 1):
        new_table_data.append([str(i)] + row)

    # --- Step 2: Clean the new data structure (including the Sl.No. column) ---
    cleaned_data = []
    for row in new_table_data:
        cleaned_row = []
        for cell in row:
            text = str(cell) if cell is not None else ""
            # Replace newlines with a space and escape pipe characters
            cleaned_text = text.replace('\n', ' ').strip().replace("|", "\\|")
            cleaned_row.append(cleaned_text)
        cleaned_data.append(cleaned_row)

    if not cleaned_data:
        return ""

    # --- Step 3: Build the final Markdown table string ---
    final_header = cleaned_data[0]
    header_len = len(final_header)
    if header_len == 0:
        return ""

    separator = ["---"] * header_len

    markdown_table = "| " + " | ".join(final_header) + " |\n"
    markdown_table += "| " + " | ".join(separator) + " |\n"

    for row in cleaned_data[1:]:
        # Ensure row has the same number of columns as the header for consistency
        while len(row) < header_len:
            row.append("")
        # Truncate row if it's somehow longer than the header
        markdown_table += "| " + " | ".join(row[:header_len]) + " |\n"

    return markdown_table
# --- (Existing functions remain the same) ---


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
            if not pdf.pages:
                logging.warning(f"No pages found in {pdf_path} by pdfplumber.")
            for i, page in enumerate(pdf.pages):
                # Try to extract text with layout awareness, which can help with columns
                page_text = page.extract_text(layout=True) or ""
                temp_plumber_pages.append(page_text)
        if any(pt.strip() for pt in temp_plumber_pages):
            page_texts_list = temp_plumber_pages
        else:
            raise Exception("pdfplumber yielded empty content or only whitespace for all pages")
    except Exception as e_plumber:
        logging.warning(f"pdfplumber failed or yielded empty for {os.path.basename(pdf_path)} ({e_plumber}). Using PyMuPDF.")
        page_texts_list = []
        try:
            with fitz.open(pdf_path) as fitz_doc:
                if not len(fitz_doc):
                    logging.warning(f"No pages found in {pdf_path} by PyMuPDF fallback.")
                for page_num_fitz in range(len(fitz_doc)):
                    page_fitz = fitz_doc.load_page(page_num_fitz)
                    page_text_fitz = page_fitz.get_text("text") or ""
                    page_texts_list.append(page_text_fitz)
        except Exception as e_fitz:
            logging.error(f"Both pdfplumber and PyMuPDF failed to extract text from {os.path.basename(pdf_path)}: {e_fitz}", exc_info=True)
            return [], ""
    concatenated_text = "\n".join(page_texts_list)
    if not concatenated_text.strip():
        logging.warning(f"No text content extracted from {os.path.basename(pdf_path)} by any method.")
    return page_texts_list, concatenated_text


def get_focused_context(all_page_texts: list[str], current_page_index: int, window_size: int = CONTEXT_WINDOW_PAGES) -> str:
    if not all_page_texts:
        return ""
    start_page = max(0, current_page_index - window_size)
    end_page = min(len(all_page_texts), current_page_index + window_size + 1)
    context_pages = all_page_texts[start_page:end_page]
    return "\n".join(context_pages).strip()


def _ocr_image_with_options(img_pil: Image.Image, psm_config: str = None, timeout: int = OCR_TIMEOUT_SECONDS):
    """Internal helper to run Tesseract OCR with given options."""
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


def _ocr_with_column_segmentation(image_bytes: bytes, psm_config: str = r'--oem 3 --psm 1'):
    """
    Attempts to OCR an image by first detecting columns and then processing each column.
    This is a heuristic approach, may not be perfect for all layouts.
    Assumes a two-column layout as a common case for academic papers.
    """
    try:
        img_pil = Image.open(BytesIO(image_bytes))
        width, height = img_pil.size

        # Define a common "middle" region to check for a column gap
        # Avoid very top (title/authors) and very bottom (footers)
        check_start_y = int(height * 0.15)
        check_end_y = int(height * 0.85)

        # Check for a vertical whitespace strip in the middle third of the page
        # This is a simple heuristic: find the darkest vertical line (least white pixels)
        # in the middle of the page, assuming it's a column separator.
        middle_x = width // 2
        search_range = int(width * 0.05)  # Search 5% left/right of center

        # Convert to grayscale for simpler pixel value analysis
        img_gray = img_pil.convert('L')

        # Sum pixel values vertically for a narrow strip around the middle
        min_density = float('inf')
        column_split_x = width // 2  # Default to exact middle if no clear split found

        for x in range(max(0, middle_x - search_range), min(width, middle_x + search_range)):
            current_density = 0
            for y in range(check_start_y, check_end_y):
                current_density += img_gray.getpixel((x, y))  # Lower value means darker/more text

            # We are looking for a *peak* in density for a column gap (i.e., whiter pixels)
            # So, sum of pixel values should be maximum for a column separator.
            # Convert to 'inverse' density to find lowest value (most black) or highest (most white)
            # Let's target the "whitest" vertical line as a separator.
            # Max sum indicates a clearer gap.
            if current_density > min_density:  # Changed to > to find the whitest line
                min_density = current_density
                column_split_x = x

        logging.debug(f"Determined column split at x={column_split_x} for image {width}x{height}.")

        # Define column regions based on the split point
        # Add a small buffer around the split to avoid cutting through text
        buffer = 5  # pixels
        col1_right = column_split_x - buffer
        col2_left = column_split_x + buffer

        # Ensure sensible boundaries
        col1_right = max(0, min(width, col1_right))
        col2_left = max(0, min(width, col2_left))

        column_texts = []

        # Crop and OCR Left Column
        if col1_right > 0:
            img_col1 = img_pil.crop((0, 0, col1_right, height))
            text_col1 = _ocr_image_with_options(img_col1, psm_config=psm_config)
            column_texts.append(text_col1)

        # Crop and OCR Right Column
        if col2_left < width:
            img_col2 = img_pil.crop((col2_left, 0, width, height))
            text_col2 = _ocr_image_with_options(img_col2, psm_config=psm_config)
            column_texts.append(text_col2)

        # If column segmentation resulted in empty text, fall back to full image OCR
        if not any(text.strip() for text in column_texts):
            logging.warning("Column segmentation resulted in empty text. Falling back to full image OCR.")
            return _ocr_image_with_options(img_pil, psm_config=psm_config)

        # Join texts, preserving reading order.
        # This simple join assumes left-then-right for typical two-column papers.
        return "\n".join(column_texts)

    except Exception as e:
        logging.error(f"Error during column segmentation and OCR: {e}. Falling back to full image OCR.", exc_info=True)
        # Fallback to single image OCR if column segmentation fails
        return _ocr_image_with_options(Image.open(BytesIO(image_bytes)), psm_config=psm_config)


# --- Replace your existing function with this faster, streamlined version ---

def is_image_worthy_for_gemini(image_bytes, min_text_len=OCR_TEXT_WORTHINESS_THRESHOLD, psm_config_override: str = None):
    """
    Performs OCR on an image and also checks if it contains enough text to be
    considered "worthy" of further analysis (for embedded images).

    REVISED: Simplified to perform only a single, reliable OCR pass. The slow,
    redundant second-pass heuristic has been removed as it is no longer needed
    thanks to the improved layout detection.
    """
    try:
        # Use the specified PSM config from the layout analysis, or a default.
        custom_oem_psm_config = psm_config_override if psm_config_override else r'--oem 3 --psm 6'

        # --- SINGLE, RELIABLE OCR PASS ---
        # We now trust the incoming PSM config and perform OCR only once.
        ocr_text = _ocr_image_with_options(Image.open(BytesIO(image_bytes)), psm_config=custom_oem_psm_config)
        ocr_text_stripped = ocr_text.strip()

        # The "worthiness" check is still used for filtering embedded images
        # before sending them to the Gemini vision model.
        is_worthy = len(ocr_text_stripped) >= min_text_len

        log_msg = f"OCR completed. Found {len(ocr_text_stripped)} chars. Image is worthy for Gemini analysis: {is_worthy}."
        logging.debug(log_msg)

        # Return the worthiness flag (for embedded images) and the extracted text (used for all cases)
        return is_worthy, ocr_text_stripped

    except Exception as e:
        logging.warning(f"Error during OCR process for an image: {e}. Pre-check skipped.", exc_info=True)
        # In case of an error, assume it's worthy to avoid losing data, and return an error message in the text.
        return True, f"[OCR failed due to error: {str(e)}]"


def process_image_task(key_for_first_attempt, image_bytes, image_ext, page_num, img_idx,
                       text_on_current_page, focused_contextual_text,
                       images_folder_for_this_pdf, original_img_info):
    img_name = f"page_{page_num+1}_image_{img_idx}.{image_ext}"
    img_path = os.path.join(images_folder_for_this_pdf, img_name)
    os.makedirs(os.path.dirname(img_path), exist_ok=True)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_process_single_image,
                                 key_for_first_attempt, image_bytes, image_ext, page_num, img_idx,
                                 text_on_current_page, focused_contextual_text,
                                 images_folder_for_this_pdf, original_img_info, img_name, img_path
                                 )

        try:
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
            "Analyze this image from a research paper thoroughly. Your goal is to extract all information in a structured and clear way.\n\n"
            "1.  **Describe Visuals:** Describe any graphs, diagrams, or charts, including their structure and significance.\n"
            "2.  **Extract Text:** Extract and clearly state all text visible in the image (e.g., labels, captions, data points).\n"
            "3.  **FORMAT TABLES:** If the image contains a table, you MUST extract its data and present it in a well-structured Markdown table format. Ensure all rows and columns are represented accurately. Handle merged cells gracefully.\n\n"
            "Summarize the key insights or findings presented in the image.\n\n"
            f"Context from the page where this image appears (page {page_num+1}):\n{text_on_current_page}\n\n"
            f"Context from surrounding pages:\n{focused_contextual_text}\n\n"
            "Provide your final, structured analysis:"
        )

        for attempt in range(MAX_API_RETRIES_IMG):
            try:
                if attempt > 0:
                    current_api_key = config.get_available_api_key()

                model_for_task = _configure_gemini_model_for_task_internal(current_api_key)
                if not model_for_task:
                    if attempt < MAX_API_RETRIES_IMG - 1:
                        time.sleep(INITIAL_BACKOFF_SECONDS_IMG * (1.5**attempt) + random.uniform(0, 1))
                        continue
                    else:
                        break

                response = model_for_task.generate_content([prompt, img_pil])
                config.record_key_success(current_api_key)

                image_context_folder_path = os.path.join(images_folder_for_this_pdf, "Extracted_images_context")
                os.makedirs(image_context_folder_path, exist_ok=True)

                image_analysis_filename = f"page_{page_num+1}_image_{img_idx}_analysis.txt"
                image_analysis_filepath = os.path.join(image_context_folder_path, image_analysis_filename)

                with open(image_analysis_filepath, "w", encoding="utf-8") as f_analysis:
                    f_analysis.write(response.text)
                logging.info(f"Saved image analysis to {image_analysis_filepath}")

                return (page_num, img_idx, f"\n[Image {img_name} Analysis (by Gemini)]\n{response.text}\n")

            except Exception as e:
                if attempt < MAX_API_RETRIES_IMG - 1:
                    time.sleep(INITIAL_BACKOFF_SECONDS_IMG * (1.5**attempt) + random.uniform(0, 1))
                else:
                    return (page_num, img_idx, f"\n[Image {img_name} Error]\nAPI call failed after retries: {str(e)}\n")

        return (page_num, img_idx, f"\n[Image {img_name} Error]\nAll API call attempts failed.\n")

    except Exception as e:
        return (page_num, img_idx, f"\n[Image {img_name} Error]\nProcessing failed: {str(e)}\n")


def _determine_pdf_layout_with_gemini(pdf_path, num_pages_to_sample=3) -> str:
    """
    Analyzes pages of a PDF using Gemini to determine its layout
    (e.g., multi-column, IEEE format) and suggest an optimal Tesseract PSM.
    Returns the Tesseract config string (e.g., '--oem 3 --psm 1') or default.
    MODIFIED: Skips the first page and uses an improved prompt for more robust analysis.
    """
    logging.info(f"Attempting to determine PDF layout for {os.path.basename(pdf_path)} using Gemini...")
    sampled_page_images = []

    try:
        with fitz.open(pdf_path) as fitz_doc:
            start_page_index = 1
            if len(fitz_doc) <= start_page_index:
                logging.warning(f"Document '{os.path.basename(pdf_path)}' is too short to skip the first page. Sampling from page 1.")
                start_page_index = 0

            end_page_index = min(len(fitz_doc), start_page_index + num_pages_to_sample)
            logging.info(f"Sampling pages {start_page_index + 1} to {end_page_index} for layout analysis.")

            for page_num in range(start_page_index, end_page_index):
                page = fitz_doc.load_page(page_num)
                pix = page.get_pixmap()
                img_bytes = pix.pil_tobytes(format="PNG")
                sampled_page_images.append(Image.open(BytesIO(img_bytes)))

        if not sampled_page_images:
            logging.warning("Could not extract any images for layout analysis. Using default PSM 6.")
            return r'--oem 3 --psm 6'

        key_for_layout_check = config.get_available_api_key()
        model_for_layout_check = _configure_gemini_model_for_task_internal(key_for_layout_check)
        if not model_for_layout_check:
            logging.error("Failed to initialize Gemini model for layout detection. Using default PSM 6.")
            return r'--oem 3 --psm 6'

        # --- MODIFIED PROMPT ---
        prompt = (
            "You are an expert document layout analyst. Your task is to analyze these page images from a research paper to determine the number of text columns. "
            "Pay close attention to the structure (single-column vs. multi-column, like an IEEE paper) and the flow of text.\n\n"
            "**Verification Step:** To confirm the layout, mentally try reading a few lines of text straight across the page. "
            "If reading across results in jumbled, nonsensical sentences, the layout is almost certainly **multi-column**. "
            "If the sentences are coherent, it is likely **single-column**.\n\n"
            "Based on your verified analysis of the layout, suggest the optimal Tesseract Page Segmentation Mode (PSM) for extracting text from the entire document. "
            "Choose *only* the PSM number from the following list that best fits:\n"
            "  * 1: (Auto-segmentation for multi-column layouts)\n"
            "  * 3: (Fully automatic page segmentation, good for multi-column)\n"
            "  * 4: (Assume a single column of text of variable sizes)\n"
            "  * 6: (Assume a single uniform block of text)\n\n"
            "Your response MUST start with 'Suggested PSM:' followed by the number, then a comma and a very brief reason for your choice (e.g., 'Multi-column text flow').\n"
            "Example: 'Suggested PSM: 1, Verified multi-column layout based on text flow.'"
        )

        contents = [prompt] + sampled_page_images

        response = model_for_layout_check.generate_content(contents)
        config.record_key_success(key_for_layout_check)

        gemini_response_text = response.text.strip()
        logging.info(f"Gemini layout analysis response: {gemini_response_text}")

        psm_match = re.search(r'Suggested PSM:\s*(\d+)', gemini_response_text)
        if psm_match:
            suggested_psm = int(psm_match.group(1))
            if suggested_psm in [1, 3, 4, 6, 11]:
                logging.info(f"Gemini suggested PSM: {suggested_psm} for '{os.path.basename(pdf_path)}'")
                return f'--oem 3 --psm {suggested_psm}'
            else:
                logging.warning(f"Gemini suggested invalid PSM: {suggested_psm}. Using default PSM 6.")
                return r'--oem 3 --psm 6'
        else:
            logging.warning("Could not extract PSM from Gemini response. Using default PSM 6.")
            return r'--oem 3 --psm 6'

    except Exception as e:
        logging.error(f"Error during Gemini PDF layout determination for {os.path.basename(pdf_path)}: {e}", exc_info=True)
        logging.warning("Falling back to default PSM 6 for OCR.")
        return r'--oem 3 --psm 6'


def _ocr_page_task(args):
    """
    Worker function to OCR a single PDF page. Designed for ThreadPoolExecutor.
    Unpacks arguments to be compatible with map.
    """
    page_num, pdf_path, psm_config, dpi = args
    try:
        # Each thread opens its own file handle to be safe.
        with fitz.open(pdf_path) as doc:
            page = doc.load_page(page_num)
            pix = page.get_pixmap(dpi=dpi)
            img_bytes = pix.pil_tobytes("png")
            # This function returns (is_worthy, ocr_text), we only need the latter.
            _, ocr_text = is_image_worthy_for_gemini(img_bytes, psm_config_override=psm_config)
            return page_num, ocr_text
    except Exception as e:
        logging.error(f"Error during OCR for page {page_num + 1} of {os.path.basename(pdf_path)}: {e}", exc_info=True)
        return page_num, f"\n[ERROR: OCR failed for page {page_num + 1}. Error: {e}]\n"


def _process_pdf_core(pdf_path, output_text_file, images_folder_root, is_streamlit_run=False, st_progress_objects=None):
    """
    Processes a PDF file by combining an intelligent adaptive strategy for text extraction
    with robust handling of tables and images, including full Streamlit UI feedback.

    - Adaptive Text Extraction: Chooses between fast direct extraction for simple
      layouts and full OCR for complex multi-column layouts.
    - Image Pre-filtering: OCRs images first to avoid sending text-heavy images
      to the more expensive vision analysis.
    - Table Extraction: Extracts digitally-defined tables into Markdown.
    - Streamlit Integration: Provides real-time progress updates to the UI.
    """
    pdf_filename = os.path.basename(pdf_path)
    pdf_filename_without_ext = os.path.splitext(pdf_filename)[0]
    images_folder_for_this_pdf = os.path.join(images_folder_root, pdf_filename_without_ext)
    os.makedirs(images_folder_for_this_pdf, exist_ok=True)

    # --- Initialize data holders ---
    list_of_page_texts = []
    all_image_tasks_for_gemini = []
    page_tables_markdown = defaultdict(list)
    # This dictionary is crucial. It will store EITHER the full Gemini description
    # OR the placeholder text for skipped images.
    processed_image_results = {}
    page_image_counts = {}
    total_pages = 0

    # --- Step 1: Intelligently Determine PDF Layout (Adaptive Strategy) ---
    logging.info(f"Analyzing layout for '{pdf_filename}'...")
    if is_streamlit_run and st_progress_objects:
        st_progress_objects['status_text'].info(f"Analyzing layout of '{pdf_filename}'...")
        st_progress_objects['progress_bar'].progress(0.01)

    # HOW THE ADAPTIVE CHOICE IS MADE:
    # This helper function returns a Tesseract Page Segmentation Mode (PSM) config.
    # If it suggests a multi-column mode ('psm 1' or 'psm 3'), we know direct
    # text extraction would fail, so we MUST use OCR. Otherwise, direct is faster and better.
    determined_psm_config = _determine_pdf_layout_with_gemini(pdf_path)
    is_multi_column = 'psm 1' in determined_psm_config or 'psm 3' in determined_psm_config
    strategy = 'Multi-column (Full OCR)' if is_multi_column else 'Single-column (Direct Extract)'
    logging.info(f"Chosen Strategy for '{pdf_filename}': {strategy}.")

    try:
        with fitz.open(pdf_path) as fitz_doc:
            total_pages = len(fitz_doc)

            # --- Step 2: Extract Main Page Text using the Chosen Strategy ---
            logging.info(f"Phase 1: Extracting text from {total_pages} pages using '{strategy}'.")
            if is_streamlit_run and st_progress_objects:
                st_progress_objects['status_text'].info(f"Step 1/3: Extracting text ({strategy})...")
                st_progress_objects['progress_bar'].progress(0.05)

            if is_multi_column:
                # --- Parallel OCR Path for Multi-Column Layouts ---
                logging.info(f"Using parallel OCR for {total_pages} pages...")
                page_ocr_results = {}  # To store results and sort later

                with ThreadPoolExecutor(max_workers=MAX_OCR_WORKERS) as executor:
                    # Each task needs the page number, file path, and OCR configuration
                    task_args = [(page_num, pdf_path, determined_psm_config, 200) for page_num in range(total_pages)]

                    # Submit tasks to the executor
                    future_to_page = {executor.submit(_ocr_page_task, args): args[0] for args in task_args}

                    processed_count = 0
                    for future in as_completed(future_to_page):
                        page_num_from_future = future_to_page[future]
                        try:
                            p_num_result, ocr_text = future.result()
                            page_ocr_results[p_num_result] = ocr_text
                        except Exception as exc:
                            logging.error(f"OCR task for page {page_num_from_future + 1} generated an exception: {exc}", exc_info=True)
                            page_ocr_results[page_num_from_future] = f"\n[ERROR: OCR process failed for page {page_num_from_future + 1}.]\n"

                        # Update progress for Streamlit if running
                        processed_count += 1
                        if is_streamlit_run and st_progress_objects:
                            progress = 0.05 + 0.35 * (processed_count / total_pages)
                            st_progress_objects['progress_bar'].progress(progress)

                # Reconstruct the list of page texts in the correct order
                list_of_page_texts = [page_ocr_results[i] for i in sorted(page_ocr_results.keys())]
            else:
                # Direct extraction path: Faster and more accurate for simple layouts.
                list_of_page_texts, _ = extract_text_from_pdf(pdf_path)
                if is_streamlit_run and st_progress_objects:
                    st_progress_objects['progress_bar'].progress(0.40)

            # --- Step 3: Extract Tables and Pre-filter Images (Page by Page) ---
            logging.info(f"Phase 2: Scanning {total_pages} pages for tables and images...")
            if is_streamlit_run and st_progress_objects:
                st_progress_objects['status_text'].info(f"Step 2/3: Finding tables & filtering images...")

            with pdfplumber.open(pdf_path) as plumber_pdf:
                for page_num in range(total_pages):
                    fitz_page = fitz_doc.load_page(page_num)

                    # 1. Extract digital tables (Functionality from Code 2)
                    for table in plumber_pdf.pages[page_num].find_tables():
                        if extracted_data := table.extract():
                            page_tables_markdown[page_num].append(_format_table_as_markdown(extracted_data))

                    # 2. OCR embedded images FIRST to decide if they need full analysis (Functionality from Code 1)
                    images_on_page_info = fitz_page.get_images(full=True)
                    page_image_counts[page_num] = len(images_on_page_info)

                    for img_idx, img_info in enumerate(images_on_page_info):
                        if base_image := fitz_doc.extract_image(img_info[0]):
                            if image_bytes := base_image.get("image"):
                                is_worthy, ocr_text_from_img = is_image_worthy_for_gemini(image_bytes, psm_config_override=determined_psm_config)

                                if is_worthy:
                                    # This image is a real picture/diagram. Queue it for Gemini.
                                    all_image_tasks_for_gemini.append({
                                        "image_bytes": image_bytes, "image_ext": base_image["ext"],
                                        "page_num": page_num, "img_idx": img_idx,
                                        "text_on_current_page": list_of_page_texts[page_num],
                                        "focused_contextual_text": get_focused_context(list_of_page_texts, page_num),
                                        "images_folder_for_this_pdf": images_folder_for_this_pdf, "original_img_info": img_info
                                    })
                                else:
                                    # This image is just text. Skip Gemini and create a placeholder.
                                    img_name = f"page_{page_num+1}_image_{img_idx}.{base_image['ext']}"
                                    placeholder_text = (f"\n[INFO: Embedded image '{img_name}' contains primarily text and was not sent for vision analysis. "
                                                        f"OCR text: '{ocr_text_from_img.strip()[:100].replace(chr(10),' ')}...']\n")
                                    # BUG FIX: Store placeholder immediately using the tuple key.
                                    processed_image_results[(page_num, img_idx)] = placeholder_text

    except Exception as e:
        error_msg = f"Critical error during PDF scan for {pdf_filename}: {e}"
        logging.error(error_msg, exc_info=True)
        if is_streamlit_run and st_progress_objects:
            st_progress_objects['status_text'].error(error_msg)
        with open(output_text_file, "w", encoding="utf-8") as f:
            f.write(error_msg)
        return error_msg

    # --- Step 4: Process "Worthy" Images in Parallel with Gemini ---
    if all_image_tasks_for_gemini:
        logging.info(f"Phase 3: Submitting {len(all_image_tasks_for_gemini)} filtered image tasks to Gemini...")
        if is_streamlit_run and st_progress_objects:
            st_progress_objects['status_text'].info(f"Step 3/3: Analyzing {len(all_image_tasks_for_gemini)} complex images...")
            st_progress_objects['progress_bar'].progress(0.45)

        with ThreadPoolExecutor(max_workers=MAX_IMAGE_WORKERS) as executor:
            future_to_meta = {executor.submit(process_image_task, config.get_available_api_key(), **task): (task["page_num"], task["img_idx"]) for task in all_image_tasks_for_gemini}
            total_to_process = len(future_to_meta)
            for i, future in enumerate(as_completed(future_to_meta), 1):
                try:
                    p_num, i_idx, result_text = future.result()
                    # BUG FIX: Store successful result with the same tuple key.
                    processed_image_results[(p_num, i_idx)] = result_text
                except Exception as e_future:
                    p_num, i_idx = future_to_meta[future]
                    logging.error(f"Future for image (Page: {p_num+1}, Img: {i_idx}) failed: {e_future}", exc_info=True)
                    processed_image_results[(p_num, i_idx)] = f"\n[ERROR: Analysis failed for image on page {p_num+1}]\n"
                if is_streamlit_run and st_progress_objects:
                    st_progress_objects['progress_bar'].progress(0.45 + 0.50 * (i / total_to_process))

    # --- Step 5: Assemble the Final Document ---
    logging.info(f"Phase 4: Assembling final document for '{pdf_filename}'...")
    if is_streamlit_run and st_progress_objects:
        st_progress_objects['status_text'].info("Assembling final document...")
        st_progress_objects['progress_bar'].progress(0.98)

    final_pages_content = []
    for page_num, page_text in enumerate(list_of_page_texts):
        page_content_parts = [page_text]
        # Append image descriptions OR placeholders in order.
        for img_idx in range(page_image_counts.get(page_num, 0)):
            # BUG FIX: Robustly retrieve content using the tuple key. This will get
            # either the Gemini result or the placeholder text we stored in Step 3.
            image_content = processed_image_results.get((page_num, img_idx), "")
            if image_content:
                page_content_parts.append(image_content)

        # Append extracted Markdown tables to the end of the page content.
        if page_tables_markdown[page_num]:
            page_content_parts.append("\n\n" + "---" * 20 + "\n### Extracted Tables\n")
            page_content_parts.extend(page_tables_markdown[page_num])

        final_pages_content.append(f"=== Page {page_num + 1} ===\n{''.join(page_content_parts)}\n\n")

    final_output = "".join(final_pages_content)
    with open(output_text_file, "w", encoding="utf-8") as f:
        f.write(final_output)

    if is_streamlit_run and st_progress_objects:
        st_progress_objects['progress_bar'].progress(1.0)
        st_progress_objects['status_text'].success(f"✅ PDF '{pdf_filename}' fully processed!")

    logging.info(f"Processing complete for '{pdf_filename}'.")
    return final_output


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