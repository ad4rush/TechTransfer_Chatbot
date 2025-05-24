

import os
import sys
import json
import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import random

try:
    import config
    from pdf_utils import process_pdf # Still needed as a fallback
    from chatbot import initialize_model_for_chat, define_question_batches, generate_answers_for_batch
except ImportError as e:
    logging.critical(f"Import error in batch_paper_analyzer.py: {e}. Ensure all modules are accessible.")
    sys.exit(1)

if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(module)s.%(funcName)s:%(lineno)d] - %(message)s')

# --- Configuration for output paths (used if process_pdf is called internally) ---
# These are primarily fallbacks if the script is run standalone or if pre-processed text isn't provided.
# When called by process_dataset_folder.py, custom_answers_dir will be provided.
BASE_OUTPUT_DIR_FALLBACK = "analysis_output_for_finetuning_standalone" # Fallback if run standalone
PROCESSED_TEXT_DIR_FALLBACK = os.path.join(BASE_OUTPUT_DIR_FALLBACK, "processed_texts")
IMAGES_DIR_ROOT_FALLBACK = os.path.join(BASE_OUTPUT_DIR_FALLBACK, "extracted_images_from_pdfs")
DEFAULT_ANSWERS_DIR_FALLBACK = os.path.join(BASE_OUTPUT_DIR_FALLBACK, "qa_finetuning_data")

MAX_QA_BATCH_WORKERS = len(config.GOOGLE_API_KEYS) if config.GOOGLE_API_KEYS else 1
if MAX_QA_BATCH_WORKERS == 0:
    logging.warning("No API keys found in config, MAX_QA_BATCH_WORKERS implies 0. Q&A generation will fail.")

def analyze_single_paper_for_finetuning(
    pdf_file_path,
    custom_answers_dir=None,
    pre_processed_text_content=None, # New parameter for pre-processed text
    pre_processed_text_file_path=None # New parameter for the path of the pre-processed file
):
    """
    Processes a single PDF: (optionally uses pre-processed text), generates batched Q&A, saves results.
    :param pdf_file_path: Absolute or relative path to the PDF (still needed for filename, metadata).
    :param custom_answers_dir: Optional: If provided, overrides the default ANSWERS_DIR.
    :param pre_processed_text_content: Optional: If provided, this text is used directly, skipping internal PDF processing.
    :param pre_processed_text_file_path: Optional: Path to the file where pre_processed_text_content was loaded from/saved to.
    """

    pdf_filename = os.path.basename(pdf_file_path)
    pdf_name_without_ext = os.path.splitext(pdf_filename)[0]

    # Determine directories
    answers_dir_to_use = custom_answers_dir if custom_answers_dir else DEFAULT_ANSWERS_DIR_FALLBACK
    os.makedirs(answers_dir_to_use, exist_ok=True)
    
    # Path for the final Q&A JSON output
    output_answers_file = os.path.join(answers_dir_to_use, f"{pdf_name_without_ext}_qna_data_batched.json")

    logging.info(f"--- Starting Q&A generation for: {pdf_filename} ---")
    logging.info(f"Output Q&A JSON will be saved to: {os.path.abspath(output_answers_file)}")

    if not os.path.exists(pdf_file_path): # Check original PDF existence
        logging.error(f"Original PDF file not found at {pdf_file_path}")
        with open(output_answers_file, 'w', encoding='utf-8') as f:
            json.dump({"error_message": f"Original PDF file not found: {pdf_file_path}", "pdf_file": pdf_filename}, f, indent=4)
        return False

    full_processed_text_to_use = None
    path_to_processed_text_file_to_log = None

    if pre_processed_text_content and pre_processed_text_file_path:
        logging.info(f"Using pre-processed text content provided from: {pre_processed_text_file_path}")
        full_processed_text_to_use = pre_processed_text_content
        path_to_processed_text_file_to_log = pre_processed_text_file_path # Already absolute
    else:
        # Fallback: Process PDF internally if no pre-processed text is given
        # This maintains standalone script functionality.
        logging.info("No pre-processed text provided. Processing PDF internally (text and image analysis)...")
        # Ensure fallback directories exist if we're processing internally
        os.makedirs(PROCESSED_TEXT_DIR_FALLBACK, exist_ok=True)
        os.makedirs(IMAGES_DIR_ROOT_FALLBACK, exist_ok=True)
        
        internal_processed_text_file = os.path.join(PROCESSED_TEXT_DIR_FALLBACK, f"{pdf_name_without_ext}_processed_content_internal.txt")
        path_to_processed_text_file_to_log = os.path.abspath(internal_processed_text_file)

        try:
            full_processed_text_to_use = process_pdf(
                pdf_path=pdf_file_path,
                output_text_file=internal_processed_text_file,
                images_folder_root=IMAGES_DIR_ROOT_FALLBACK
            )
            if hasattr(full_processed_text_to_use, 'startswith') and full_processed_text_to_use.startswith("Error:"):
                 raise RuntimeError(f"Internal PDF processing returned an error: {full_processed_text_to_use[:300]}")
            logging.info(f"Successfully processed PDF internally. Detailed text at: {path_to_processed_text_file_to_log}")
        except Exception as e:
            logging.error(f"Critical error during internal PDF processing for {pdf_filename}: {e}", exc_info=True)
            qbs_structure = {name: [q['param'] for q in qs] for name, qs in define_question_batches().items()}
            with open(output_answers_file, 'w', encoding='utf-8') as f:
                json.dump({"error_message": f"Internal PDF processing stage failed: {str(e)}",
                           "pdf_file": pdf_filename, "question_batches_structure": qbs_structure}, f, indent=4)
            return False

    if not full_processed_text_to_use or not full_processed_text_to_use.strip():
        logging.error(f"No processable text available for {pdf_filename} (either pre-processed or internally generated). Skipping Q&A.")
        qbs_structure = {name: [q['param'] for q in qs] for name, qs in define_question_batches().items()}
        with open(output_answers_file, 'w', encoding='utf-8') as f:
            json.dump({"error_message": "No processable text available for Q&A generation.",
                       "pdf_file": pdf_filename, "question_batches_structure": qbs_structure}, f, indent=4)
        return False

    # --- Q&A Generation Stage ---
    all_question_batches = define_question_batches()
    logging.info(f"Step 2 (Q&A): Defined {len(all_question_batches)} question batches. Preparing for parallel Q&A generation.")
    
    aggregated_answers_by_batch = {}
    futures_qna = {}
    
    num_available_keys_for_qna = len(config.GOOGLE_API_KEYS)
    actual_max_workers = min(MAX_QA_BATCH_WORKERS, len(all_question_batches), num_available_keys_for_qna)
    
    if actual_max_workers == 0 and len(all_question_batches) > 0:
        logging.error("Not enough API keys (or MAX_QA_BATCH_WORKERS=0) for Q&A batch processing.")
        with open(output_answers_file, 'w', encoding='utf-8') as f:
            json.dump({"error_message": "Insufficient API keys/workers for Q&A processing.",
                       "pdf_file": pdf_filename, "question_batches_structure": {name: [q['param'] for q in qs] for name, qs in define_question_batches().items()}},
                      f, indent=4)
        return False

    logging.info(f"Using {actual_max_workers} parallel workers for {len(all_question_batches)} Q&A batches.")

    with ThreadPoolExecutor(max_workers=actual_max_workers if actual_max_workers > 0 else 1) as executor:
        # Assign initial keys for Q&A batches. get_available_api_key will manage rotation if retries happen inside generate_answers_for_batch
        keys_assigned_for_qna_batches = set()

        for batch_name, question_list_in_batch in all_question_batches.items():
            try:
                # Each batch submission gets an initial key. Retries within generate_answers_for_batch will handle further key needs.
                initial_api_key_for_batch = config.get_available_api_key(exclude_keys=list(keys_assigned_for_qna_batches))
                keys_assigned_for_qna_batches.add(initial_api_key_for_batch) # Track to try and vary initial assignments
            except ValueError as e_get_key_init:
                logging.error(f"Could not get initial API key for Q&A batch '{batch_name}': {e_get_key_init}. Marking batch as error.")
                aggregated_answers_by_batch[batch_name] = [{
                    "parameter": q_data["param"], "question": q_data["question"],
                    "answer": f"Failed to obtain an initial API key for this Q&A batch: {e_get_key_init}"
                } for q_data in question_list_in_batch]
                continue

            logging.info(f"Submitting Q&A batch '{batch_name}' ({len(question_list_in_batch)} Qs) with initial key ...{initial_api_key_for_batch[-6:]}")
            future = executor.submit(
                generate_answers_for_batch, initial_api_key_for_batch,
                question_list_in_batch, full_processed_text_to_use, batch_name
            )
            futures_qna[future] = (batch_name, question_list_in_batch)

        for future in as_completed(futures_qna):
            batch_name_completed, original_questions_in_batch = futures_qna[future]
            try:
                # Timeout for each Q&A batch generation (e.g., 20 minutes = 1200 seconds)
                # Adjust as needed based on typical batch processing time
                batch_result_dict = future.result(timeout=1200)
                logging.info(f"Q&A Batch '{batch_name_completed}' completed processing.")
                current_batch_qa_pairs = []
                for q_data in original_questions_in_batch:
                    param, question_text = q_data["param"], q_data["question"]
                    answer = batch_result_dict.get(param, f"Answer not retrieved or error in batch result for '{param}'.")
                    current_batch_qa_pairs.append({"parameter": param, "question": question_text, "answer": answer})
                aggregated_answers_by_batch[batch_name_completed] = current_batch_qa_pairs
            except Exception as e_result:
                logging.error(f"Q&A Batch '{batch_name_completed}' generated an exception or timed out during result retrieval: {e_result}", exc_info=True)
                aggregated_answers_by_batch[batch_name_completed] = [{
                    "parameter": q["param"], "question": q["question"], "answer": f"Error processing Q&A batch '{batch_name_completed}': {e_result}"
                } for q in original_questions_in_batch]

    logging.info("Step 3 (Q&A): All Q&A batches processed (or results collected with errors/timeouts).")
    
    final_output_structure = {
        "pdf_filename": pdf_filename,
        "processed_text_file_source": path_to_processed_text_file_to_log, # Log the path of the text used
        "answer_batches": aggregated_answers_by_batch
    }
    try:
        with open(output_answers_file, 'w', encoding='utf-8') as f:
            json.dump(final_output_structure, f, indent=4, ensure_ascii=False)
        logging.info(f"Successfully saved batched Q&A pairs for fine-tuning to: {output_answers_file}")
    except Exception as e:
        logging.error(f"Error saving final answers to JSON for {pdf_filename}: {e}", exc_info=True)
        return False

    logging.info(f"--- Finished Q&A generation for: {pdf_filename} ---")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze a research paper PDF (optionally using pre-processed text) and generate Q&A pairs for fine-tuning.")
    parser.add_argument("pdf_path", type=str, help="Path to the PDF file to analyze.")
    parser.add_argument("--answers_dir", type=str,
                        help=f"Directory to save Q&A JSON files. Default: '{DEFAULT_ANSWERS_DIR_FALLBACK}' (used if run standalone).")
    # Optional arguments for providing pre-processed text (mainly for when called by another script)
    parser.add_argument("--pre_processed_text_file", type=str, default=None,
                        help="Path to an existing file containing the pre-processed text content of the PDF.")

    args = parser.parse_args()

    if not os.path.isfile(args.pdf_path):
        logging.critical(f"The PDF path is not a file or does not exist: {args.pdf_path}")
        sys.exit(1)

    answers_output_directory = args.answers_dir if args.answers_dir else DEFAULT_ANSWERS_DIR_FALLBACK
    os.makedirs(answers_output_directory, exist_ok=True)
    
    # Logic to load pre-processed text if file path is given (for standalone testing)
    text_content_from_file = None
    if args.pre_processed_text_file:
        if os.path.exists(args.pre_processed_text_file):
            logging.info(f"Standalone run: Attempting to load pre-processed text from {args.pre_processed_text_file}")
            with open(args.pre_processed_text_file, 'r', encoding='utf-8') as f_in:
                text_content_from_file = f_in.read()
            if not text_content_from_file.strip():
                logging.warning(f"Standalone run: Pre-processed text file {args.pre_processed_text_file} is empty. Will process PDF internally.")
                text_content_from_file = None # Fallback to internal processing
        else:
            logging.warning(f"Standalone run: Pre-processed text file {args.pre_processed_text_file} not found. Will process PDF internally.")

    analyze_single_paper_for_finetuning(
        args.pdf_path,
        custom_answers_dir=answers_output_directory,
        pre_processed_text_content=text_content_from_file, # Pass loaded content
        pre_processed_text_file_path=os.path.abspath(args.pre_processed_text_file) if text_content_from_file else None
    )
