# batch_paper_analyzer.py
import os
import sys
import json
import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import time # For potential delays if needed
import random

# Assuming these files are in the same directory or your PYTHONPATH is set up
try:
    import config
    from pdf_utils import process_pdf
    from chatbot import initialize_chat_model, define_question_batches, generate_answers_for_batch
except ImportError as e:
    print(f"Import error in batch_paper_analyzer.py: {e}. Ensure all modules are accessible.")
    sys.exit(1)


# Configure logging (if not already configured by an importing script)
# This allows it to run standalone with logging too.
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s')

# --- Configuration for output paths ---
# These paths are relative to where this script (or the calling script) is run,
# or absolute if an absolute path is given.
# It's often better to make these configurable or derive from the input PDF's location.
BASE_OUTPUT_DIR = "analysis_output_for_finetuning"
PROCESSED_TEXT_DIR = os.path.join(BASE_OUTPUT_DIR, "processed_texts")
IMAGES_DIR_ROOT = os.path.join(BASE_OUTPUT_DIR, "extracted_images_from_pdfs") # Root for all PDF image folders
ANSWERS_DIR = os.path.join(BASE_OUTPUT_DIR, "qa_finetuning_data") # Where final JSONs go

# Max parallel batches for Q&A
# Using number of API keys as a heuristic for max parallelism
MAX_QA_BATCH_WORKERS = len(config.GOOGLE_API_KEYS) if config.GOOGLE_API_KEYS else 1


def analyze_single_paper_for_finetuning(pdf_file_path, custom_answers_dir=None):
    """
    Processes a single PDF: extracts content, generates batched Q&A, saves results.
    :param pdf_file_path: Absolute or relative path to the PDF.
    :param custom_answers_dir: Optional: If provided, overrides the default ANSWERS_DIR.
    """
    # Ensure output directories exist for this specific run
    # These are created here to be certain, even if called as a module.
    os.makedirs(PROCESSED_TEXT_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR_ROOT, exist_ok=True)
    
    current_answers_dir = custom_answers_dir if custom_answers_dir else ANSWERS_DIR
    os.makedirs(current_answers_dir, exist_ok=True)


    if not os.path.exists(pdf_file_path):
        logging.error(f"PDF file not found at {pdf_file_path}")
        # Create an error JSON file for this PDF
        pdf_filename_err = os.path.basename(pdf_file_path)
        pdf_name_without_ext_err = os.path.splitext(pdf_filename_err)[0]
        output_answers_file_err = os.path.join(current_answers_dir, f"{pdf_name_without_ext_err}_qna_data_batched.json")
        with open(output_answers_file_err, 'w', encoding='utf-8') as f:
            json.dump({"error_message": f"PDF file not found: {pdf_file_path}",
                       "pdf_file": pdf_filename_err}, f, indent=4)
        return False # Indicate failure

    pdf_filename = os.path.basename(pdf_file_path)
    pdf_name_without_ext = os.path.splitext(pdf_filename)[0]

    # Define output paths specific to this PDF
    output_processed_text_file = os.path.join(PROCESSED_TEXT_DIR, f"{pdf_name_without_ext}_processed_content.txt")
    # images_folder_for_this_pdf is created by process_pdf inside IMAGES_DIR_ROOT/pdf_name_without_ext
    output_answers_file = os.path.join(current_answers_dir, f"{pdf_name_without_ext}_qna_data_batched.json")

    logging.info(f"--- Starting analysis for fine-tuning data: {pdf_filename} ---")

    # Stage 1: PDF Processing (Text and Image Analysis)
    full_processed_text = None
    try:
        logging.info(f"Step 1: Processing PDF into detailed text format (includes image analysis)...")
        full_processed_text = process_pdf( # from pdf_utils.py
            pdf_path=pdf_file_path,
            output_text_file=output_processed_text_file,
            images_folder_root=IMAGES_DIR_ROOT
        )
        if hasattr(full_processed_text, 'startswith') and full_processed_text.startswith("Error:"):
             raise RuntimeError(f"PDF processing returned an error state: {full_processed_text[:300]}")
        logging.info(f"Successfully processed PDF. Detailed text saved to: {output_processed_text_file}")
    except Exception as e:
        logging.error(f"Critical error during PDF processing for {pdf_filename}: {e}", exc_info=True)
        with open(output_answers_file, 'w', encoding='utf-8') as f:
            json.dump({"error_message": f"PDF processing stage failed: {str(e)}",
                       "pdf_file": pdf_filename,
                       "question_batches_structure": {name: [q['param'] for q in qs] for name, qs in define_question_batches().items()}},
                      f, indent=4)
        return False # Indicate failure

    if not full_processed_text or not full_processed_text.strip():
        logging.error(f"No processable text extracted from {pdf_filename}. Skipping Q&A generation.")
        with open(output_answers_file, 'w', encoding='utf-8') as f:
            json.dump({"error_message": "No processable text extracted from PDF.",
                       "pdf_file": pdf_filename,
                       "question_batches_structure": {name: [q['param'] for q in qs] for name, qs in define_question_batches().items()}},
                      f, indent=4)
        return False


    # Stage 2: Q&A Generation in Batches
    all_question_batches = define_question_batches() # from chatbot.py
    logging.info(f"Step 2: Defined {len(all_question_batches)} question batches. Preparing for parallel Q&A generation.")
    
    aggregated_answers_by_batch = {}
    # Use a list of available keys that threads can pick from, or manage more centrally
    available_api_keys = list(config.GOOGLE_API_KEYS) 
    random.shuffle(available_api_keys) # Shuffle to vary initial selection
    key_idx = 0

    futures_qna = {}
    # Adjust MAX_QA_BATCH_WORKERS if it's more than the number of batches
    actual_max_workers = min(MAX_QA_BATCH_WORKERS, len(all_question_batches))
    
    with ThreadPoolExecutor(max_workers=actual_max_workers) as executor:
        for batch_idx, (batch_name, question_list_in_batch) in enumerate(all_question_batches.items()):
            # Cycle through API keys for batches
            api_key_for_batch = available_api_keys[key_idx % len(available_api_keys)]
            key_idx += 1
            
            logging.info(f"Submitting Q&A batch '{batch_name}' ({len(question_list_in_batch)} questions) with key {api_key_for_batch[:10]}...")
            
            future = executor.submit(
                generate_answers_for_batch, # from chatbot.py
                api_key_for_batch,
                question_list_in_batch,
                full_processed_text,
                batch_name
            )
            futures_qna[future] = (batch_name, question_list_in_batch)

        for future in as_completed(futures_qna):
            batch_name_completed, original_questions_in_batch = futures_qna[future]
            try:
                batch_result_dict = future.result(timeout=900) # 15 min timeout per Q&A batch (can be long)
                logging.info(f"Q&A Batch '{batch_name_completed}' completed.")
                
                current_batch_qa_pairs = []
                for q_data in original_questions_in_batch:
                    param = q_data["param"]
                    question_text = q_data["question"]
                    answer = batch_result_dict.get(param, "Answer not retrieved for this param in batch result.")
                    current_batch_qa_pairs.append({
                        "parameter": param,
                        "question": question_text,
                        "answer": answer
                    })
                aggregated_answers_by_batch[batch_name_completed] = current_batch_qa_pairs
            except Exception as e_result:
                logging.error(f"Q&A Batch '{batch_name_completed}' generated an exception: {e_result}", exc_info=True)
                batch_error_qa_pairs = [{
                    "parameter": q_data["param"], "question": q_data["question"],
                    "answer": f"Error processing this batch: {str(e_result)}"
                } for q_data in original_questions_in_batch]
                aggregated_answers_by_batch[batch_name_completed] = batch_error_qa_pairs

    logging.info("Step 3: All Q&A batches processed (or timed out/failed).")

    # Stage 4: Save the aggregated, batched Q&A data
    try:
        # Final structure: {"pdf_filename": ..., "content_source_file": ..., "batches": { "Batch Name 1": [...], ...}}
        final_output_structure = {
            "pdf_filename": pdf_filename,
            "processed_text_file": os.path.abspath(output_processed_text_file), # Store absolute path
            "answer_batches": aggregated_answers_by_batch
        }
        with open(output_answers_file, 'w', encoding='utf-8') as f:
            json.dump(final_output_structure, f, indent=4, ensure_ascii=False)
        logging.info(f"Successfully saved batched Q&A pairs for fine-tuning to: {output_answers_file}")
    except Exception as e:
        logging.error(f"Error saving final answers to JSON for {pdf_filename}: {e}", exc_info=True)
        return False # Indicate failure

    logging.info(f"--- Finished analysis for fine-tuning data: {pdf_filename} ---")
    return True # Indicate success


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze a research paper PDF and generate Q&A pairs for fine-tuning, using batched questions.")
    parser.add_argument("pdf_path", type=str, help="Path to the PDF file to analyze.")
    # Optional: allow overriding the main output directory for answers
    parser.add_argument("--answers_dir", type=str, default=ANSWERS_DIR, help=f"Directory to save the Q&A JSON files (default: {ANSWERS_DIR})")

    args = parser.parse_args()

    if not os.path.isfile(args.pdf_path):
        logging.error(f"The provided PDF path is not a file or does not exist: {args.pdf_path}")
        sys.exit(1)
    
    # Ensure the custom answers directory exists if provided
    if args.answers_dir:
        os.makedirs(args.answers_dir, exist_ok=True)

    analyze_single_paper_for_finetuning(args.pdf_path, custom_answers_dir=args.answers_dir)