# FILE: process_dataset_folder.py
# Complete file with modifications for single PDF processing pass

import os
import sys
import argparse
import logging
import time
from pathlib import Path
import json
from pdf_utils import process_pdf # Ensure this is imported

try:
    # analyze_single_paper_for_finetuning will be modified to accept pre-processed text
    from batch_paper_analyzer import analyze_single_paper_for_finetuning
except ImportError:
    print("Error: Could not import 'analyze_single_paper_for_finetuning' from batch_paper_analyzer.py.")
    print("Ensure batch_paper_analyzer.py is in the same directory or your PYTHONPATH.")
    sys.exit(1)

log_file_name = f"dataset_processing_run_{time.strftime('%Y%m%d-%H%M%S')}.log"
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - [%(filename)s.%(funcName)s:%(lineno)d] - %(message)s',
                        handlers=[logging.FileHandler(log_file_name, mode='w'),
                                  logging.StreamHandler(sys.stdout)])

CONTEXT_OUTPUT_DIR_NAME = "processed_pdf" # Renamed for clarity
PAUSE_BETWEEN_PAPERS = 10  # Reduced pause as API calls are less frequent per paper now

def process_folder(input_folder_path, target_output_answers_dir):
    input_path = Path(input_folder_path)
    if not input_path.is_dir():
        logging.error(f"Error: Input path '{input_folder_path}' is not a valid directory.")
        return

    # Create base output directory for Q&A JSONs
    os.makedirs(target_output_answers_dir, exist_ok=True)

    # Directory to store the single-pass processed text (context files)
    # This will be inside the target_output_answers_dir for better organization
    processed_content_output_dir = os.path.join(target_output_answers_dir, CONTEXT_OUTPUT_DIR_NAME)
    os.makedirs(processed_content_output_dir, exist_ok=True)

    # Directory for images extracted during the single PDF processing pass
    # This is where pdf_utils.process_pdf will store images.
    # It's good to have this also organized under the main output.
    images_root_for_pdf_processing = os.path.join(target_output_answers_dir, "extracted_images_from_pdfs")
    os.makedirs(images_root_for_pdf_processing, exist_ok=True)


    logging.info(f"Starting batch processing for PDF files in: {input_folder_path}")
    logging.info(f"Final Q&A JSON outputs will be saved to: {os.path.abspath(target_output_answers_dir)}")
    logging.info(f"Single-pass processed PDF content (text with image analysis) will be saved to: {os.path.abspath(processed_content_output_dir)}")
    logging.info(f"Images extracted during PDF processing will be saved under: {os.path.abspath(images_root_for_pdf_processing)}")


    pdf_files_processed_qna = 0
    pdf_files_skipped_qna = 0
    pdf_files_failed_processing = 0 # Tracks failures in either PDF processing or Q&A

    pdf_file_paths = list(input_path.rglob('*.pdf'))
    total_pdfs_found = len(pdf_file_paths)

    for i, pdf_file_path_obj in enumerate(pdf_file_paths):
        pdf_name = pdf_file_path_obj.stem
        absolute_pdf_path = str(pdf_file_path_obj.resolve())

        # Path for the final Q&A JSON
        output_qna_json_path = os.path.join(target_output_answers_dir, f"{pdf_name}_qna_data_batched.json")
        # Path for the comprehensive processed text file (generated once)
        processed_text_file_path = os.path.join(processed_content_output_dir, f"{pdf_name}_context.txt")

        logging.info(f"\n>>> Checking PDF {i+1}/{total_pdfs_found}: {pdf_file_path_obj.name} <<<")

        # Check if Q&A JSON already exists. If so, we can skip everything for this PDF.
        if os.path.exists(output_qna_json_path):
            logging.info(f"Skipping {pdf_name} - Q&A JSON output already exists at {output_qna_json_path}")
            pdf_files_skipped_qna += 1
            continue

        processed_text_content = None
        pdf_processing_successful = False

        try:
            # STEP 1: Process PDF (text and images) ONCE
            # Check if the processed text file already exists (e.g., from a previous partial run)
            if os.path.exists(processed_text_file_path):
                logging.info(f"Found existing processed text file: {processed_text_file_path}. Reading content.")
                with open(processed_text_file_path, 'r', encoding='utf-8') as f_in:
                    processed_text_content = f_in.read()
                if not processed_text_content.strip() or "Error:" in processed_text_content[:100]: # Basic check
                    logging.warning(f"Existing processed text file {processed_text_file_path} seems empty or contains an error. Reprocessing PDF.")
                    processed_text_content = None # Force reprocessing
                else:
                    pdf_processing_successful = True
            
            if not processed_text_content:
                logging.info(f"Processing PDF content for {pdf_name} into {processed_text_file_path}")
                # This call to process_pdf will:
                # 1. Extract text and analyze images.
                # 2. Save the combined output to `processed_text_file_path`.
                # 3. Return the combined text content.
                # Images will be saved relative to `images_root_for_pdf_processing`.
                processed_text_content = process_pdf(
                    pdf_path=absolute_pdf_path,
                    output_text_file=processed_text_file_path, # Where the full processed text is saved
                    images_folder_root=images_root_for_pdf_processing # Base for images from this PDF
                )
                if "Error:" in processed_text_content[:100]: # Check if process_pdf reported an error
                    raise RuntimeError(f"pdf_utils.process_pdf returned an error for {pdf_name}: {processed_text_content[:200]}")
                logging.info(f"PDF content processed and saved to {processed_text_file_path}")
                pdf_processing_successful = True

            # STEP 2: Generate Q&A using the pre-processed text
            if pdf_processing_successful and processed_text_content:
                logging.info(f"Generating Q&A for {pdf_name} using pre-processed text...")
                qna_success = analyze_single_paper_for_finetuning(
                    pdf_file_path=absolute_pdf_path, # Still needed for metadata like filename
                    custom_answers_dir=target_output_answers_dir,
                    # Pass the pre-processed content and its path
                    pre_processed_text_content=processed_text_content,
                    pre_processed_text_file_path=os.path.abspath(processed_text_file_path)
                )

                if qna_success:
                    pdf_files_processed_qna += 1
                    logging.info(f"Successfully generated Q&A for: {pdf_file_path_obj.name}")
                else:
                    pdf_files_failed_processing += 1
                    logging.warning(f"Q&A generation reported failure for: {pdf_file_path_obj.name}")
            else:
                # This case should ideally not be hit if error handling in process_pdf is robust
                # and raises exceptions on failure.
                logging.error(f"Skipping Q&A for {pdf_name} due to issues in prior PDF processing or empty content.")
                pdf_files_failed_processing += 1


        except Exception as e:
            pdf_files_failed_processing += 1
            logging.error(f"Error during main processing loop for {pdf_file_path_obj.name}. Error: {e}", exc_info=True)

        # Pause between papers unless it's the last one
        if i < total_pdfs_found - 1:
            logging.info(f"Waiting {PAUSE_BETWEEN_PAPERS} seconds before next paper...")
            time.sleep(PAUSE_BETWEEN_PAPERS)

    logging.info("\n--- Batch Processing Summary ---")
    logging.info(f"Total PDF files found: {total_pdfs_found}")
    logging.info(f"Q&A generated successfully: {pdf_files_processed_qna}")
    logging.info(f"Skipped (Q&A JSON already existed): {pdf_files_skipped_qna}")
    logging.info(f"Failed (PDF processing or Q&A generation): {pdf_files_failed_processing}")
    logging.info(f"Output Q&A JSON files saved in: {os.path.abspath(target_output_answers_dir)}")
    logging.info(f"Processed PDF content files saved in: {os.path.abspath(processed_content_output_dir)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process all PDF research papers in a folder to generate Q&A datasets.")
    parser.add_argument("input_folder", type=str, help="Path to the folder containing PDF files.")
    parser.add_argument("--output_answers_dir", type=str, required=True,
                        help="Specific directory where the final Q&A JSON files and processed content will be saved.")
    args = parser.parse_args()
    Path(args.output_answers_dir).mkdir(parents=True, exist_ok=True) # Ensure target dir exists
    process_folder(args.input_folder, args.output_answers_dir)
