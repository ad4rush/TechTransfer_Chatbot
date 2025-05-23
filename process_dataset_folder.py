# process_dataset_folder.py
import os
import sys
import argparse
import logging
import time
from pathlib import Path

# Assuming batch_paper_analyzer.py contains the core processing logic for a single PDF
# We can import its main function.
# If batch_paper_analyzer.py is intended to be runnable itself AND used as a module,
# its main execution block should be under `if __name__ == "__main__":`.
# Let's assume analyze_single_paper_for_finetuning is importable.
try:
    from batch_paper_analyzer import analyze_single_paper_for_finetuning
except ImportError:
    print("Error: Could not import 'analyze_single_paper_for_finetuning' from batch_paper_analyzer.py.")
    print("Ensure batch_paper_analyzer.py is in the same directory or your PYTHONPATH.")
    sys.exit(1)

# Configure logging
# This script will have its own log, or you can configure a shared one.
# For simplicity, let's use a basic config here.
# The modules it calls (pdf_utils, chatbot, batch_paper_analyzer) already have their own logging.
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)]) # Log to console

def process_folder(input_folder_path, base_output_folder_path):
    """
    Iterates through all PDF files in the input_folder_path, processes each one
    using analyze_single_paper_for_finetuning, and saves outputs to subdirectories
    within base_output_folder_path.
    """
    input_path = Path(input_folder_path)
    if not input_path.is_dir():
        logging.error(f"Error: Input path '{input_folder_path}' is not a valid directory.")
        return

    # The output structure is already handled by analyze_single_paper_for_finetuning
    # which uses BASE_OUTPUT_DIR defined in batch_paper_analyzer.py.
    # We just need to ensure that BASE_OUTPUT_DIR is what we want for this batch run.
    # For clarity, batch_paper_analyzer.py's BASE_OUTPUT_DIR will be used.
    # The `base_output_folder_path` argument here is more for conceptual grouping if needed,
    # but the actual save paths are determined by `batch_paper_analyzer.py`.
    # We can ensure the output directory for answers is correctly set up.
    
    # The `batch_paper_analyzer.py` script already defines its output structure relative to its
    # own `BASE_OUTPUT_DIR`. We are essentially calling its main worker function.
    # The `ANSWERS_DIR` from `batch_paper_analyzer` will be used.
    # If you want this script to control the *absolute* path of ANSWERS_DIR,
    # then `analyze_single_paper_for_finetuning` would need to accept it as an argument.
    # For now, we rely on `batch_paper_analyzer.py`'s internal output structure.

    logging.info(f"Starting batch processing for PDF files in: {input_folder_path}")
    logging.info(f"Outputs will be saved within the structure defined by batch_paper_analyzer.py (typically under 'analysis_output_for_finetuning')")

    pdf_files_processed = 0
    pdf_files_failed = 0
    
    # Ensure the target output directory for answers exists, as defined in batch_paper_analyzer
    # This is more of a check, as batch_paper_analyzer should create it.
    # We need to import ANSWERS_DIR from batch_paper_analyzer if we want to use its exact value here.
    # Let's assume batch_paper_analyzer.py handles its own output directory creation.

    for pdf_file in input_path.rglob('*.pdf'): # rglob searches recursively
        logging.info(f"\n>>> Found PDF: {pdf_file.name} <<<")
        try:
            # Call the main processing function for a single paper
            analyze_single_paper_for_finetuning(str(pdf_file))
            pdf_files_processed += 1
            logging.info(f"Successfully finished processing: {pdf_file.name}")
        except Exception as e:
            pdf_files_failed += 1
            logging.error(f"Failed to process {pdf_file.name}. Error: {e}", exc_info=True)
        
        # Optional: Add a small delay between processing files if desired,
        # though API key rotation should help with per-file rate limits.
        # time.sleep(5) # e.g., 5-second delay

    logging.info("\n--- Batch Processing Summary ---")
    logging.info(f"Total PDF files found and attempted: {pdf_files_processed + pdf_files_failed}")
    logging.info(f"Successfully processed: {pdf_files_processed}")
    logging.info(f"Failed to process: {pdf_files_failed}")
    logging.info(f"Output Q&A JSON files should be in the 'qa_finetuning_data' subdirectory within 'analysis_output_for_finetuning'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process all PDF research papers in a folder to generate Q&A datasets.")
    parser.add_argument("input_folder", type=str, help="Path to the folder containing PDF files.")
    # The output folder structure is largely determined by batch_paper_analyzer.py's constants.
    # This script acts as a wrapper to iterate.
    # parser.add_argument("output_base_folder", type=str, help="Base path where output subdirectories will be created.")

    args = parser.parse_args()

    # Example usage:
    # python process_dataset_folder.py "C:\Users\adars\Downloads\TechTransfer_Chatbot-main\Dataset\Dataset_1"
    # Output will go into "analysis_output_for_finetuning" as defined in batch_paper_analyzer.py

    process_folder(args.input_folder, None) # output_base_folder not strictly used if batch_analyzer defines it