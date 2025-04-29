"""
Extracts naming elements from Hugging Face package names using OpenAI GPT.
Reads package names from a CSV, filters by download count, sends names in batches
to the OpenAI API for analysis based on a predefined prompt, and saves the results.
"""

import os
import csv
import json
import time
import traceback
import argparse
import random
import pandas as pd
from openai import OpenAI
from loguru import logger
from tqdm import tqdm

from system_prompt import BACKGROUND_PROMPT
from schema import JSON_SCHEMA, CATEGORIES

# --- Configuration ---
CSV_FILE_PATH = 'data/HF_pkgs.csv'
OUTPUT_JSON_PATH = 'data/hf_pkg_elements.json'
OUTPUT_CSV_PATH = 'data/hf_pkg_elements.csv'
MIN_DOWNLOADS = 1000
BATCH_SIZE = 100  # Number of models to process in one API call
MAX_RERUNS = 8  # Maximum retries for a failed batch with reduced size
MODEL_NAME = "o4-mini-2025-04-16"
COST_PER_PROMPT_TOKEN = 1.10 / 1_000_000
COST_PER_COMPLETION_TOKEN = 4.40 / 1_000_000
NUM_MODELS = None  # Default to None, which means process all models

# --- Helper Functions ---

def load_packages_from_csv(file_path: str, min_downloads: int, num_models: int = None) -> list[str]:
    """
    Loads package names from a CSV file, filtering by minimum downloads.

    Args:
        file_path: Path to the CSV file.
        min_downloads: Minimum download count required.
        num_models: Number of models to return (randomly sampled). If None, returns all.

    Returns:
        A list of package names (context_id) meeting the criteria.
    """
    package_names = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    if int(row['downloads']) > min_downloads:
                        package_names.append(row['context_id'])
                except (ValueError, KeyError) as e:
                    logger.warning(f"Skipping row due to error: {e} - Row: {row}")
    except FileNotFoundError:
        logger.error(f"CSV file not found at {file_path}")
    except Exception as e:
        logger.error(f"Error reading CSV file: {e}")
    
    logger.info(f"Loaded {len(package_names)} package names with > {min_downloads} downloads.")
    
    # If num_models is specified, randomly sample that many models
    if num_models is not None and num_models < len(package_names):
        logger.info(f"Randomly sampling {num_models} package names.")
        package_names = random.sample(package_names, num_models)
        
    return package_names

def call_openai_api(package_batch: list[str]):
    """
    Sends a batch of package names to the OpenAI API for analysis.

    Args:
        package_batch: A list of package names.

    Returns:
        response: The OpenAI API response object.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it in a .env file.")

    client = OpenAI(api_key=api_key)
    
    # Prepare the input string for the API call
    # We only send the part after the '/' if it exists, as per the original script's behavior.
    simplified_names = [name.split('/')[-1] if '/' in name else name for name in package_batch]
    question_content = "\n".join(simplified_names)

    chatlog = [
        {"role": "system", "content": BACKGROUND_PROMPT},
        {"role": "user", "content": question_content}
    ]

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=chatlog,
        reasoning_effort="low",
        response_format={"type": "json_object"}
    )
    return response

def parse_api_response(response_content: str, original_batch: list[str]) -> dict:
    """
    Parses the JSON response from the OpenAI API.

    Args:
        response_content: The JSON string content from the API response.
        original_batch: The original list of package names sent in the batch.

    Returns:
        A dictionary mapping original package name to its component mappings.
    """
    parsed_results = {}
    simplified_batch = [name.split('/')[-1] if '/' in name else name for name in original_batch]
    
    # Create a lookup dictionary for simplified names to original names
    simplified_to_original = {simplified: original for original, simplified in zip(original_batch, simplified_batch)}
    
    try:
        # Parse the JSON response
        response_json = json.loads(response_content)
        
        if "packageAnalysis" not in response_json:
            logger.error(f"Response does not contain 'packageAnalysis' key: {response_content}")
            return parsed_results
            
        for item in response_json["packageAnalysis"]:
            if "name" not in item or "componentMapping" not in item:
                logger.warning(f"Missing required fields in analysis item: {item}")
                continue
                
            simplified_name = item["name"]
            component_mapping = item["componentMapping"]
            
            # Find the original name that matches this simplified name
            if simplified_name in simplified_to_original:
                original_name = simplified_to_original[simplified_name]
                parsed_results[original_name] = component_mapping
            else:
                logger.warning(f"Could not find original name for simplified name: {simplified_name}")
                
    except json.JSONDecodeError:
        logger.error(f"Failed to parse JSON response: {response_content}")
    except Exception as e:
        logger.error(f"Unexpected error parsing API response: {e}")
        logger.error(f"Response content: {response_content}")
        
    # Log stats on how many packages were successfully parsed
    logger.info(f"Successfully parsed {len(parsed_results)}/{len(original_batch)} packages")
    
    return parsed_results

def calculate_cost(usage) -> float:
    """Calculates the cost based on token usage."""
    return (usage.prompt_tokens * COST_PER_PROMPT_TOKEN +
            usage.completion_tokens * COST_PER_COMPLETION_TOKEN)

def convert_to_csv(json_data: dict, output_csv_path: str):
    """
    Converts the JSON results to a CSV format for better analysis.
    
    Args:
        json_data: Dictionary with model name as key and component mappings as value.
        output_csv_path: Path to save the CSV file.
    """
    rows = []
    
    # Process each model
    for model_name, component_mappings in json_data.items():
        # Extract the namespace and model part
        if '/' in model_name:
            namespace, model_part = model_name.split('/', 1)
        else:
            namespace = ''
            model_part = model_name
            
        # Process each component in the model name
        for mapping in component_mappings:
            component = mapping.get('component', '')
            category = mapping.get('category', '')
            category_name = CATEGORIES.get(category, 'Unknown')
            
            # Create a row for this component
            row = {
                'model_name': model_name,
                'namespace': namespace,
                'model_part': model_part,
                'component': component,
                'category': category,
                'category_name': category_name
            }
            rows.append(row)
    
    # Convert to DataFrame and save as CSV
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(output_csv_path, index=False)
        logger.info(f"Saved {len(rows)} component mappings to {output_csv_path}")
    else:
        logger.warning("No data to save to CSV")

def run_extraction(package_names: list[str], output_json_path: str, output_csv_path: str, batch_size: int):
    """
    Runs the extraction process in batches and saves results.

    Args:
        package_names: List of package names to process.
        output_json_path: Path to save the JSON results.
        output_csv_path: Path to save the CSV results.
        batch_size: Number of names per API call.
    """
    start_time = time.time()
    total_packages = len(package_names)
    all_results = {}
    processed_packages = set()

    # Load existing results if the file exists
    if os.path.exists(output_json_path):
        try:
            with open(output_json_path, 'r', encoding='utf-8') as f:
                all_results = json.load(f)
                processed_packages = set(all_results.keys())
                logger.info(f"Loaded {len(processed_packages)} existing results from {output_json_path}")
        except json.JSONDecodeError:
            logger.warning(f"Could not decode existing JSON from {output_json_path}. Starting fresh.")
        except Exception as e:
            logger.error(f"Error loading existing results: {e}. Starting fresh.")

    packages_to_process = [name for name in package_names if name not in processed_packages]
    num_to_process = len(packages_to_process)
    logger.info(f"Packages remaining to process: {num_to_process}")

    if num_to_process == 0:
        logger.info("All packages already processed.")
        # Still convert existing results to CSV
        convert_to_csv(all_results, output_csv_path)
        return

    i = 0
    rerun_cnt = 0
    total_cost = 0.0
    failed_batches_indices = [] # Store start indices of batches that failed permanently

    # Create progress bar
    pbar = tqdm(total=num_to_process, desc="Extracting name elements", unit="pkg")
    # Track already processed in this run for progress bar
    this_run_processed = 0

    while i < num_to_process:
        batch_start_time = time.time()
        # Adjust batch size based on reruns
        current_batch_size = max(1, int(batch_size / (2 ** rerun_cnt)))
        batch_end_index = min(i + current_batch_size, num_to_process)
        current_batch = packages_to_process[i:batch_end_index]
        actual_batch_size = len(current_batch) # Should equal current_batch_size unless at the end

        if actual_batch_size == 0:
            break # Should not happen, but safety check

        try:
            logger.info(f"Processing batch: {i // batch_size + 1} (Index {i} to {batch_end_index - 1}), Size: {actual_batch_size}, Rerun: {rerun_cnt}")
            response = call_openai_api(current_batch)

            # Process response
            api_content = response.choices[0].message.content
            batch_results = parse_api_response(api_content, current_batch)
            batch_cost = calculate_cost(response.usage)
            total_cost += batch_cost

            # Update results and save incrementally
            all_results.update(batch_results)
            processed_count_in_batch = len(batch_results)
            
            # Update progress bar
            pbar.update(processed_count_in_batch)
            this_run_processed += processed_count_in_batch
            
            if processed_count_in_batch < actual_batch_size:
                 logger.warning(f"Successfully processed {processed_count_in_batch}/{actual_batch_size} packages in the batch.")
            
            # Save after each successful batch (even partial)
            try:
                 with open(output_json_path, 'w', encoding='utf-8') as f:
                     json.dump(all_results, f, indent=2)
                 
                 # Also update the CSV after each batch
                 convert_to_csv(all_results, output_csv_path)
            except Exception as e:
                 logger.error(f"Failed to save results: {e}")

            batch_end_time = time.time()
            logger.success(f"Finished batch {i // batch_size + 1}. Processed: {processed_count_in_batch}/{actual_batch_size}. Cost: ${batch_cost:.4f}. Time: {batch_end_time - batch_start_time:.2f}s")
            
            i += actual_batch_size # Move index forward by the number of items sent in the batch
            rerun_cnt = 0 # Reset rerun count on success

        except Exception as e:
            logger.error(f"Error processing batch starting at index {i}: {e}")
            traceback.print_exc()
            rerun_cnt += 1
            batch_end_time = time.time()
            logger.warning(f"Batch failed (Attempt {rerun_cnt}). Time: {batch_end_time - batch_start_time:.2f}s")

            if rerun_cnt > MAX_RERUNS:
                logger.error(f"Max retries exceeded for batch starting at index {i}. Skipping this batch.")
                failed_batches_indices.append(i)
                # Save results collected so far before skipping
                try:
                     with open(output_json_path, 'w', encoding='utf-8') as f:
                         json.dump(all_results, f, indent=2)
                     
                     # Also update the CSV after skipping a batch
                     convert_to_csv(all_results, output_csv_path)
                except Exception as save_e:
                     logger.error(f"Failed to save results before skipping batch: {save_e}")

                i += actual_batch_size # Move index forward past the failed batch
                rerun_cnt = 0 # Reset for the next batch
                
                # Update progress bar to skip failed items
                pbar.update(actual_batch_size - processed_count_in_batch)
            else:
                 logger.info(f"Reducing batch size and retrying (Attempt {rerun_cnt+1})...")
                 time.sleep(2 ** rerun_cnt) # Exponential backoff before retry

    # Close the progress bar
    pbar.close()

    end_time = time.time()
    logger.info(f"--- Extraction Complete ---")
    logger.info(f"Total packages analyzed in this run: {this_run_processed}")
    logger.info(f"Total results saved: {len(all_results)}")
    logger.info(f"Total estimated cost for this run: ${total_cost:.4f}")
    logger.info(f"Total time taken: {end_time - start_time:.2f} seconds")
    if failed_batches_indices:
         logger.error(f"Failed to process batches starting at indices: {failed_batches_indices}")

    # Make sure CSV is updated at the end
    convert_to_csv(all_results, output_csv_path)


def main():
    # Parse input arguments
    parser = argparse.ArgumentParser(description='Extract naming elements from Hugging Face package names.')
    parser.add_argument('--csv_file', type=str, default=CSV_FILE_PATH, help='Path to the CSV file containing package names.')
    parser.add_argument('--output_json', type=str, default=OUTPUT_JSON_PATH, help='Path to save the JSON results.')
    parser.add_argument('--output_csv', type=str, default=OUTPUT_CSV_PATH, help='Path to save the CSV results.')
    parser.add_argument('--min_downloads', type=int, default=MIN_DOWNLOADS, help='Minimum download count required.')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Number of names per API call.')
    parser.add_argument('--num_models', type=int, default=NUM_MODELS, help='Number of models to analyze (randomly sampled). If not provided, analyze all.')
    args = parser.parse_args()

    logger.add("extraction.log", rotation="5 MB") # Log to file
    logger.info("Starting Hugging Face package name extraction...")

    # Ensure output directory exists
    output_dir = os.path.dirname(args.output_json)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")

    package_names_to_analyze = load_packages_from_csv(args.csv_file, args.min_downloads, args.num_models)

    if package_names_to_analyze:
        run_extraction(package_names_to_analyze, args.output_json, args.output_csv, args.batch_size)
    else:
        logger.warning("No package names found to analyze. Exiting.")

    logger.info("Extraction process finished.")


# --- Main Execution ---
if __name__ == "__main__":
    main()