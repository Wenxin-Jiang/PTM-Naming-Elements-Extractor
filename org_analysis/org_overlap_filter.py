import pandas as pd
import requests
import os
import time
import logging
from pathlib import Path
import argparse
from typing import Set, Optional, Tuple, Dict, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
DEFAULT_INPUT_DIR = Path(__file__).parent / "results"
DEFAULT_GEMINI_INPUT_FILE = DEFAULT_INPUT_DIR / "gemini_filtered_results_all.csv"
DEFAULT_OPENAI_INPUT_FILE = DEFAULT_INPUT_DIR / "openai_filtered_results_all.csv"
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR / "overlap_analysis" # Changed to output directory
HF_API_BASE = "https://huggingface.co/api"
API_CALL_DELAY = 2  # Seconds between API calls to avoid rate limiting
MAX_RETRIES = 3
RETRY_DELAY = 5  # Seconds to wait before retrying API call
# --- End Configuration ---


def get_organization_members(org_id: str, hf_token: Optional[str] = None) -> Tuple[Optional[Set[str]], str]:
    """
    Fetches the list of member usernames for a given Hugging Face organization ID.

    Args:
        org_id: The Hugging Face organization ID (e.g., 'google').
        hf_token: Optional Hugging Face API token.

    Returns:
        A tuple containing:
        - A set of member usernames if successful, None otherwise.
        - A status message ('success', 'not_found', 'private', 'api_error', 'rate_limited', 'unknown_error').
    """
    url = f"{HF_API_BASE}/organizations/{org_id}/members" # Correct endpoint for members
    headers = {}
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, headers=headers, timeout=30)

            if response.status_code == 200:
                # The /members endpoint likely returns the list directly
                members_list = response.json()
                if not isinstance(members_list, list):
                    logger.error(f"Unexpected response format from {url}. Expected a list, got {type(members_list)}.")
                    return None, "api_error"
                
                # Extract usernames - members is a list of dicts
                member_usernames = {member['user'] for member in members_list if isinstance(member, dict) and 'user' in member}
                return member_usernames, "success"
            
            elif response.status_code == 404:
                logger.warning(f"Organization '{org_id}' not found (404) or members endpoint unavailable.")
                return None, "not_found"
            elif response.status_code == 429:
                logger.warning(f"Rate limited while fetching members for '{org_id}'. Attempt {attempt + 1}/{MAX_RETRIES}.")
                if attempt < MAX_RETRIES - 1:
                    wait_time = RETRY_DELAY * (2 ** attempt) # Exponential backoff
                    logger.info(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                    continue
                else:
                    return None, "rate_limited"
            elif response.status_code == 401:
                 logger.warning(f"Unauthorized (401) fetching members for '{org_id}'. Members might be private. Token: {'Provided' if hf_token else 'Not Provided'}")
                 # If we hit the /members endpoint and get 401, it's likely private
                 return set(), "private" 
            else:
                logger.error(f"API error fetching members for '{org_id}'. Status: {response.status_code}, Response: {response.text}")
                if attempt < MAX_RETRIES - 1:
                     wait_time = RETRY_DELAY * (2 ** attempt) # Exponential backoff
                     logger.info(f"Waiting {wait_time} seconds before retry...")
                     time.sleep(wait_time)
                     continue
                else:
                    return None, "api_error"

        except requests.exceptions.RequestException as e:
            logger.error(f"Network error fetching members for '{org_id}': {e}")
            if attempt < MAX_RETRIES - 1:
                 wait_time = RETRY_DELAY * (2 ** attempt) # Exponential backoff
                 logger.info(f"Waiting {wait_time} seconds before retry...")
                 time.sleep(wait_time)
                 continue
            else:
                return None, "api_error"
        except Exception as e:
            logger.error(f"Unexpected error fetching members for '{org_id}': {e}")
            return None, "unknown_error"

    return None, "max_retries_exceeded" # Should not be reached if loop logic is correct

def check_member_overlap(spoof_org_id: str, target_org_id: str, hf_token: Optional[str] = None) -> Tuple[Optional[bool], str, str]:
    """
    Checks if two Hugging Face organizations have any overlapping members.

    Args:
        spoof_org_id: The ID of the potential spoofing organization.
        target_org_id: The ID of the legitimate target organization.
        hf_token: Optional Hugging Face API token.

    Returns:
        Tuple: (has_overlap, spoof_status, target_status)
         - has_overlap: True if overlap found, False if no overlap, None if overlap status couldn't be determined.
         - spoof_status: Status message from get_organization_members for the spoof org.
         - target_status: Status message from get_organization_members for the target org.
    """
    logger.info(f"Checking member overlap between '{spoof_org_id}' and '{target_org_id}'")

    spoof_members, spoof_status = get_organization_members(spoof_org_id, hf_token)
    # Add delay between API calls
    time.sleep(API_CALL_DELAY) 
    target_members, target_status = get_organization_members(target_org_id, hf_token)
    time.sleep(API_CALL_DELAY) # Add delay even after the second call

    if spoof_members is None or target_members is None:
        logger.warning(f"Could not determine overlap for '{spoof_org_id}' vs '{target_org_id}' due to API issues.")
        return None, spoof_status, target_status # Cannot determine overlap if one failed

    if not spoof_members or not target_members:
        # If either list is empty (or considered private/empty), assume no overlap unless error occurred
         if spoof_status == "success" and target_status == "success":
             logger.info(f"No overlap found for '{spoof_org_id}' vs '{target_org_id}' (one or both member lists are empty).")
             return False, spoof_status, target_status
         # Check if either org returned 'private' status - treat as no overlap if the other was successful or also private
         elif (spoof_status == "private" and target_status in ["success", "private"]) or \
              (target_status == "private" and spoof_status in ["success", "private"]):
             logger.info(f"Assuming no overlap for '{spoof_org_id}' vs '{target_org_id}' due to private/empty member lists.")
             return False, spoof_status, target_status # Treat private as no overlap if the other is okay or also private
         else:
             # If one had an error but the other was empty/private/success
             logger.warning(f"Could not reliably determine overlap for '{spoof_org_id}' vs '{target_org_id}' (API status: {spoof_status}, {target_status}).")
             return None, spoof_status, target_status

    # Check for intersection
    overlap = spoof_members.intersection(target_members)

    if overlap:
        logger.info(f"Overlap found for '{spoof_org_id}' vs '{target_org_id}': {overlap}")
        return True, spoof_status, target_status
    else:
        logger.info(f"No overlap found for '{spoof_org_id}' vs '{target_org_id}'.")
        return False, spoof_status, target_status

def process_input_file(input_path: Path, output_dir: Path, hf_token: Optional[str], limit: Optional[int]):
    """Processes a single input CSV file for member overlap."""
    logger.info(f"--- Starting processing for: {input_path.name} ---")
    output_dir.mkdir(parents=True, exist_ok=True) # Ensure output dir exists for this file

    # Define output paths based on input file name
    base_output_name = input_path.stem.replace("_filtered_results_all", "") # e.g., "gemini" or "openai"
    output_path = output_dir / f"{base_output_name}_overlap_filtered.csv"
    tp_path = output_dir / f"{base_output_name}_overlap_true_positives.csv"
    fp_path = output_dir / f"{base_output_name}_overlap_false_positives.csv"
    und_path = output_dir / f"{base_output_name}_overlap_undetermined.csv"

    # Load input CSV
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return

    try:
        df = pd.read_csv(input_path)
        # Filter for rows marked as typosquats
        # Check if 'is_typosquat' column exists and is boolean
        if 'is_typosquat' in df.columns and df['is_typosquat'].dtype == 'bool':
             initial_count = len(df)
             df = df[df['is_typosquat'] == True].copy()
             logger.info(f"Loaded {initial_count} rows from {input_path}, filtered to {len(df)} confirmed typosquat pairs.")
        elif 'is_typosquat' in df.columns:
            # Handle cases where it might be string "True"/"False"
            initial_count = len(df)
            df['is_typosquat'] = df['is_typosquat'].astype(str).str.lower() == 'true'
            df = df[df['is_typosquat'] == True].copy()
            logger.warning(f"'is_typosquat' column in {input_path.name} was not boolean. Attempted conversion.")
            logger.info(f"Loaded {initial_count} rows, filtered to {len(df)} confirmed typosquat pairs after conversion.")
        else:
             logger.warning(f"'is_typosquat' column not found in {input_path.name}. Processing all rows.")
             # Decide how to handle this: process all, or skip? Let's process all for now.
             # df = df.copy() # No filtering needed if column is missing
        
    except Exception as e:
        logger.error(f"Error loading or filtering input CSV '{input_path}': {str(e)}")
        return

    if df.empty:
        logger.info(f"Input file {input_path.name} contains no typosquat pairs to process after filtering.")
        # Create empty output file with expected columns
        empty_cols = list(pd.read_csv(input_path, nrows=0).columns) + ['has_member_overlap', 'final_verdict', 'spoof_org_status', 'target_org_status']
        pd.DataFrame(columns=empty_cols).to_csv(output_path, index=False)
        logger.info(f"Created empty output file: {output_path}")
        return

    # Limit number of pairs if requested
    if limit and limit > 0:
        df = df.head(limit)
        logger.info(f"Limited to first {limit} pairs for processing from {input_path.name}")

    results = []
    processed_count = 0
    start_time = time.time()

    for index, row in df.iterrows():
        # Ensure columns exist before accessing
        if 'potential_spoof_org' not in row or 'legitimate_target_org' not in row:
            logger.warning(f"Skipping row {index} in {input_path.name} due to missing required columns ('potential_spoof_org' or 'legitimate_target_org').")
            continue
            
        spoof_org = row['potential_spoof_org']
        target_org = row['legitimate_target_org']

        # Basic check for valid org names (simple example, might need refinement)
        if not isinstance(spoof_org, str) or not spoof_org or not isinstance(target_org, str) or not target_org:
             logger.warning(f"Skipping row {index} in {input_path.name} due to invalid/empty organization names: Spoof='{spoof_org}', Target='{target_org}'")
             continue

        logger.info(f"Processing pair {index + 1}/{len(df)} from {input_path.name}: '{spoof_org}' vs '{target_org}'")

        has_overlap, spoof_status, target_status = check_member_overlap(spoof_org, target_org, hf_token)

        result_row = row.to_dict()
        result_row['has_member_overlap'] = has_overlap
        result_row['spoof_org_status'] = spoof_status
        result_row['target_org_status'] = target_status

        if has_overlap is True:
            result_row['final_verdict'] = "False Positive (Overlap)"
        elif has_overlap is False:
            result_row['final_verdict'] = "True Positive (No Overlap)"
        else: # has_overlap is None
            result_row['final_verdict'] = "Undetermined (API Error/Private)" # Adjusted verdict

        results.append(result_row)
        processed_count += 1

        # Optional incremental saving
        # if processed_count % 50 == 0:
        #     temp_df = pd.DataFrame(results)
        #     temp_df.to_csv(output_path.with_suffix('.tmp.csv'), index=False)
        #     logger.info(f"Saved intermediate results for {processed_count} pairs from {input_path.name}.")


    # Create final DataFrame for this input file
    if not results:
        logger.warning(f"No results generated for {input_path.name}. Skipping saving.")
        return
        
    results_df = pd.DataFrame(results)

    # Separate into True Positives, False Positives, and Undetermined
    true_positives_df = results_df[results_df['final_verdict'] == "True Positive (No Overlap)"].copy()
    false_positives_df = results_df[results_df['final_verdict'] == "False Positive (Overlap)"].copy()
    undetermined_df = results_df[results_df['final_verdict'] == "Undetermined (API Error/Private)"].copy()

    # Save results
    try:
        results_df.to_csv(output_path, index=False)
        logger.info(f"Saved all {len(results_df)} overlap-checked results for {input_path.name} to {output_path}")

        true_positives_df.to_csv(tp_path, index=False)
        logger.info(f"Saved {len(true_positives_df)} true positives (no overlap) for {input_path.name} to {tp_path}")

        false_positives_df.to_csv(fp_path, index=False)
        logger.info(f"Saved {len(false_positives_df)} false positives (overlap) for {input_path.name} to {fp_path}")

        undetermined_df.to_csv(und_path, index=False)
        logger.info(f"Saved {len(undetermined_df)} undetermined pairs (API errors/private) for {input_path.name} to {und_path}")

    except Exception as e:
        logger.error(f"Error saving results for {input_path.name} to CSV: {e}")

    total_time = time.time() - start_time
    logger.info(f"Overlap filtering for {input_path.name} complete in {total_time:.1f}s")
    logger.info(f"Results for {input_path.name} -> True Positives: {len(true_positives_df)}, False Positives: {len(false_positives_df)}, Undetermined: {len(undetermined_df)}")
    logger.info(f"--- Finished processing for: {input_path.name} ---")


def main():
    parser = argparse.ArgumentParser(description="Filter confirmed typosquats based on Hugging Face organization member overlap.")
    parser.add_argument("--input", type=str, nargs='+', help="Path(s) to input CSV file(s) with potential typosquats",
                        default=[str(DEFAULT_GEMINI_INPUT_FILE), str(DEFAULT_OPENAI_INPUT_FILE)])
    parser.add_argument("--output-dir", type=str, help="Path to output directory for overlap-filtered results",
                        default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--hf-token", type=str, help="Hugging Face API token (optional, increases rate limits and accesses private info)", default=None)
    parser.add_argument("--limit", type=int, help="Limit number of pairs to process per input file (for testing)", default=None)

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    # Loop through each input file path provided
    for input_file_str in args.input:
        input_path = Path(input_file_str)
        process_input_file(input_path, output_dir, hf_token, args.limit)


if __name__ == "__main__":
    main()
