import pandas as pd
import subprocess
import tempfile
import os
import sys
from pathlib import Path
import logging
import shutil
import time
from typing import List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
INPUT_CSV = Path(__file__).parent / "huggingface_organizations.csv"
OUTPUT_DIR = Path(__file__).parent / "results"
SPOOFING_RESULTS_FILE = OUTPUT_DIR / "typomind_spoofing_results.csv"
LEGIT_FOLLOWER_THRESHOLD = 50
TYPOMIND_CLI_PATH = Path("~/depscan/submodules/typomind-release/__main__.py").expanduser()
BATCH_SIZE = 500  # Process this many non-legitimate orgs at once
TIMEOUT = 600  # Seconds (10 minutes) per batch
MAX_BATCHES = None  # Set to a number to limit the batches for testing; None for all
# --- End Configuration ---

def run_typomind(base_file_path: Path, adv_file_path: Path, output_file_path: Path):
    """Runs the Typomind CLI."""
    if not TYPOMIND_CLI_PATH.exists():
        logging.error(f"Typomind CLI not found at {TYPOMIND_CLI_PATH}")
        raise FileNotFoundError(f"Typomind CLI not found at {TYPOMIND_CLI_PATH}")
    
    # Create logs directory to prevent FileNotFoundError in typomind script
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    command = [
        sys.executable, # Use the current Python interpreter
        str(TYPOMIND_CLI_PATH),
        "--base_file", str(base_file_path),
        "--adv_file", str(adv_file_path),
        "--outfile_path", str(output_file_path)
    ]
    logging.info(f"Running Typomind command: {' '.join(command)}")
    try:
        # Using check=True will raise CalledProcessError if the command fails
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=TIMEOUT)
        logging.info("Typomind execution successful.")
        logging.debug(f"Typomind stdout:\n{result.stdout}")
        if result.stderr:
            logging.warning(f"Typomind stderr:\n{result.stderr}")
    except FileNotFoundError:
        logging.error(f"Error: Python executable not found or Typomind script path is incorrect.")
        raise
    except subprocess.TimeoutExpired:
        logging.error(f"Typomind process timed out after {TIMEOUT} seconds.")
        raise
    except subprocess.CalledProcessError as e:
        logging.error(f"Typomind execution failed with return code {e.returncode}")
        logging.error(f"Stdout: {e.stdout}")
        logging.error(f"Stderr: {e.stderr}")
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred while running Typomind: {e}")
        raise

def parse_typomind_output(output_file_path: Path) -> pd.DataFrame:
    """Parses the Typomind output file to extract spoofing pairs."""
    spoofing_pairs = []
    if not output_file_path.exists() or output_file_path.stat().st_size == 0:
        logging.warning(f"Typomind output file {output_file_path} is empty or does not exist.")
        return pd.DataFrame(spoofing_pairs, columns=['potential_spoof_org', 'legitimate_target_org', 'detection_info', 'timing'])

    logging.info(f"Parsing Typomind output from: {output_file_path}")
    with open(output_file_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        try:
            # Expecting format: ('target_org', 'input_org'): {'detection_info'}, timing
            parts = line.split(': ', 1)
            if len(parts) != 2:
                logging.warning(f"Skipping malformed line (unexpected format): {line}")
                continue

            # Extract pair part
            pair_str = parts[0].strip()
            if not (pair_str.startswith('(') and pair_str.endswith(')')):
                logging.warning(f"Skipping malformed line (invalid pair format): {pair_str}")
                continue

            pair_content = pair_str[1:-1].split(', ')
            if len(pair_content) != 2:
                logging.warning(f"Skipping malformed line (invalid pair content): {pair_content}")
                continue

            target_org = pair_content[0].strip("'\"")
            input_org = pair_content[1].strip("'\"")

            # Extract detection info and timing
            remainder = parts[1].strip()
            # Find the last comma to split timing from detection info
            last_comma_index = remainder.rfind(',')
            if last_comma_index == -1 or not remainder.endswith('}'): # Simple check if format is as expected
                 detection_info_str = remainder # Assume whole remainder is info if format deviates slightly
                 timing_str = "N/A"
                 logging.warning(f"Could not precisely split detection info and timing for line: {line}. Using fallback.")
            else:
                 detection_info_str = remainder[:last_comma_index].strip()
                 timing_str = remainder[last_comma_index+1:].strip()


            spoofing_pairs.append({
                'potential_spoof_org': input_org,
                'legitimate_target_org': target_org,
                'detection_info': detection_info_str,
                'timing': timing_str
            })

        except Exception as e:
            logging.warning(f"Failed to parse line: {line}. Error: {e}")

    logging.info(f"Parsed {len(spoofing_pairs)} potential spoofing pairs.")
    return pd.DataFrame(spoofing_pairs, columns=['potential_spoof_org', 'legitimate_target_org', 'detection_info', 'timing'])

def process_batch(legitimate_orgs: List[str], non_legitimate_batch: List[str], batch_num: int, temp_dir: Path) -> pd.DataFrame:
    """Process a single batch of non-legitimate organizations."""
    logging.info(f"Processing batch {batch_num} with {len(non_legitimate_batch)} organizations")
    
    # Create batch-specific input/output files
    base_file = temp_dir / f"legit_orgs.txt"
    adv_file = temp_dir / f"potential_spoofs_batch_{batch_num}.txt"
    output_file = temp_dir / f"typomind_output_batch_{batch_num}.txt"
    
    # Always rewrite the base file to ensure consistency
    with open(base_file, 'w') as f:
        f.write("\n".join(legitimate_orgs))
    
    with open(adv_file, 'w') as f:
        f.write("\n".join(non_legitimate_batch))
    
    try:
        run_typomind(base_file, adv_file, output_file)
        if output_file.exists():
            # Copy batch output to results directory for inspection
            batch_output_file = OUTPUT_DIR / f"typomind_output_batch_{batch_num}.txt"
            shutil.copy2(output_file, batch_output_file)
            logging.info(f"Copied batch {batch_num} output to {batch_output_file}")
        
        # Parse the results
        return parse_typomind_output(output_file)
    
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
        logging.error(f"Error processing batch {batch_num}: {str(e)}")
        if output_file.exists() and output_file.stat().st_size > 0:
            # If there's partial output, try to parse it
            logging.info(f"Attempting to parse partial output from batch {batch_num}")
            return parse_typomind_output(output_file)
        return pd.DataFrame(columns=['potential_spoof_org', 'legitimate_target_org', 'detection_info', 'timing'])
    except Exception as e:
        logging.error(f"Unexpected error processing batch {batch_num}: {str(e)}")
        return pd.DataFrame(columns=['potential_spoof_org', 'legitimate_target_org', 'detection_info', 'timing'])


def main():
    """Main function to orchestrate the analysis."""
    logging.info("Starting Hugging Face organization spoofing analysis.")
    
    start_time = time.time()

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Read CSV
    try:
        df = pd.read_csv(INPUT_CSV)
        logging.info(f"Read {len(df)} organizations from {INPUT_CSV}")
    except FileNotFoundError:
        logging.error(f"Input CSV file not found at {INPUT_CSV}")
        return
    except Exception as e:
        logging.error(f"Error reading CSV file: {e}")
        return

    # Ensure required columns exist
    required_cols = ['Org_ID', 'Is_Verified', 'Is_Enterprise', 'Followers']
    if not all(col in df.columns for col in required_cols):
        logging.error(f"Input CSV missing one or more required columns: {required_cols}")
        return

    # Handle potential boolean interpretation issues
    df['Is_Verified'] = df['Is_Verified'].astype(str).str.lower() == 'true'
    df['Is_Enterprise'] = df['Is_Enterprise'].astype(str).str.lower() == 'true'
    df['Followers'] = pd.to_numeric(df['Followers'], errors='coerce').fillna(0).astype(int)

    # 2. Identify Legitimate Orgs
    legit_mask = (df['Is_Verified'] == True) | \
                 (df['Is_Enterprise'] == True) | \
                 (df['Followers'] > LEGIT_FOLLOWER_THRESHOLD)
    legitimate_orgs = df[legit_mask]['Org_ID'].dropna().unique().tolist()
    logging.info(f"Identified {len(legitimate_orgs)} legitimate organizations.")

    # 3. Identify Non-Legitimate Orgs (Potential Adversaries)
    non_legitimate_orgs = df[~legit_mask]['Org_ID'].dropna().unique().tolist()
    logging.info(f"Identified {len(non_legitimate_orgs)} non-legitimate organizations.")

    if not legitimate_orgs or not non_legitimate_orgs:
        logging.warning("No legitimate or non-legitimate organizations found to compare. Exiting.")
        return

    # Calculate number of batches
    num_batches = (len(non_legitimate_orgs) + BATCH_SIZE - 1) // BATCH_SIZE
    if MAX_BATCHES and MAX_BATCHES < num_batches:
        num_batches = MAX_BATCHES
        logging.info(f"Limited to first {MAX_BATCHES} batches as configured.")
    
    logging.info(f"Processing {len(non_legitimate_orgs)} organizations in {num_batches} batches of size {BATCH_SIZE}")

    # Create a temporary directory for all batches
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        logging.info(f"Using temporary directory: {temp_dir}")
        
        # Initialize combined results dataframe
        all_results = pd.DataFrame(columns=['potential_spoof_org', 'legitimate_target_org', 'detection_info', 'timing'])
        
        # Process each batch
        for batch_num in range(num_batches):
            batch_start = batch_num * BATCH_SIZE
            batch_end = min((batch_num + 1) * BATCH_SIZE, len(non_legitimate_orgs))
            batch = non_legitimate_orgs[batch_start:batch_end]
            
            batch_start_time = time.time()
            logging.info(f"Processing batch {batch_num+1}/{num_batches} ({batch_start+1}-{batch_end} of {len(non_legitimate_orgs)})")
            
            # Process batch
            batch_results = process_batch(legitimate_orgs, batch, batch_num, temp_path)
            
            # Combine results
            if not batch_results.empty:
                all_results = pd.concat([all_results, batch_results], ignore_index=True)
                
                # Save current results after each batch to avoid losing data
                interim_save_path = OUTPUT_DIR / "typomind_spoofing_interim_results.csv"
                all_results.to_csv(interim_save_path, index=False)
                logging.info(f"Saved interim results to {interim_save_path} ({len(all_results)} entries so far)")
            
            batch_duration = time.time() - batch_start_time
            logging.info(f"Batch {batch_num+1}/{num_batches} completed in {batch_duration:.2f} seconds")

    # 7. Save final Spoofing Results
    if not all_results.empty:
        try:
            all_results.to_csv(SPOOFING_RESULTS_FILE, index=False)
            logging.info(f"Saved {len(all_results)} potential spoofing pairs detected by Typomind to {SPOOFING_RESULTS_FILE}")
        except Exception as e:
            logging.error(f"Error saving spoofing results to CSV: {e}")
    else:
        logging.info("No spoofing pairs were detected or parsed from Typomind output.")

    total_duration = time.time() - start_time
    logging.info(f"Analysis complete. Total runtime: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")


if __name__ == "__main__":
    main()
