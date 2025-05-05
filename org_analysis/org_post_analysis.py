import pandas as pd
from openai import OpenAI, RateLimitError, APIError
import os
import time
import logging
from pathlib import Path
import json
import glob
from typing import Dict, List, Optional, Tuple, Set
import argparse
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
DEFAULT_INPUT_DIR = Path(__file__).parent / "results"
DEFAULT_INPUT_FILE = DEFAULT_INPUT_DIR / "typomind_spoofing_results.csv"
DEFAULT_OUTPUT_FILE = DEFAULT_INPUT_DIR / "openai_filtered_results.csv"
RESUME_FILE = DEFAULT_INPUT_DIR / "openai_processing_state.json"
OPENAI_MODEL = "o4-mini-2025-04-16"
API_CALLS_PER_MINUTE = 90
BATCH_SIZE = 10
MAX_RETRIES = 3

def generate_pair_id(pair: Dict) -> str:
    """Generate a unique ID for a typosquat pair to track processing state."""
    pair_str = f"{pair['potential_spoof_org']}:{pair['legitimate_target_org']}"
    return hashlib.md5(pair_str.encode()).hexdigest()

def setup_openai_api(api_key: str):
    """Configure OpenAI API client."""
    try:
        client = OpenAI(api_key=api_key)
        client.models.list() 
        logger.info(f"Configured OpenAI API client with model: {OPENAI_MODEL}")
        return client
    except Exception as e:
        logger.error(f"Failed to configure OpenAI API: {e}")
        raise

def parse_typomind_raw_outputs(results_dir: Path, processed_pairs: Optional[Set[str]] = None) -> Tuple[pd.DataFrame, Set[str]]:
    """
    Parse raw Typomind output files and convert them to a structured DataFrame.
    Skip pairs that have already been processed if processed_pairs set is provided.
    
    Args:
        results_dir: Directory containing Typomind output files (typomind_output_batch_*.txt)
        processed_pairs: Set of already processed pair IDs to skip
        
    Returns:
        Tuple of (DataFrame with parsed results, Set of processed pair IDs)
    """
    logger.info(f"Searching for raw Typomind output files in {results_dir}")
    
    # Find all Typomind output files
    output_files = list(results_dir.glob("typomind_output_batch_*.txt"))
    
    if not output_files:
        # Try with different patterns
        output_files = list(results_dir.glob("typomind_output*.txt"))
        
    if not output_files:
        logger.warning(f"No Typomind output files found in {results_dir}")
        return pd.DataFrame(columns=["potential_spoof_org", "legitimate_target_org", "detection_info"]), set()
    
    logger.info(f"Found {len(output_files)} Typomind output files to parse")
    
    all_pairs = []
    all_pair_ids = set()
    processed_pairs = processed_pairs or set()
    
    for file_path in output_files:
        logger.info(f"Parsing {file_path}")
        
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    # Expected format: ('target_org', 'input_org'): {'detection_info'}, timing
                    parts = line.split(': ', 1)
                    if len(parts) != 2:
                        logger.debug(f"Skipping malformed line (unexpected format): {line}")
                        continue
                    
                    # Extract pair part
                    pair_str = parts[0].strip()
                    if not (pair_str.startswith('(') and pair_str.endswith(')')):
                        logger.debug(f"Skipping malformed line (invalid pair format): {pair_str}")
                        continue
                    
                    pair_content = pair_str[1:-1].split(', ')
                    if len(pair_content) != 2:
                        logger.debug(f"Skipping malformed line (invalid pair content): {pair_content}")
                        continue
                    
                    target_org = pair_content[0].strip("'\"")
                    input_org = pair_content[1].strip("'\"")
                    
                    # Extract detection info and timing
                    remainder = parts[1].strip()
                    
                    # Find the last comma to separate timing
                    last_comma_index = remainder.rfind(',')
                    if last_comma_index == -1:
                        detection_info_str = remainder
                        timing_str = "N/A"
                    else:
                        detection_info_str = remainder[:last_comma_index].strip()
                        timing_str = remainder[last_comma_index+1:].strip()
                    
                    pair_dict = {
                        "potential_spoof_org": input_org,
                        "legitimate_target_org": target_org,
                        "detection_info": detection_info_str,
                        "timing": timing_str
                    }
                    
                    # Generate unique ID for this pair to check if it's already processed
                    pair_id = generate_pair_id(pair_dict)
                    
                    # Skip if already processed
                    if pair_id in processed_pairs:
                        continue
                        
                    all_pairs.append(pair_dict)
                    all_pair_ids.add(pair_id)
                    
                except Exception as e:
                    logger.warning(f"Failed to parse line: {line}. Error: {str(e)}")
        
        except Exception as e:
            logger.error(f"Error parsing file {file_path}: {str(e)}")
    
    logger.info(f"Parsed {len(all_pairs)} new potential typosquat pairs from raw output files")
    
    # Create a DataFrame with the parsed pairs
    df = pd.DataFrame(all_pairs) if all_pairs else pd.DataFrame(columns=["potential_spoof_org", "legitimate_target_org", "detection_info", "timing"])
    
    # Save the parsed data to CSV for future use
    parsed_csv_path = results_dir / "typomind_spoofing_parsed_latest.csv"
    df.to_csv(parsed_csv_path, index=False)
    logger.info(f"Saved newly parsed typosquat pairs to {parsed_csv_path}")
    
    return df, all_pair_ids

def analyze_potential_typosquat(
    client: OpenAI,
    pair: Dict, 
    hf_data: Optional[Dict] = None
) -> Dict:
    """
    Use OpenAI to analyze if the potential typosquat is a legitimate threat.
    
    Args:
        client: OpenAI API client instance
        pair: Dict containing the potential_spoof_org and legitimate_target_org
        hf_data: Optional dict containing additional HuggingFace organization data
    
    Returns:
        Dict with original pair data and OpenAI analysis results
    """
    potential_spoof = pair["potential_spoof_org"]
    legitimate_target = pair["legitimate_target_org"]
    detection_info = pair.get("detection_info", "")
    
    # Add organization info if available
    spoof_info = hf_data.get(potential_spoof, {}) if hf_data else {}
    target_info = hf_data.get(legitimate_target, {}) if hf_data else {}
    
    spoof_followers = spoof_info.get("Followers", "Unknown")
    target_followers = target_info.get("Followers", "Unknown")
    
    spoof_models = spoof_info.get("Models", "Unknown")
    target_models = target_info.get("Models", "Unknown")
    
    system_prompt = "You are a cybersecurity expert analyzing potential typosquatting attacks on Hugging Face organization names."
    user_prompt = f"""
Analyze this potential typosquatting case:
- Potential Typosquat Org: {potential_spoof}
- Legitimate Target Org: {legitimate_target}

Determine if this is a genuine typosquatting attack attempt. Consider:
1. Name similarity and confusion potential (visual, phonetic, semantic)
2. Likelihood of user confusion based on naming patterns
3. Potential malicious intent versus coincidental similarity
4. Organizational size/popularity difference as a potential motive

Respond ONLY with a JSON object with the following structure (no introductory text or markdown):
{{
  "is_typosquat": true/false,
  "confidence": 0-1 (how confident you are in this assessment),
  "reasoning": "brief explanation of your decision",
  "risk_level": "high/medium/low",
  "recommendation": "brief action recommendation"
}}
"""

    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            reasoning_effort="low",
            response_format={ "type": "json_object" }
        )
        
        # Extract the JSON response from the message content
        resp_text = response.choices[0].message.content
        
        analysis = json.loads(resp_text)
        
        # Add original data to results
        result = {
            "potential_spoof_org": potential_spoof,
            "legitimate_target_org": legitimate_target,
            "detection_info": detection_info,
            "is_typosquat": analysis.get("is_typosquat", False),
            "confidence": analysis.get("confidence", 0),
            "reasoning": analysis.get("reasoning", ""),
            "risk_level": analysis.get("risk_level", "low"),
            "recommendation": analysis.get("recommendation", "")
        }
        return result
    
    except RateLimitError as e:
        logger.error(f"OpenAI Rate Limit Error for {potential_spoof} vs {legitimate_target}: {e}")
        raise
    except APIError as e:
        logger.error(f"OpenAI API Error for {potential_spoof} vs {legitimate_target}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error analyzing {potential_spoof} vs {legitimate_target}: {str(e)}")
        return {
            "potential_spoof_org": potential_spoof,
            "legitimate_target_org": legitimate_target,
            "detection_info": detection_info,
            "is_typosquat": False,
            "confidence": 0,
            "reasoning": f"Error in analysis: {str(e)}",
            "risk_level": "unknown",
            "recommendation": "Review manually due to analysis failure"
        }

def load_huggingface_data(csv_path: Path) -> Dict:
    """Load HuggingFace organization data from CSV into a dictionary."""
    if not csv_path.exists():
        logger.warning(f"HuggingFace organizations CSV not found at {csv_path}")
        return {}
    
    try:
        df = pd.read_csv(csv_path)
        # Create a dictionary with Org_ID as key
        org_data = {}
        for _, row in df.iterrows():
            if "Org_ID" in row and pd.notna(row["Org_ID"]):
                org_data[row["Org_ID"]] = dict(row)
        logger.info(f"Loaded data for {len(org_data)} HuggingFace organizations")
        return org_data
    except Exception as e:
        logger.error(f"Error loading HuggingFace organization data: {str(e)}")
        return {}

def load_processing_state(resume_file: Path) -> Tuple[List[Dict], Set[str]]:
    """
    Load the processing state from a previous run.
    
    Returns:
        Tuple of (List of previous results, Set of processed pair IDs)
    """
    if not resume_file.exists():
        return [], set()
    
    try:
        with open(resume_file, 'r') as f:
            state = json.load(f)
        
        previous_results = state.get("results", [])
        processed_pairs = set(state.get("processed_pair_ids", []))
        
        logger.info(f"Loaded processing state with {len(previous_results)} results and {len(processed_pairs)} processed pairs")
        return previous_results, processed_pairs
    except Exception as e:
        logger.warning(f"Failed to load processing state: {e}. Starting fresh.")
        return [], set()

def save_processing_state(resume_file: Path, all_results: List[Dict], processed_pair_ids: Set[str]) -> None:
    """Save the current processing state to resume later."""
    try:
        state = {
            "results": all_results,
            "processed_pair_ids": list(processed_pair_ids),
            "timestamp": time.time(),
            "datetime": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(resume_file, 'w') as f:
            json.dump(state, f)
        
        logger.info(f"Saved processing state with {len(all_results)} results and {len(processed_pair_ids)} processed pairs")
    except Exception as e:
        logger.error(f"Failed to save processing state: {e}")

def main():
    parser = argparse.ArgumentParser(description="Filter typosquatting detection results using OpenAI API")
    parser.add_argument("--input", type=str, help="Path to input CSV file with Typomind results", 
                        default=str(DEFAULT_INPUT_FILE))
    parser.add_argument("--results-dir", type=str, help="Directory containing Typomind output files",
                        default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--output", type=str, help="Path to output CSV file for filtered results", 
                        default=str(DEFAULT_OUTPUT_FILE))
    parser.add_argument("--hf-data", type=str, help="Path to HuggingFace organizations CSV",
                        default=str(Path(__file__).parent / "huggingface_organizations.csv"))
    parser.add_argument("--api-key", type=str, help="OpenAI API key (or set OPENAI_API_KEY environment variable)")
    parser.add_argument("--limit", type=int, help="Limit number of pairs to process per run", default=None)
    parser.add_argument("--min-confidence", type=float, help="Minimum confidence threshold for results", default=0.0)
    # Advanced options (typically not needed for normal use)
    advanced_group = parser.add_argument_group('advanced', 'Advanced options (typically not needed)')
    advanced_group.add_argument("--no-resume", action="store_true", help="Start fresh without resuming")
    advanced_group.add_argument("--no-watch", action="store_true", help="Don't watch for new batch files")
    advanced_group.add_argument("--check-interval", type=int, help=argparse.SUPPRESS, default=300)
    advanced_group.add_argument("--max-runtime", type=int, help=argparse.SUPPRESS, default=0)
    advanced_group.add_argument("--resume-file", type=str, help=argparse.SUPPRESS,
                        default=str(RESUME_FILE))
    
    args = parser.parse_args()
    
    # Get API key from args or environment variable
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("OpenAI API key not provided. Use --api-key or set OPENAI_API_KEY environment variable.")
        return
    
    # Setup paths
    results_dir = Path(args.results_dir)
    output_path = Path(args.output)
    hf_data_path = Path(args.hf_data)
    resume_file = Path(args.resume_file)
    
    # Ensure results directory exists
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize OpenAI API
    try:
        client = setup_openai_api(api_key)
    except Exception:
        return
    
    # Load HuggingFace organization data if available
    hf_data = load_huggingface_data(hf_data_path)
    
    # Load previous results and processed pairs if resuming
    previous_results = []
    processed_pair_ids = set()
    
    if not args.no_resume:
        previous_results, processed_pair_ids = load_processing_state(resume_file)
        
        # Load existing output files to ensure we have all processed pairs
        if output_path.exists():
            try:
                existing_df = pd.read_csv(output_path)
                for _, row in existing_df.iterrows():
                    pair = {
                        "potential_spoof_org": row["potential_spoof_org"],
                        "legitimate_target_org": row["legitimate_target_org"]
                    }
                    processed_pair_ids.add(generate_pair_id(pair))
            except Exception as e:
                logger.warning(f"Could not read existing output file: {e}")
                
        # Also check batch results directory
        batch_dir = output_path.parent / "batch_results"
        if batch_dir.exists():
            for batch_file in batch_dir.glob("batch_*_all.csv"):
                try:
                    batch_df = pd.read_csv(batch_file)
                    for _, row in batch_df.iterrows():
                        pair = {
                            "potential_spoof_org": row["potential_spoof_org"],
                            "legitimate_target_org": row["legitimate_target_org"]
                        }
                        processed_pair_ids.add(generate_pair_id(pair))
                except Exception as e:
                    logger.warning(f"Could not read batch file {batch_file}: {e}")
    
    # Initialize runtime tracking
    start_time = time.time()
    max_runtime_seconds = args.max_runtime
    
    # Function to check if we should continue running
    def should_continue_running():
        if max_runtime_seconds <= 0:
            return True
        
        elapsed = time.time() - start_time
        return elapsed < max_runtime_seconds
    
    # Main processing loop
    all_results = previous_results.copy()
    filtered_results = [r for r in all_results if r.get("confidence", 0) >= args.min_confidence]
    high_confidence_count = sum(1 for r in filtered_results if r.get("is_typosquat", False))
    
    if all_results:
        logger.info(f"Resuming with {len(all_results)} previous results")
        logger.info(f"Already processed {len(processed_pair_ids)} pairs")
        logger.info(f"High confidence typosquats (>={args.min_confidence}): {high_confidence_count}")
    else:
        logger.info("Starting fresh - no previous results found")
    
    # Track the iteration number
    iteration = 0
    should_watch = not args.no_watch
    
    while should_continue_running():
        iteration += 1
        if iteration > 1:
            logger.info(f"Checking for new batch files (iteration {iteration})")
        
        # Parse raw Typomind output files, skipping already processed pairs
        df, new_pair_ids = parse_typomind_raw_outputs(results_dir, processed_pair_ids)
        
        if df.empty:
            if not should_watch:
                logger.info("No new pairs to process. Exiting.")
                break
            else:
                logger.info(f"No new pairs to process. Will check again in {args.check_interval} seconds.")
                time.sleep(args.check_interval)
                continue
        
        # Limit number of pairs if requested
        if args.limit and args.limit > 0:
            df = df.head(args.limit)
            logger.info(f"Limited to {args.limit} pairs per iteration")
        
        # Process in batches
        total_pairs = len(df)
        total_batches = (total_pairs + BATCH_SIZE - 1) // BATCH_SIZE
        
        logger.info(f"Processing {total_pairs} new typosquat pairs in {total_batches} batches")
        
        # Initialize counters for this iteration
        processed = 0
        confirmed_typosquats = 0
        
        # Process each batch
        for batch_idx in range(total_batches):
            if not should_continue_running():
                logger.info("Maximum runtime reached. Saving progress and exiting.")
                break
                
            batch_start = batch_idx * BATCH_SIZE
            batch_end = min((batch_idx + 1) * BATCH_SIZE, total_pairs)
            
            logger.info(f"Processing batch {batch_idx+1}/{total_batches} (pairs {batch_start+1}-{batch_end})")
            
            # Convert batch to list of dicts
            batch_data = df.iloc[batch_start:batch_end].to_dict(orient="records")
            
            # Process batch
            batch_results = []
            for pair in batch_data:
                # Rate limiting - sleep to avoid hitting API limits
                time.sleep(60 / API_CALLS_PER_MINUTE)
                
                # Try multiple times in case of API errors
                for attempt in range(MAX_RETRIES):
                    try:
                        result = analyze_potential_typosquat(client, pair, hf_data)
                        
                        # Only consider as true typosquat if confidence > 0.95
                        if result.get("confidence", 0) <= 0.95:
                            result["is_typosquat"] = False
                            
                        all_results.append(result)  # Always add to all_results
                        
                        # Add to processed pairs
                        processed_pair_ids.add(generate_pair_id(pair))
                        
                        # Only add to filtered_results if it meets confidence threshold
                        if result.get("confidence") >= args.min_confidence:
                            filtered_results.append(result)
                            if result.get("is_typosquat", False):
                                high_confidence_count += 1
                        
                        batch_results.append(result)
                        break
                    
                    except (RateLimitError, APIError) as e:
                        logger.warning(f"OpenAI API Error (Attempt {attempt+1}/{MAX_RETRIES}): {e}")
                        if attempt == MAX_RETRIES - 1:
                            logger.error(f"Max retries reached for pair {pair}. Recording error.")
                            # Add failed entry with error message on last attempt
                            error_result = {
                                "potential_spoof_org": pair["potential_spoof_org"],
                                "legitimate_target_org": pair["legitimate_target_org"],
                                "detection_info": pair.get("detection_info", ""),
                                "is_typosquat": False,
                                "confidence": 0,
                                "reasoning": f"Failed after {MAX_RETRIES} attempts due to API error: {e}",
                                "risk_level": "unknown",
                                "recommendation": "Manual review required"
                            }
                            all_results.append(error_result)
                            processed_pair_ids.add(generate_pair_id(pair))
                            batch_results.append(error_result)
                        else:
                             time.sleep(2 * (attempt + 1))
                    
                    except Exception as e:
                        logger.error(f"Non-API error processing pair {pair}: {e}")
                        error_result = {
                            "potential_spoof_org": pair["potential_spoof_org"],
                            "legitimate_target_org": pair["legitimate_target_org"],
                            "detection_info": pair.get("detection_info", ""),
                            "is_typosquat": False,
                            "confidence": 0,
                            "reasoning": f"Non-API error during analysis: {e}",
                            "risk_level": "unknown",
                            "recommendation": "Manual review required"
                        }
                        all_results.append(error_result)
                        processed_pair_ids.add(generate_pair_id(pair))
                        batch_results.append(error_result)
                        break
            
            # Update counters
            batch_processed = len(batch_results)
            batch_confirmed = sum(1 for r in batch_results if r.get("is_typosquat", False))
            
            processed += batch_processed
            confirmed_typosquats += batch_confirmed
            
            # Save interim results - all results
            all_df = pd.DataFrame(all_results)
            all_path = output_path.with_name(f"{output_path.stem}_all{output_path.suffix}")
            all_df.to_csv(all_path, index=False)
            
            # Save batch-specific results (never overwritten)
            batch_df = pd.DataFrame(batch_results)
            batch_dir = output_path.parent / "batch_results"
            batch_dir.mkdir(exist_ok=True)
            batch_path = batch_dir / f"batch_{iteration}_{batch_idx+1}_all.csv"
            batch_df.to_csv(batch_path, index=False)
            
            # Save batch-specific filtered results if available
            batch_filtered = [r for r in batch_results if r.get("confidence", 0) >= args.min_confidence]
            if batch_filtered:
                batch_filtered_df = pd.DataFrame(batch_filtered)
                batch_filtered_path = batch_dir / f"batch_{iteration}_{batch_idx+1}_filtered.csv"
                batch_filtered_df.to_csv(batch_filtered_path, index=False)
                
                # Save batch-specific confirmed typosquats
                batch_confirmed_records = [r for r in batch_filtered if r.get("is_typosquat", False)]
                if batch_confirmed_records:
                    batch_confirmed_df = pd.DataFrame(batch_confirmed_records)
                    batch_confirmed_path = batch_dir / f"batch_{iteration}_{batch_idx+1}_confirmed.csv"
                    batch_confirmed_df.to_csv(batch_confirmed_path, index=False)
            
            # Save filtered results
            filtered_df = pd.DataFrame(filtered_results)
            if not filtered_df.empty:
                filtered_interim_path = output_path.with_name(f"{output_path.stem}_interim{output_path.suffix}")
                filtered_df.to_csv(filtered_interim_path, index=False)
            
            # Save processing state after each batch
            save_processing_state(resume_file, all_results, processed_pair_ids)
            
            logger.info(f"Batch {batch_idx+1} complete: {batch_confirmed}/{batch_processed} identified as typosquats")
            logger.info(f"Progress: {processed}/{total_pairs} processed in this iteration")
            logger.info(f"Total progress: {len(all_results)} total results, {high_confidence_count} high confidence typosquats")
            logger.info(f"Batch results saved to {batch_path}")
        
        # Save final results for this iteration
        all_results_df = pd.DataFrame(all_results)
        all_results_path = output_path.with_name(f"{output_path.stem}_all{output_path.suffix}")
        all_results_df.to_csv(all_results_path, index=False)
        
        # Sort and save filtered results
        if filtered_results:
            filtered_df = pd.DataFrame(filtered_results)
            
            # Sort by confidence and risk level (high risk and high confidence first)
            filtered_df["confidence"] = pd.to_numeric(filtered_df["confidence"], errors="coerce")
            risk_order = {"high": 0, "medium": 1, "low": 2, "unknown": 3}
            filtered_df["risk_order"] = filtered_df["risk_level"].map(lambda x: risk_order.get(x.lower() if isinstance(x, str) else "", 4))
            
            # Sort results
            sorted_df = filtered_df.sort_values(
                by=["is_typosquat", "risk_order", "confidence"], 
                ascending=[False, True, False]
            ).drop(columns=["risk_order"])
            
            # Save filtered results to CSV
            sorted_df.to_csv(output_path, index=False)
            
            # Save confirmed typosquats to separate file
            confirmed_df = sorted_df[sorted_df["is_typosquat"] == True].copy()
            confirmed_path = output_path.with_name(f"{output_path.stem}_confirmed{output_path.suffix}")
            confirmed_df.to_csv(confirmed_path, index=False)
        
        # Log iteration summary
        logger.info(f"Iteration {iteration} complete: Processed {processed} new pairs")
        logger.info(f"Total results: {len(all_results)}, High confidence typosquats: {high_confidence_count}")
        
        # If not watching for new files, exit after processing
        if not should_watch:
            logger.info("Processing complete. Exiting.")
            break
        
        # If watching for new files, wait before checking again
        logger.info(f"Waiting {args.check_interval} seconds before checking for new files")
        time.sleep(args.check_interval)
    
    # Log final summary
    total_time = time.time() - start_time
    logger.info(f"Analysis complete in {total_time:.1f}s ({total_time/60:.1f} minutes)")
    logger.info(f"Processed {len(all_results)} total potential typosquat pairs")
    logger.info(f"Identified {sum(1 for r in all_results if r.get('is_typosquat', False))} total typosquats")
    logger.info(f"Found {high_confidence_count} high confidence typosquats (>={args.min_confidence})")
    logger.info(f"All results saved to {all_results_path}")
    logger.info(f"Filtered results saved to {output_path}")

if __name__ == "__main__":
    main() 