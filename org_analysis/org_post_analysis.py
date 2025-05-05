import pandas as pd
import google.generativeai as genai
import os
import time
import logging
from pathlib import Path
import json
import glob
from typing import Dict, List, Optional, Tuple
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
DEFAULT_INPUT_DIR = Path(__file__).parent / "results"
DEFAULT_INPUT_FILE = DEFAULT_INPUT_DIR / "typomind_spoofing_results.csv"
DEFAULT_OUTPUT_FILE = DEFAULT_INPUT_DIR / "gemini_filtered_results.csv"
GEMINI_MODEL = "gemini-2.5-flash-preview-04-17"
SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"},
]
API_CALLS_PER_MINUTE = 30  # Rate limit for Gemini API
BATCH_SIZE = 10  # Number of pairs to process in each batch
MAX_RETRIES = 3  # Maximum number of retries for API calls

def setup_gemini_api(api_key: str):
    """Configure Gemini API with the provided key."""
    genai.configure(api_key=api_key)
    logger.info(f"Configured Gemini API with model: {GEMINI_MODEL}")
    return genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        safety_settings=SAFETY_SETTINGS,
        generation_config={"temperature": 0.1}  # Low temperature for more consistent responses
    )

def parse_typomind_raw_outputs(results_dir: Path) -> pd.DataFrame:
    """
    Parse raw Typomind output files and convert them to a structured DataFrame.
    
    Args:
        results_dir: Directory containing Typomind output files (typomind_output_batch_*.txt)
        
    Returns:
        DataFrame with parsed results
    """
    logger.info(f"Searching for raw Typomind output files in {results_dir}")
    
    # Find all Typomind output files
    output_files = list(results_dir.glob("typomind_output_batch_*.txt"))
    
    if not output_files:
        # Try with different patterns
        output_files = list(results_dir.glob("typomind_output*.txt"))
        
    if not output_files:
        logger.warning(f"No Typomind output files found in {results_dir}")
        return pd.DataFrame(columns=["potential_spoof_org", "legitimate_target_org", "detection_info"])
    
    logger.info(f"Found {len(output_files)} Typomind output files to parse")
    
    all_pairs = []
    
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
                    
                    all_pairs.append({
                        "potential_spoof_org": input_org,
                        "legitimate_target_org": target_org,
                        "detection_info": detection_info_str,
                        "timing": timing_str
                    })
                    
                except Exception as e:
                    logger.warning(f"Failed to parse line: {line}. Error: {str(e)}")
        
        except Exception as e:
            logger.error(f"Error parsing file {file_path}: {str(e)}")
    
    logger.info(f"Parsed {len(all_pairs)} potential typosquat pairs from raw output files")
    df = pd.DataFrame(all_pairs)
    
    # Save the parsed data to CSV for future use
    parsed_csv_path = results_dir / "typomind_spoofing_parsed.csv"
    df.to_csv(parsed_csv_path, index=False)
    logger.info(f"Saved parsed typosquat pairs to {parsed_csv_path}")
    
    return df

def analyze_potential_typosquat(
    model, 
    pair: Dict, 
    hf_data: Optional[Dict] = None
) -> Dict:
    """
    Use Gemini to analyze if the potential typosquat is a legitimate threat.
    
    Args:
        model: Gemini model instance
        pair: Dict containing the potential_spoof_org and legitimate_target_org
        hf_data: Optional dict containing additional HuggingFace organization data
    
    Returns:
        Dict with original pair data and Gemini analysis results
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
    
    prompt = f"""
You are a cybersecurity expert analyzing potential typosquatting attacks on Hugging Face organization names.

Analyze this potential typosquatting case:
- Potential Typosquat Org: {potential_spoof}
- Legitimate Target Org: {legitimate_target}
- Detection Method: {detection_info}

Determine if this is a genuine typosquatting attack attempt. Consider:
1. Name similarity and confusion potential (visual, phonetic, semantic)
2. Likelihood of user confusion based on naming patterns
3. Potential malicious intent versus coincidental similarity
4. Organizational size/popularity difference as a potential motive

Respond with a JSON object with the following structure:
{{
  "is_typosquat": true/false,
  "confidence": 0-1 (how confident you are in this assessment),
  "reasoning": "brief explanation of your decision",
  "risk_level": "high/medium/low",
  "recommendation": "brief action recommendation"
}}
"""

    try:
        response = model.generate_content(prompt)
        
        # Extract the JSON response
        resp_text = response.text
        # Remove any markdown formatting if present
        if "```json" in resp_text:
            resp_text = resp_text.split("```json")[1].split("```")[0].strip()
        elif "```" in resp_text:
            resp_text = resp_text.split("```")[1].split("```")[0].strip()
            
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

def process_batch(
    model,
    batch: List[Dict],
    hf_data: Optional[Dict] = None
) -> List[Dict]:
    """Process a batch of potential typosquat pairs."""
    results = []
    for pair in batch:
        # Rate limiting - sleep to avoid hitting API limits
        time.sleep(60 / API_CALLS_PER_MINUTE)
        
        # Try multiple times in case of API errors
        for attempt in range(MAX_RETRIES):
            try:
                result = analyze_potential_typosquat(model, pair, hf_data)
                if result.get("confidence") > 0.95:
                    results.append(result)
                break
            except Exception as e:
                logger.warning(f"Attempt {attempt+1}/{MAX_RETRIES} failed: {str(e)}")
                if attempt == MAX_RETRIES - 1:
                    # Add failed entry with error message on last attempt
                    results.append({
                        "potential_spoof_org": pair["potential_spoof_org"],
                        "legitimate_target_org": pair["legitimate_target_org"],
                        "detection_info": pair.get("detection_info", ""),
                        "is_typosquat": False,
                        "confidence": 0,
                        "reasoning": f"Failed after {MAX_RETRIES} attempts: {str(e)}",
                        "risk_level": "unknown",
                        "recommendation": "Manual review required"
                    })
                time.sleep(2)  # Wait before retry
    
    return results

def load_or_parse_input(input_path: Path, results_dir: Path) -> Tuple[pd.DataFrame, bool]:
    """
    Try to load the input CSV file or parse raw Typomind outputs if CSV not found.
    
    Returns:
        Tuple of (DataFrame, bool) where bool indicates if the data was parsed from raw outputs
    """
    parsed_from_raw = False
    
    try:
        if input_path.exists():
            df = pd.read_csv(input_path)
            logger.info(f"Loaded {len(df)} potential typosquat pairs from {input_path}")
            return df, parsed_from_raw
        else:
            logger.warning(f"Input CSV file not found at {input_path}, will try to parse raw outputs")
    except Exception as e:
        logger.error(f"Error loading input CSV: {str(e)}, will try to parse raw outputs")
    
    # If we get here, we need to parse raw outputs
    df = parse_typomind_raw_outputs(results_dir)
    parsed_from_raw = True
    
    return df, parsed_from_raw

def main():
    parser = argparse.ArgumentParser(description="Filter typosquatting detection results using Gemini API")
    parser.add_argument("--input", type=str, help="Path to input CSV file with Typomind results", 
                        default=str(DEFAULT_INPUT_FILE))
    parser.add_argument("--results-dir", type=str, help="Directory containing Typomind output files",
                        default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--output", type=str, help="Path to output CSV file for filtered results", 
                        default=str(DEFAULT_OUTPUT_FILE))
    parser.add_argument("--hf-data", type=str, help="Path to HuggingFace organizations CSV",
                        default=str(Path(__file__).parent / "huggingface_organizations.csv"))
    parser.add_argument("--api-key", type=str, help="Gemini API key (or set GEMINI_API_KEY environment variable)")
    parser.add_argument("--limit", type=int, help="Limit number of pairs to process (for testing)", default=None)
    parser.add_argument("--parse-raw", action="store_true", help="Force parsing raw Typomind output files")
    parser.add_argument("--min-confidence", type=float, help="Minimum confidence threshold for results", default=0.0)
    
    args = parser.parse_args()
    
    # Get API key from args or environment variable
    api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.error("Gemini API key not provided. Use --api-key or set GEMINI_API_KEY environment variable.")
        return
    
    # Setup paths
    input_path = Path(args.input)
    results_dir = Path(args.results_dir)
    output_path = Path(args.output)
    hf_data_path = Path(args.hf_data)
    
    # Ensure results directory exists
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize Gemini API
    model = setup_gemini_api(api_key)
    
    # Load HuggingFace organization data if available
    hf_data = load_huggingface_data(hf_data_path)
    
    # Always parse raw files to ensure we process all batches
    df = parse_typomind_raw_outputs(results_dir)
    
    if df.empty:
        logger.error("No potential typosquat pairs found. Cannot proceed.")
        return
    
    # Limit number of pairs if requested
    if args.limit and args.limit > 0:
        df = df.head(args.limit)
        logger.info(f"Limited to first {args.limit} pairs for processing")
    
    # Process in batches
    all_results = []
    filtered_results = []
    total_pairs = len(df)
    total_batches = (total_pairs + BATCH_SIZE - 1) // BATCH_SIZE
    
    start_time = time.time()
    logger.info(f"Starting analysis of {total_pairs} potential typosquat pairs in {total_batches} batches")
    
    # Initialize counters
    processed = 0
    confirmed_typosquats = 0
    high_confidence_count = 0
    
    # Process each batch
    for batch_idx in range(total_batches):
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
                    result = analyze_potential_typosquat(model, pair, hf_data)
                    all_results.append(result)  # Always add to all_results
                    
                    # Only add to filtered_results if it meets confidence threshold
                    if result.get("confidence", 0) >= args.min_confidence:
                        filtered_results.append(result)
                        if result.get("is_typosquat", False):
                            high_confidence_count += 1
                    break
                except Exception as e:
                    logger.warning(f"Attempt {attempt+1}/{MAX_RETRIES} failed: {str(e)}")
                    if attempt == MAX_RETRIES - 1:
                        # Add failed entry with error message on last attempt
                        error_result = {
                            "potential_spoof_org": pair["potential_spoof_org"],
                            "legitimate_target_org": pair["legitimate_target_org"],
                            "detection_info": pair.get("detection_info", ""),
                            "is_typosquat": False,
                            "confidence": 0,
                            "reasoning": f"Failed after {MAX_RETRIES} attempts: {str(e)}",
                            "risk_level": "unknown",
                            "recommendation": "Manual review required"
                        }
                        all_results.append(error_result)
                    time.sleep(2)  # Wait before retry
        
        # Update counters
        batch_processed = len(batch_data)
        batch_confirmed = sum(1 for r in all_results[-batch_processed:] if r.get("is_typosquat", False))
        
        processed += batch_processed
        confirmed_typosquats += batch_confirmed
        
        # Save interim results
        all_df = pd.DataFrame(all_results)
        all_path = output_path.with_name(f"{output_path.stem}_all{output_path.suffix}")
        all_df.to_csv(all_path, index=False)
        
        filtered_df = pd.DataFrame(filtered_results)
        if not filtered_df.empty:
            filtered_interim_path = output_path.with_name(f"{output_path.stem}_interim{output_path.suffix}")
            filtered_df.to_csv(filtered_interim_path, index=False)
        
        logger.info(f"Batch {batch_idx+1} complete: {batch_confirmed}/{batch_processed} identified as typosquats")
        logger.info(f"Progress: {processed}/{total_pairs} processed, {confirmed_typosquats} total typosquats identified")
        if args.min_confidence > 0:
            logger.info(f"High confidence typosquats (>={args.min_confidence}): {high_confidence_count}")
        
        # Calculate and log estimated time remaining
        elapsed = time.time() - start_time
        avg_time_per_batch = elapsed / (batch_idx + 1)
        remaining_batches = total_batches - (batch_idx + 1)
        est_remaining_time = avg_time_per_batch * remaining_batches
        
        logger.info(f"Elapsed time: {elapsed:.1f}s. Estimated time remaining: {est_remaining_time:.1f}s " +
                    f"({est_remaining_time/60:.1f} minutes)")
    
    # Save final results - all results
    all_results_df = pd.DataFrame(all_results)
    all_results_path = output_path.with_name(f"{output_path.stem}_all{output_path.suffix}")
    all_results_df.to_csv(all_results_path, index=False)
    
    # Save filtered results if we have a confidence threshold
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
    else:
        # If no results meet the confidence threshold, still sort and save all results
        all_results_df["confidence"] = pd.to_numeric(all_results_df["confidence"], errors="coerce")
        risk_order = {"high": 0, "medium": 1, "low": 2, "unknown": 3}
        all_results_df["risk_order"] = all_results_df["risk_level"].map(lambda x: risk_order.get(x.lower() if isinstance(x, str) else "", 4))
        
        sorted_df = all_results_df.sort_values(
            by=["is_typosquat", "risk_order", "confidence"], 
            ascending=[False, True, False]
        ).drop(columns=["risk_order"])
        
        sorted_df.to_csv(output_path, index=False)
        
        confirmed_df = sorted_df[sorted_df["is_typosquat"] == True].copy()
        confirmed_path = output_path.with_name(f"{output_path.stem}_confirmed{output_path.suffix}")
        confirmed_df.to_csv(confirmed_path, index=False)
    
    # Log final summary
    total_time = time.time() - start_time
    logger.info(f"Analysis complete in {total_time:.1f}s ({total_time/60:.1f} minutes)")
    logger.info(f"Processed {processed} potential typosquat pairs")
    logger.info(f"Identified {confirmed_typosquats} total typosquats")
    if args.min_confidence > 0:
        logger.info(f"Found {high_confidence_count} high confidence typosquats (>={args.min_confidence})")
    logger.info(f"All results saved to {all_results_path}")
    logger.info(f"Filtered results saved to {output_path}")
    logger.info(f"Confirmed typosquats saved to {confirmed_path}")

if __name__ == "__main__":
    main() 