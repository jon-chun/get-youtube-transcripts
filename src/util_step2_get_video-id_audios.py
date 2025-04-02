import os
import json
import time
import yaml
import logging
import argparse
import random
import requests
from pathlib import Path
from datetime import datetime
import yt_dlp
from fake_useragent import UserAgent  # New import for rotating user agents

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x  # fallback if tqdm is not available

# --- Load Configuration ---
def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logging.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logging.error(f"Error loading configuration from {config_path}: {e}")
        logging.warning("Using default configuration")
        # Return minimal default configuration
        return {
            "step2": {
                "metadata_json_dir": "data/videos_metadata_annalia_20250401",
                "metadata_json_file": "accepted_get_channel_video_ids.json",
                "audio_format_option": ["ydl_opts", ["wav", "m4a"]],
                "audio_lang": ["en", "es"],
                "audio_kbps_min": 128,
                "sleep_delay": 1.0,
                "retry_delay": 5.0,
                "max_retries": 3,
                "audio_output_dir": "data/audio",
                "channel_prefix": "channel_at_",
                "audio_file_prefix": "audio_file_",
                "meta_audio_report_dir": "data/audio/audio_metadata",
                "output_files": {
                    "all_json": "all_audio_downloads.json",
                    "success_json": "successful_audio_downloads.json",
                    "failed_json": "failed_audio_downloads.json",
                    "report": "audio_download_report.txt"
                }
            }
        }

# --- Helper Functions ---
def ensure_directory_exists(directory):
    """Ensure that a directory and all its parent directories exist."""
    os.makedirs(directory, exist_ok=True)
    logging.debug(f"Ensured directory exists: {directory}")

def get_metadata_path(config):
    """Get the full path to the metadata JSON file."""
    base_dir = Path(os.path.dirname(os.path.abspath(__file__))).parent
    metadata_dir = base_dir / config["step2"]["metadata_json_dir"]
    metadata_file = metadata_dir / config["step2"]["metadata_json_file"]
    return metadata_file

def get_report_directory(config):
    """Get the directory for saving reports and metadata."""
    base_dir = Path(os.path.dirname(os.path.abspath(__file__))).parent
    report_dir = base_dir / config["step2"]["meta_audio_report_dir"]
    ensure_directory_exists(report_dir)
    return report_dir

def get_output_directory(config, channel_id):
    """Get the output directory for a specific channel."""
    base_dir = Path(os.path.dirname(os.path.abspath(__file__))).parent
    audio_dir = base_dir / config["step2"]["audio_output_dir"]
    
    # Sanitize channel_id to create a valid directory name
    channel_prefix = config["step2"]["channel_prefix"]
    channel_dir_name = f"{channel_prefix}{channel_id.lstrip('@')}"
    
    channel_dir = audio_dir / channel_dir_name
    ensure_directory_exists(channel_dir)
    return channel_dir

def generate_report(all_downloads, successful_downloads, failed_downloads):
    """
    Generate a human-readable report summarizing the audio download process.
    
    Args:
        all_downloads (dict): {channel_id: {video_id: metadata, ...}, ...}
        successful_downloads (dict): {channel_id: {video_id: metadata, ...}, ...}
        failed_downloads (dict): {channel_id: {video_id: metadata, ...}, ...}
        
    Returns:
        str: Human-readable report text.
    """
    lines = []
    total_channels = len(all_downloads)
    lines.append(f"Audio Download Report - {datetime.now().isoformat()}")
    lines.append(f"Total channels processed: {total_channels}\n")
    total_all = total_success = total_failed = 0

    for chan in all_downloads:
        count_all = len(all_downloads.get(chan, {}))
        count_success = len(successful_downloads.get(chan, {}))
        count_failed = len(failed_downloads.get(chan, {}))
        total_all += count_all
        total_success += count_success
        total_failed += count_failed
        lines.append(f"Channel {chan}: Total Videos Processed: {count_all}, Successful: {count_success}, Failed: {count_failed}")
    
    lines.append(f"\nGrand Totals - Processed: {total_all}, Successfully Downloaded: {total_success}, Failed/Skipped: {total_failed}")
    
    # Add summary of formats and failures
    if total_success > 0:
        lines.append("\nDownloaded Audio Formats:")
        format_counts = {}
        for chan in successful_downloads:
            for video_id, metadata in successful_downloads[chan].items():
                fmt = metadata.get("format", "unknown")
                format_counts[fmt] = format_counts.get(fmt, 0) + 1
        
        for fmt, count in format_counts.items():
            lines.append(f"  {fmt}: {count} files")
    
    if total_failed > 0:
        lines.append("\nFailure Reasons:")
        reason_counts = {}
        for chan in failed_downloads:
            for video_id, metadata in failed_downloads[chan].items():
                reason = metadata.get("reason", "unknown")
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
        
        for reason, count in reason_counts.items():
            lines.append(f"  {reason}: {count} files")
    
    return "\n".join(lines)

def save_checkpoint_data(all_downloads, successful_downloads, failed_downloads, report_text, report_dir, output_files):
    """
    Save checkpoint data to JSON files and a human-readable report.
    
    Args:
        all_downloads (dict): Dictionary of all processed videos
        successful_downloads (dict): Dictionary of successfully downloaded videos
        failed_downloads (dict): Dictionary of failed/skipped videos
        report_text (str): Human-readable report text
        report_dir (Path): Directory to save reports
        output_files (dict): Dictionary with output filenames
    """
    ensure_directory_exists(report_dir)
    
    def save_json(filename, data):
        path = report_dir / filename
        try:
            with open(path, 'w') as f:
                json.dump(data, f, indent=4)
            logging.info(f"Saved {filename}")
        except Exception as e:
            logging.error(f"Error saving {filename}: {e}")
    
    save_json(output_files["all_json"], all_downloads)
    save_json(output_files["success_json"], successful_downloads)
    save_json(output_files["failed_json"], failed_downloads)
    
    report_path = report_dir / output_files["report"]
    try:
        with open(report_path, 'w') as f:
            f.write(report_text)
        logging.info(f"Saved report to {output_files['report']}")
    except Exception as e:
        logging.error(f"Error saving report: {e}")

# --- New Helper Functions ---
def get_random_user_agent():
    """Generate a random user agent string."""
    try:
        ua = UserAgent()
        return ua.random
    except Exception as e:
        logging.warning(f"Error generating random user agent: {e}. Using fallback.")
        # Fallback user agents if fake_useragent fails
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.2 Safari/605.1.15',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/109.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:109.0) Gecko/20100101 Firefox/109.0'
        ]
        return random.choice(user_agents)

def humanized_sleep(min_seconds=0.5, max_seconds=2.0):
    """
    Sleep for a random amount of time that follows a more human-like pattern.
    
    Args:
        min_seconds (float): Minimum sleep time in seconds
        max_seconds (float): Maximum sleep time in seconds
    """
    # Generate a random sleep time following a power law distribution
    # This is more human-like than uniform distribution
    alpha = 1.5  # Power law exponent
    raw = random.random()
    
    # Apply power law and scale to our desired range
    sleep_time = min_seconds + (max_seconds - min_seconds) * (raw ** alpha)
    
    logging.debug(f"Sleeping for {sleep_time:.2f} seconds")
    time.sleep(sleep_time)

def calculate_backoff_delay(attempt, initial_delay=5.0, max_delay=120.0):
    """
    Calculate exponential backoff delay.
    
    Args:
        attempt (int): The current attempt number (0-indexed)
        initial_delay (float): Initial delay in seconds
        max_delay (float): Maximum delay in seconds
        
    Returns:
        float: Delay time in seconds
    """
    # Exponential backoff with jitter
    delay = min(initial_delay * (2 ** attempt), max_delay)
    # Add jitter (±25%)
    jitter = random.uniform(-0.25 * delay, 0.25 * delay)
    return max(initial_delay, delay + jitter)

def download_audio(video_id, output_dir, config):
    """
    Download audio for a YouTube video.
    
    Args:
        video_id (str): YouTube video ID
        output_dir (Path): Directory to save the audio file
        config (dict): Configuration dictionary
    
    Returns:
        tuple: (bool, dict) - (True, metadata) if successful, (False, error_info) if failed
    """
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    audio_format_option = config["step2"]["audio_format_option"][0]
    formats_to_try = config["step2"]["audio_format_option"][1]
    audio_kbps_min = config["step2"]["audio_kbps_min"]
    file_prefix = config["step2"]["audio_file_prefix"]
    max_retries = config["step2"]["max_retries"]
    retry_delay = config["step2"]["retry_delay"]
    
    # Check if any output file already exists (regardless of format)
    for fmt in formats_to_try:
        expected_file = output_dir / f"{file_prefix}{video_id}.{fmt}"
        if expected_file.exists():
            logging.info(f"File already exists for video {video_id}: {expected_file}")
            return True, {
                "id": video_id,
                "format": fmt,
                "quality": audio_kbps_min,
                "download_date": "pre-existing",
                "file_path": str(expected_file)
            }
    
    # Different approaches based on the audio_format_option
    if audio_format_option == "ydl_opts":
        # Direct approach - use yt-dlp's format selection to get audio directly
        for audio_format in formats_to_try:
            # Full output path template
            output_template = str(output_dir / f"{file_prefix}{video_id}.%(ext)s")
            
            # Try to download with this format
            for attempt in range(max_retries):
                # Get a random user agent for this attempt
                user_agent = get_random_user_agent()
                
                # Set up options to directly download audio in specified format
                ydl_opts = {
                    'format': f'bestaudio[ext={audio_format}]/bestaudio',
                    'outtmpl': output_template,
                    'quiet': True,
                    'no_warnings': True,
                    'user_agent': user_agent,
                    'http_headers': {'User-Agent': user_agent},
                    'socket_timeout': 30,  # Increase socket timeout
                    'retries': 3,          # Internal yt-dlp retries
                    'fragment_retries': 3, # Fragment download retries
                }
                
                try:
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        logging.info(f"Downloading {audio_format} audio for video: {video_id} (Attempt {attempt+1}/{max_retries})")
                        info = ydl.extract_info(video_url, download=True)
                        
                        # Determine the actual file extension from the downloaded file
                        downloaded_ext = info.get('ext', audio_format)
                        actual_file = output_dir / f"{file_prefix}{video_id}.{downloaded_ext}"
                        
                        if actual_file.exists():
                            logging.info(f"Successfully downloaded {downloaded_ext} audio for video: {video_id}")
                            # Return success metadata
                            return True, {
                                "id": video_id,
                                "format": downloaded_ext,
                                "quality": info.get('abr', audio_kbps_min),
                                "download_date": datetime.now().strftime("%Y%m%d"),
                                "file_path": str(actual_file),
                                "title": info.get("title", ""),
                                "duration": info.get("duration", "")
                            }
                        else:
                            logging.warning(f"Download completed but file not found: {actual_file}")
                            
                except Exception as e:
                    logging.warning(f"Attempt {attempt+1}/{max_retries} failed for video {video_id} format {audio_format}: {str(e)}")
                
                if attempt < max_retries - 1:
                    # Calculate backoff delay for next retry
                    backoff_time = calculate_backoff_delay(attempt, initial_delay=retry_delay)
                    logging.info(f"Retrying in {backoff_time:.2f} seconds...")
                    time.sleep(backoff_time)
                
            # If we get here, all attempts with this format failed
            logging.warning(f"Failed to download {audio_format} audio for video {video_id} after {max_retries} attempts.")
    
    elif audio_format_option == "alt_formats":
        # Alternative approach - explicitly try each format
        for audio_format in formats_to_try:
            # Full output path template
            output_template = str(output_dir / f"{file_prefix}{video_id}.{audio_format}")
            
            # Try to download with this format
            for attempt in range(max_retries):
                # Get a random user agent for this attempt
                user_agent = get_random_user_agent()
                
                # Set up options to get best audio and specify output format
                ydl_opts = {
                    'format': 'bestaudio',
                    'outtmpl': output_template,
                    'quiet': True,
                    'no_warnings': True,
                    'user_agent': user_agent,
                    'http_headers': {'User-Agent': user_agent},
                    'socket_timeout': 30,
                    'retries': 3,
                    'fragment_retries': 3,
                }
                
                try:
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        logging.info(f"Downloading {audio_format} audio for video: {video_id} (Attempt {attempt+1}/{max_retries})")
                        info = ydl.extract_info(video_url, download=True)
                    
                    # Check if file was created
                    expected_file = Path(output_template)
                    if expected_file.exists():
                        logging.info(f"Successfully downloaded {audio_format} audio for video: {video_id}")
                        # Return success metadata
                        return True, {
                            "id": video_id,
                            "format": audio_format,
                            "quality": audio_kbps_min,
                            "download_date": datetime.now().strftime("%Y%m%d"),
                            "file_path": str(expected_file),
                            "title": info.get("title", ""),
                            "duration": info.get("duration", "")
                        }
                    else:
                        logging.warning(f"Download completed but file not found: {expected_file}")
                        
                except Exception as e:
                    logging.warning(f"Attempt {attempt+1}/{max_retries} failed for video {video_id} format {audio_format}: {str(e)}")
                
                if attempt < max_retries - 1:
                    # Calculate backoff delay for next retry
                    backoff_time = calculate_backoff_delay(attempt, initial_delay=retry_delay)
                    logging.info(f"Retrying in {backoff_time:.2f} seconds...")
                    time.sleep(backoff_time)
                
            # If we get here, all attempts with this format failed
            logging.warning(f"Failed to download {audio_format} audio for video {video_id} after {max_retries} attempts.")
    
    else:
        logging.error(f"Unknown audio format option: {audio_format_option}")
        return False, {
            "id": video_id,
            "reason": f"Unknown audio format option: {audio_format_option}",
            "date": datetime.now().strftime("%Y%m%d")
        }
    
    # If we get here, all formats failed
    logging.error(f"Failed to download audio for video {video_id} in any format.")
    return False, {
        "id": video_id,
        "reason": "Failed to download in any format after all retries",
        "attempted_formats": formats_to_try,
        "date": datetime.now().strftime("%Y%m%d")
    }

def process_videos(metadata, config):
    """
    Process videos from metadata and download audio.
    
    Args:
        metadata (dict): Dictionary containing video metadata
        config (dict): Configuration dictionary
    """
    # Get allowed languages
    allowed_langs = config["step2"]["audio_lang"]
    sleep_delay_min = max(0.5, config["step2"]["sleep_delay"] * 0.5)
    sleep_delay_max = config["step2"]["sleep_delay"] * 1.5
    
    # This is the proper way to access the format options now
    audio_format_option = config["step2"]["audio_format_option"][0]
    formats_to_try = config["step2"]["audio_format_option"][1]
    
    file_prefix = config["step2"]["audio_file_prefix"]
    
    # Get report directory
    report_dir = get_report_directory(config)
    output_files = config["step2"]["output_files"]
    
    # Initialize tracking dictionaries
    all_downloads = {}
    successful_downloads = {}
    failed_downloads = {}
    
    # Create a lock file to indicate the script is running
    lock_file = report_dir / "process_running.lock"
    
    try:
        # Load existing checkpoint data if available
        try:
            if (report_dir / output_files["all_json"]).exists():
                with open(report_dir / output_files["all_json"], 'r') as f:
                    all_downloads = json.load(f)
                
                with open(report_dir / output_files["success_json"], 'r') as f:
                    successful_downloads = json.load(f)
                
                with open(report_dir / output_files["failed_json"], 'r') as f:
                    failed_downloads = json.load(f)
                
                logging.info("Loaded existing checkpoint data")
        except Exception as e:
            logging.warning(f"Failed to load checkpoint data, starting fresh: {e}")
        
        # Create lock file
        with open(lock_file, 'w') as f:
            f.write(f"Process started at {datetime.now().isoformat()}")
        
        # Process each channel
        for channel_id, videos in tqdm(metadata.items(), desc="Processing channels"):
            logging.info(f"Processing channel: {channel_id}")
            output_dir = get_output_directory(config, channel_id)
            
            # Ensure channel entries exist in tracking dictionaries
            if channel_id not in all_downloads:
                all_downloads[channel_id] = {}
            if channel_id not in successful_downloads:
                successful_downloads[channel_id] = {}
            if channel_id not in failed_downloads:
                failed_downloads[channel_id] = {}
            
            # Process each video in the channel
            for video_id, video_meta in tqdm(videos.items(), desc=f"Channel {channel_id}", unit="video"):
                # Skip if already processed
                if video_id in all_downloads[channel_id]:
                    logging.info(f"Skipping already processed video: {video_id}")
                    continue
                
                # Check if video language is allowed
                video_lang = video_meta.get("language", "").lower()
                if video_lang not in [lang.lower() for lang in allowed_langs]:
                    logging.info(f"Skipping video {video_id} due to language: {video_lang}")
                    
                    # Add to failed downloads with reason
                    failed_downloads[channel_id][video_id] = {
                        "id": video_id,
                        "reason": f"Language not supported: {video_lang}",
                        "title": video_meta.get("title", ""),
                        "duration": video_meta.get("duration", "")
                    }
                    
                    # Add to all downloads
                    all_downloads[channel_id][video_id] = {
                        "id": video_id,
                        "status": "skipped",
                        "reason": f"Language not supported: {video_lang}"
                    }
                    
                    continue
                
                # Check if audio file already exists in any of the allowed formats
                exists = any((output_dir / f"{file_prefix}{video_id}.{fmt}").exists() for fmt in formats_to_try)
                if exists:
                    logging.info(f"Audio file for video {video_id} already exists, skipping")
                    
                    # Find which format exists
                    existing_format = next((fmt for fmt in formats_to_try if 
                                          (output_dir / f"{file_prefix}{video_id}.{fmt}").exists()), None)
                    
                    # Add to successful downloads
                    successful_downloads[channel_id][video_id] = {
                        "id": video_id,
                        "format": existing_format,
                        "quality": config["step2"]["audio_kbps_min"],
                        "download_date": "pre-existing",
                        "file_path": str(output_dir / f"{file_prefix}{video_id}.{existing_format}"),
                        "title": video_meta.get("title", ""),
                        "duration": video_meta.get("duration", "")
                    }
                    
                    # Add to all downloads
                    all_downloads[channel_id][video_id] = {
                        "id": video_id,
                        "status": "success",
                        "reason": "pre-existing"
                    }
                    
                    continue
                
                # Download audio
                success, result = download_audio(video_id, output_dir, config)
                
                # Update metadata
                result.update({
                    "title": video_meta.get("title", ""),
                    "original_duration": video_meta.get("duration", ""),
                    "upload_date": video_meta.get("upload_date", "")
                })
                
                if success:
                    successful_downloads[channel_id][video_id] = result
                    all_downloads[channel_id][video_id] = {
                        "id": video_id,
                        "status": "success",
                        "format": result.get("format", "")
                    }
                else:
                    failed_downloads[channel_id][video_id] = result
                    all_downloads[channel_id][video_id] = {
                        "id": video_id,
                        "status": "failed",
                        "reason": result.get("reason", "unknown")
                    }
                
                # Save checkpoint every 5 videos
                if len(all_downloads[channel_id]) % 5 == 0:
                    report_text = generate_report(all_downloads, successful_downloads, failed_downloads)
                    save_checkpoint_data(all_downloads, successful_downloads, failed_downloads, 
                                        report_text, report_dir, output_files)
                    logging.info(f"Checkpoint saved for channel {channel_id}")
                
                # Use humanized sleep between requests
                humanized_sleep(sleep_delay_min, sleep_delay_max)
        
        # Generate final report
        report_text = generate_report(all_downloads, successful_downloads, failed_downloads)
        save_checkpoint_data(all_downloads, successful_downloads, failed_downloads, 
                            report_text, report_dir, output_files)
        logging.info("Audio download process completed")
        print("\nFinal Report:")
        print(report_text)
        
    finally:
        # Always remove the lock file when done
        if lock_file.exists():
            try:
                os.remove(lock_file)
                logging.info("Removed process lock file")
            except Exception as e:
                logging.error(f"Error removing lock file: {e}")

# --- Main Function ---
def main():
    parser = argparse.ArgumentParser(
        description="Download YouTube audio from videos in metadata JSON."
    )
    parser.add_argument(
        "--log", type=str, choices=["INFO", "DEBUG", "NONE"], default="INFO",
        help="Set logging level (INFO, DEBUG, NONE)."
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to configuration YAML file."
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from checkpoint if available."
    )
    args = parser.parse_args()
    
    # Configure logging
    if args.log == "DEBUG":
        logging.basicConfig(level=logging.DEBUG, 
                           format='%(asctime)s - %(levelname)s - %(message)s',
                           datefmt='%Y-%m-%d %H:%M:%S')
    elif args.log == "INFO":
        logging.basicConfig(level=logging.INFO, 
                           format='%(asctime)s - %(levelname)s - %(message)s',
                           datefmt='%Y-%m-%d %H:%M:%S')
    elif args.log == "NONE":
        logging.basicConfig(level=logging.CRITICAL, 
                           format='%(asctime)s - %(levelname)s - %(message)s',
                           datefmt='%Y-%m-%d %H:%M:%S')
    
    # Load configuration
    config = load_config(args.config)
    
    # Get metadata file path
    metadata_file = get_metadata_path(config)
    
    # Check for existing lock file
    report_dir = get_report_directory(config)
    lock_file = report_dir / "process_running.lock"
    
    if lock_file.exists() and not args.resume:
        logging.error("Lock file exists, another instance may be running.")
        print("A lock file exists, which indicates another instance of this script may be running.")
        print("If you're sure no other instance is running, you can:")
        print("  1. Use --resume to continue from the last checkpoint")
        print("  2. Delete the lock file at:", lock_file)
        return
    
    # Load metadata JSON
    try:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        logging.info(f"Loaded metadata from {metadata_file}")
    except Exception as e:
        logging.error(f"Error loading metadata from {metadata_file}: {e}")
        return
    
    # Process videos
    process_videos(metadata, config)
    logging.info("Audio download process completed.")

if __name__ == "__main__":
    main()