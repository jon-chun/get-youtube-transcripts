import os
import json
import time
import yaml
import logging
import argparse
from pathlib import Path
from datetime import datetime
import yt_dlp

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
                "audio_lang": ["en", "es"],
                "audio_format": ["mp3", "wav"],
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
    audio_formats = config["step2"]["audio_format"]
    audio_kbps_min = config["step2"]["audio_kbps_min"]
    file_prefix = config["step2"]["audio_file_prefix"]
    max_retries = config["step2"]["max_retries"]
    sleep_delay = config["step2"]["sleep_delay"]
    retry_delay = config["step2"]["retry_delay"]
    
    # Try each format in order of preference
    for audio_format in audio_formats:
        # Full output path template
        output_template = str(output_dir / f"{file_prefix}{video_id}.%(ext)s")
        
        # Set up options for yt-dlp
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': output_template,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': audio_format,
                'preferredquality': str(audio_kbps_min),
            }],
            'quiet': True,
            'no_warnings': True,
        }
        
        # Try to download with this format
        for attempt in range(max_retries):
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    logging.info(f"Downloading {audio_format} audio for video: {video_id}")
                    info = ydl.extract_info(video_url, download=True)
                
                # Check if file was created
                expected_file = output_dir / f"{file_prefix}{video_id}.{audio_format}"
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
                logging.warning(f"Attempt {attempt+1}/{max_retries} failed for video {video_id} format {audio_format}: {e}")
            
            if attempt < max_retries - 1:
                logging.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            
        # If we get here, all attempts with this format failed
        logging.warning(f"Failed to download {audio_format} audio for video {video_id} after {max_retries} attempts.")
    
    # If we get here, all formats failed
    logging.error(f"Failed to download audio for video {video_id} in any format.")
    return False, {
        "id": video_id,
        "reason": "Failed to download in any format after all retries",
        "attempted_formats": audio_formats,
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
    sleep_delay = config["step2"]["sleep_delay"]
    audio_formats = config["step2"]["audio_format"]
    file_prefix = config["step2"]["audio_file_prefix"]
    
    # Get report directory
    report_dir = get_report_directory(config)
    output_files = config["step2"]["output_files"]
    
    # Initialize tracking dictionaries
    all_downloads = {}
    successful_downloads = {}
    failed_downloads = {}
    
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
            exists = any((output_dir / f"{file_prefix}{video_id}.{fmt}").exists() for fmt in audio_formats)
            if exists:
                logging.info(f"Audio file for video {video_id} already exists, skipping")
                
                # Find which format exists
                existing_format = next((fmt for fmt in audio_formats if 
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
            
            # Pause to avoid overwhelming the server
            time.sleep(sleep_delay)
    
    # Generate final report
    report_text = generate_report(all_downloads, successful_downloads, failed_downloads)
    save_checkpoint_data(all_downloads, successful_downloads, failed_downloads, 
                        report_text, report_dir, output_files)
    logging.info("Audio download process completed")
    print("\nFinal Report:")
    print(report_text)

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
    args = parser.parse_args()
    
    # Configure logging
    if args.log == "DEBUG":
        logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
    elif args.log == "INFO":
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    elif args.log == "NONE":
        logging.basicConfig(level=logging.CRITICAL, format='%(levelname)s: %(message)s')
    
    # Load configuration
    config = load_config(args.config)
    
    # Get metadata file path
    metadata_file = get_metadata_path(config)
    
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