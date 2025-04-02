import yt_dlp
import json
import os
import logging
import argparse
import time
import yaml
from datetime import datetime

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x  # fallback if tqdm is not available

# --- Load Configuration ---
def load_config(config_path="config.yaml"):
    """
    Load configuration from YAML file.
    
    Args:
        config_path (str): Path to configuration YAML file.
        
    Returns:
        dict: Configuration dictionary.
    """
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
            "step1": {
                "channels": ["@ColumbusCatholic/videos"],
                "channel_default_suffix": "/videos",
                "checkpoint_video_count": 5,
                "sleep_delay": 1.0,
                "video_length_min": 300,
                "video_length_max": 7200,
                "video_date_after": "20000101",
                "video_date_before": "20250401",
                "video_language": ["en"],
                "video_views_min": 20,
                "video_views_max": 1000000,
                "video_upvotes_min": 0,
                "video_upvotes_max": 1000000,
                "output_files": {
                    "all_json": "all_get_channel_video_ids.json",
                    "accepted_json": "accepted_get_channel_video_ids.json",
                    "rejected_json": "rejected_get_channel_video_ids.json",
                    "report": "report_get_channel_video_ids.txt"
                }
            }
        }

# --- Helper Functions ---

def parse_channel_list(channels_csv, channel_default_suffix):
    """
    Parse a list of YouTube channels from CSV content.
    
    Args:
        channels_csv (str): String content of the CSV file with channels.
        channel_default_suffix (str): Default suffix to append if none exists.
        
    Returns:
        list: List of properly formatted channel URLs.
    """
    channels = []
    for line in channels_csv.splitlines():
        # Skip blank lines
        line = line.strip()
        if not line:
            continue
        
        # Apply default suffix if none exists
        if not any(suffix in line for suffix in ['/videos', '/streams', '/playlists', '/featured']):
            line = line + channel_default_suffix
            
        channels.append(line)
    
    return channels

def ensure_directory_exists(directory):
    """
    Ensure that a directory and all its parent directories exist.
    
    Args:
        directory (str): The directory path to ensure exists.
    """
    os.makedirs(directory, exist_ok=True)
    logging.debug(f"Ensured directory exists: {directory}")

def get_output_directory():
    """
    Determine the output directory relative to the parent directory of the script.
    Creates the directory if it doesn't exist.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(base_dir, "..", "data", "videos_metadata")
    ensure_directory_exists(out_dir)
    return out_dir

def load_checkpoint_data(out_dir, output_files):
    """
    Load checkpoint data from output files if they exist.
    
    Args:
        out_dir (str): Directory containing checkpoint files.
        output_files (dict): Dictionary with output filenames.
        
    Returns:
        all_videos (dict): {channel_id: {video_id: metadata, ...}, ...}
        accepted_videos (dict): {channel_id: {video_id: metadata, ...}, ...}
        rejected_videos (dict): {channel_id: {video_id: metadata, ...}, ...}
    """
    def load_json(filename):
        path = os.path.join(out_dir, filename)
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logging.error(f"Error loading {filename}: {e}")
        return {}
    
    all_videos = load_json(output_files["all_json"])
    accepted_videos = load_json(output_files["accepted_json"])
    rejected_videos = load_json(output_files["rejected_json"])
    
    return all_videos, accepted_videos, rejected_videos

def save_checkpoint_data(all_videos, accepted_videos, rejected_videos, report_text, out_dir, output_files):
    """
    Save checkpoint data to JSON files and a human-readable report.
    Ensures the output directory exists before saving.
    
    Args:
        all_videos (dict): Dictionary of all processed videos.
        accepted_videos (dict): Dictionary of accepted videos.
        rejected_videos (dict): Dictionary of rejected videos.
        report_text (str): Human-readable report text.
        out_dir (str): Directory to save output files.
        output_files (dict): Dictionary with output filenames.
    """
    ensure_directory_exists(out_dir)
    
    def save_json(filename, data):
        path = os.path.join(out_dir, filename)
        try:
            with open(path, 'w') as f:
                json.dump(data, f, indent=4)
            logging.info(f"Saved {filename}")
        except Exception as e:
            logging.error(f"Error saving {filename}: {e}")
    
    save_json(output_files["all_json"], all_videos)
    save_json(output_files["accepted_json"], accepted_videos)
    save_json(output_files["rejected_json"], rejected_videos)
    
    report_path = os.path.join(out_dir, output_files["report"])
    try:
        with open(report_path, 'w') as f:
            f.write(report_text)
        logging.info(f"Saved report to {output_files['report']}")
    except Exception as e:
        logging.error(f"Error saving report: {e}")

def generate_report(all_videos, accepted_videos, rejected_videos):
    """
    Generate a human-readable report summarizing the extraction.
    
    Args:
        all_videos (dict): Dictionary of all processed videos.
        accepted_videos (dict): Dictionary of accepted videos.
        rejected_videos (dict): Dictionary of rejected videos.
        
    Returns:
        str: Human-readable report text.
    """
    lines = []
    total_channels = len(all_videos)
    lines.append(f"Extraction Report - {datetime.now().isoformat()}")
    lines.append(f"Total channels processed: {total_channels}\n")
    total_all = total_acc = total_rej = 0

    for chan in all_videos:
        count_all = len(all_videos.get(chan, {}))
        count_acc = len(accepted_videos.get(chan, {}))
        count_rej = len(rejected_videos.get(chan, {}))
        total_all += count_all
        total_acc += count_acc
        total_rej += count_rej
        lines.append(f"Channel {chan}: Total Videos Processed: {count_all}, Accepted: {count_acc}, Rejected: {count_rej}")
    
    lines.append(f"\nGrand Totals - Processed: {total_all}, Accepted: {total_acc}, Rejected: {total_rej}")
    return "\n".join(lines)

def get_video_data_from_info(video_info):
    """
    Extract a subset of metadata keys from video_info, using empty string as a filler for missing values.
    
    Args:
        video_info (dict): Full video metadata.
        
    Returns:
        dict: Simplified video metadata.
    """
    keys = ["id", "title", "upload_date", "duration", "view_count", "like_count", "language"]
    data = {}
    for key in keys:
        value = video_info.get(key)
        data[key] = value if value is not None else ""
    return data

def meets_criteria(video, config):
    """
    Check if the video's metadata meets all filtering criteria from configuration.
    
    Args:
        video (dict): Metadata for a single video.
        config (dict): Configuration with filtering criteria.
        
    Returns:
        tuple: (bool, str) - (True, None) if video meets all criteria, 
                            (False, reason) with rejection reason otherwise.
    """
    # Get filtering criteria from config
    video_length_min = config["video_length_min"]
    video_length_max = config["video_length_max"]
    video_date_after = config["video_date_after"]
    video_date_before = config["video_date_before"]
    video_language = config["video_language"]
    video_views_min = config["video_views_min"]
    video_views_max = config["video_views_max"]
    video_upvotes_min = config["video_upvotes_min"]
    video_upvotes_max = config["video_upvotes_max"]
    
    # Duration check (in seconds)
    duration = video.get('duration')
    if duration is None or duration < video_length_min:
        reason = f"duration too short: {duration}s < {video_length_min}s"
        logging.debug(f"Video {video.get('id', 'unknown')} rejected: {reason}")
        return False, reason
    if duration > video_length_max:
        reason = f"duration too long: {duration}s > {video_length_max}s"
        logging.debug(f"Video {video.get('id', 'unknown')} rejected: {reason}")
        return False, reason

    # Upload date check (YYYYMMDD; lexicographical compare works for valid date strings)
    upload_date = video.get('upload_date')
    if upload_date is None or upload_date <= video_date_after:
        reason = f"upload date too early: {upload_date} <= {video_date_after}"
        logging.debug(f"Video {video.get('id', 'unknown')} rejected: {reason}")
        return False, reason
    if upload_date >= video_date_before:
        reason = f"upload date too late: {upload_date} >= {video_date_before}"
        logging.debug(f"Video {video.get('id', 'unknown')} rejected: {reason}")
        return False, reason

    # Language check
    language = video.get('language')
    if language is None or language.lower() not in [l.lower() for l in video_language]:
        reason = f"language not supported: {language}"
        logging.debug(f"Video {video.get('id', 'unknown')} rejected: {reason}")
        return False, reason

    # View count check
    view_count = video.get('view_count')
    if view_count is None or view_count < video_views_min:
        reason = f"too few views: {view_count} < {video_views_min}"
        logging.debug(f"Video {video.get('id', 'unknown')} rejected: {reason}")
        return False, reason
    if view_count > video_views_max:
        reason = f"too many views: {view_count} > {video_views_max}"
        logging.debug(f"Video {video.get('id', 'unknown')} rejected: {reason}")
        return False, reason

    # Like count check
    like_count = video.get('like_count')
    # Convert None to 0 if minimum upvotes is 0
    if like_count is None or like_count == 'None':
        like_count = 0
    if like_count < video_upvotes_min:
        reason = f"too few likes: {like_count} < {video_upvotes_min}"
        logging.debug(f"Video {video.get('id', 'unknown')} rejected: {reason}")
        return False, reason
    if like_count > video_upvotes_max:
        reason = f"too many likes: {like_count} > {video_upvotes_max}"
        logging.debug(f"Video {video.get('id', 'unknown')} rejected: {reason}")
        return False, reason

    return True, None

def get_video_metadata(video_id):
    """
    Given a video ID, extract full metadata using yt_dlp.
    
    Args:
        video_id (str): The YouTube video ID.
        
    Returns:
        dict: The video's metadata, or None if extraction fails.
    """
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    try:
        with yt_dlp.YoutubeDL({'quiet': True, 'skip_download': True}) as ydl:
            video_info = ydl.extract_info(video_url, download=False)
            return video_info
    except yt_dlp.utils.DownloadError as e:
        logging.error(f"Download error extracting video {video_id}: {e}")
    except Exception as e:
        logging.error(f"Unexpected error extracting video {video_id}: {e}")
    return None

def has_channel_been_scraped(channel_id, all_videos):
    """
    Determine if a channel has already been fully scraped.
    
    Args:
        channel_id (str): The YouTube channel ID.
        all_videos (dict): Dictionary containing all processed videos.
        
    Returns:
        bool: True if the channel exists and has videos, False otherwise.
    """
    return channel_id in all_videos and len(all_videos[channel_id]) > 0

# --- Main Extraction Function ---

def get_yt_channel_videos(channels, out_dir, config):
    """
    Fetch video IDs from channels, extract metadata, filter videos, and save checkpoints.
    
    Args:
        channels (list): List of YouTube channel URLs or shorthand.
        out_dir (str): Output directory for checkpoint files.
        config (dict): Configuration dictionary.
    """
    # Extract config values
    step1_config = config["step1"]
    channel_default_suffix = step1_config["channel_default_suffix"]
    checkpoint_video_count = step1_config["checkpoint_video_count"]
    sleep_delay = step1_config["sleep_delay"]
    output_files = step1_config["output_files"]
    
    # Ensure output directory exists
    ensure_directory_exists(out_dir)
    
    # Load existing checkpoint data if available.
    all_videos, accepted_videos, rejected_videos = load_checkpoint_data(out_dir, output_files)

    # Process each channel.
    for original_channel in channels:
        logging.info(f"Processing channel: {original_channel}")
        try:
            # Convert shorthand to full URL if needed.
            channel_url = original_channel
            if not channel_url.startswith("http"):
                channel_url = "https://www.youtube.com/" + channel_url
                logging.debug(f"Converted shorthand to full URL: {channel_url}")

            # Extract channel ID from the URL
            # Expected formats:
            # https://www.youtube.com/@ChannelName/videos
            # https://www.youtube.com/@ChannelName/streams
            # https://www.youtube.com/@ChannelName/playlists
            parts = channel_url.split('/')
            
            # Channel ID is always after youtube.com/ but before any suffix
            channel_id = parts[3] if len(parts) > 3 else channel_url
            logging.info(f"Extracted channel ID: {channel_id}")
            
            # Check if the channel has already been scraped
            if has_channel_been_scraped(channel_id, all_videos):
                logging.info(f"Skipping channel {channel_id} as it has already been scraped.")
                continue
            
            # Ensure we have a URL that will list videos
            if not any(suffix in channel_url for suffix in ['/videos', '/streams', '/playlists', '/featured']):
                channel_url = channel_url + channel_default_suffix
                logging.debug(f"Added default suffix: {channel_url}")

            # Ensure channel entries exist in checkpoint dictionaries.
            if channel_id not in all_videos:
                all_videos[channel_id] = {}
            if channel_id not in accepted_videos:
                accepted_videos[channel_id] = {}
            if channel_id not in rejected_videos:
                rejected_videos[channel_id] = {}

            # Use flat extraction to retrieve list of video entries.
            ydl_opts = {
                'quiet': True,
                'extract_flat': True,
                'skip_download': True,
                'forcejson': False,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                logging.debug(f"Extracting flat info for channel: {channel_url}")
                info = ydl.extract_info(channel_url, download=False)

            if not info or 'entries' not in info:
                logging.warning(f"Could not retrieve video list for {channel_url}. Skipping channel.")
                continue

            video_entries = info['entries']
            total_videos = len(video_entries)
            processed_ct = 0

            # Process each video with a progress bar.
            for entry in tqdm(video_entries, desc=f"Channel {channel_id}", unit="video"):
                if entry is None or 'id' not in entry:
                    continue
                video_id = entry['id']

                # Skip if already processed.
                if video_id in all_videos[channel_id]:
                    logging.info(f"Skipping already processed video: {video_id}")
                    continue

                logging.info(f"Processing video: {video_id}")
                video_info = get_video_metadata(video_id)
                if not video_info:
                    logging.warning(f"Skipping video {video_id} due to metadata extraction failure.")
                    continue

                # Provide terminal feedback: video title and duration.
                title = video_info.get('title', '')
                duration = video_info.get('duration', '')
                logging.info(f"Video {video_id}: '{title}' with duration {duration}s")

                # Pause to avoid overwhelming the server.
                time.sleep(sleep_delay)

                # Prepare metadata dictionary with fallback for missing values.
                metadata = get_video_data_from_info(video_info)
                all_videos[channel_id][video_id] = metadata

                # Filter based on criteria.
                meets, reason = meets_criteria(video_info, step1_config)
                if meets:
                    accepted_videos[channel_id][video_id] = metadata
                    logging.info(f"Accepted video {video_id}")
                else:
                    # Add rejection reason to metadata
                    metadata["reason"] = reason
                    rejected_videos[channel_id][video_id] = metadata
                    logging.info(f"Rejected video {video_id}: {reason}")

                processed_ct += 1

                # Save checkpoint every checkpoint_video_count videos.
                if processed_ct % checkpoint_video_count == 0:
                    report_text = generate_report(all_videos, accepted_videos, rejected_videos)
                    save_checkpoint_data(all_videos, accepted_videos, rejected_videos, report_text, out_dir, output_files)
                    logging.info(f"Checkpoint saved after processing {processed_ct} videos in channel {channel_id}")

            # End of channel processing; save checkpoint.
            report_text = generate_report(all_videos, accepted_videos, rejected_videos)
            save_checkpoint_data(all_videos, accepted_videos, rejected_videos, report_text, out_dir, output_files)
            logging.info(f"Finished processing channel {channel_id}. Total videos processed: {len(all_videos[channel_id])}")

        except yt_dlp.utils.DownloadError as e:
            logging.error(f"Download error for channel {original_channel}: {e}")
        except Exception as e:
            logging.error(f"Unexpected error processing channel {original_channel}: {e}")

    # Final report generation.
    final_report = generate_report(all_videos, accepted_videos, rejected_videos)
    save_checkpoint_data(all_videos, accepted_videos, rejected_videos, final_report, out_dir, output_files)
    logging.info("Completed extraction and filtering for all channels.")
    print("\nFinal Report:")
    print(final_report)

# --- Main Function ---

def main():
    parser = argparse.ArgumentParser(
        description="Fetch YouTube channel video metadata, filter by criteria, and save restartable checkpoints."
    )
    parser.add_argument(
        "--log", type=str, choices=["INFO", "DEBUG", "NONE"], default="INFO",
        help="Set logging level (INFO, DEBUG, NONE)."
    )
    parser.add_argument(
        "--channels", type=str, nargs='+',
        help="List of YouTube channel URLs or shorthand (with /videos appended)."
    )
    parser.add_argument(
        "--channels_csv", type=str,
        help="Path to CSV file containing YouTube channel URLs."
    )
    parser.add_argument(
        "--channels_csv_content", type=str,
        help="Direct CSV content with YouTube channel URLs."
    )
    parser.add_argument(
        "--sleep", type=float,
        help="Delay (in seconds) between video metadata requests to avoid server overload."
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to configuration YAML file."
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    step1_config = config["step1"]
    
    # Configure logging.
    if args.log == "DEBUG":
        logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
    elif args.log == "INFO":
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    elif args.log == "NONE":
        logging.basicConfig(level=logging.CRITICAL, format='%(levelname)s: %(message)s')

    # Update sleep delay if provided in command line arguments
    if args.sleep:
        step1_config["sleep_delay"] = args.sleep
        logging.info(f"Updated sleep delay to {args.sleep} from command line argument")

    # Determine the channels to process
    channels = []
    
    # Option 1: Direct CSV content
    if args.channels_csv_content:
        channels = parse_channel_list(args.channels_csv_content, step1_config["channel_default_suffix"])
        logging.info(f"Parsed {len(channels)} channels from direct CSV content")
    
    # Option 2: CSV file path
    elif args.channels_csv:
        try:
            with open(args.channels_csv, 'r') as f:
                channels_csv = f.read()
            channels = parse_channel_list(channels_csv, step1_config["channel_default_suffix"])
            logging.info(f"Loaded {len(channels)} channels from CSV file")
        except Exception as e:
            logging.error(f"Error loading channels from CSV file: {e}")
            return
    
    # Option 3: Command line channel list
    elif args.channels:
        channels = args.channels
    
    # Option 4: Default channel list from config
    else:
        channels = step1_config["channels"]
    
    out_dir = get_output_directory()
    logging.info("Starting YouTube channel video metadata extraction and filtering.")
    get_yt_channel_videos(channels, out_dir, config)
    logging.info("Program finished.")

if __name__ == '__main__':
    main()