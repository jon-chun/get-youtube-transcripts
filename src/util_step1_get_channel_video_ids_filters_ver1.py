import yt_dlp
import json
import logging
import argparse

# Default channel list and output filename.
YT_CHANNEL_LS = ['@StPatrickChurchColumbusOhio/videos', '@CovenantPresbyterian/videos']
YT_VIDEOS_JSON = 'yt_videos_by_channel.json'

# Filtering criteria (constants)
VIDEO_LEN_MIN = 5 * 60         # Minimum video length in seconds (5 minutes)
VIDEO_LEN_MAX = 120 * 60       # Maximum video length in seconds (120 minutes)
VIDEO_DATE_AFTER = "20231231"  # Videos must be published after this date (YYYYMMDD)
VIDEO_DATE_BEFORE = "20250101" # Videos must be published before this date (YYYYMMDD)
VIDEO_LANG = ['en']            # Allowed video language(s)
VIDEO_VIEWS_MIN = 10           # Minimum view count
VIDEO_VIEWS_MAX = 1000000      # Maximum view count
VIDEO_UPVOTES_MIN = 0          # Minimum upvotes (like_count)
VIDEO_UPVOTES_MAX = 1000000    # Maximum upvotes (like_count)

def meets_criteria(video):
    """
    Check if the video's metadata meets all filtering criteria.
    
    Args:
        video (dict): Metadata for a single video.
        
    Returns:
        bool: True if video meets all criteria, False otherwise.
    """
    # Check duration: 'duration' is in seconds.
    duration = video.get('duration')
    if duration is None or not (VIDEO_LEN_MIN <= duration <= VIDEO_LEN_MAX):
        logging.debug(f"Video {video.get('id', 'unknown')} filtered out due to duration: {duration}")
        return False

    # Check upload date: expected as a string in YYYYMMDD format.
    upload_date = video.get('upload_date')
    if upload_date is None or not (VIDEO_DATE_AFTER < upload_date < VIDEO_DATE_BEFORE):
        logging.debug(f"Video {video.get('id', 'unknown')} filtered out due to upload_date: {upload_date}")
        return False

    # Check language: using the 'language' field if available.
    language = video.get('language')
    # If language info is missing, we assume it does not meet criteria.
    if language is None or language.lower() not in [lang.lower() for lang in VIDEO_LANG]:
        logging.debug(f"Video {video.get('id', 'unknown')} filtered out due to language: {language}")
        return False

    # Check view count.
    view_count = video.get('view_count')
    if view_count is None or not (VIDEO_VIEWS_MIN <= view_count <= VIDEO_VIEWS_MAX):
        logging.debug(f"Video {video.get('id', 'unknown')} filtered out due to view_count: {view_count}")
        return False

    # Check like count (upvotes).
    like_count = video.get('like_count')
    if like_count is None or not (VIDEO_UPVOTES_MIN <= like_count <= VIDEO_UPVOTES_MAX):
        logging.debug(f"Video {video.get('id', 'unknown')} filtered out due to like_count: {like_count}")
        return False

    return True

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

def get_yt_channel_videos(channels=YT_CHANNEL_LS, output_json=YT_VIDEOS_JSON):
    """
    Fetches video IDs from all provided channels, extracts full metadata for each video,
    applies filtering criteria, and saves the filtered video IDs to a JSON file.
    
    Args:
        channels (list): List of YouTube channel URLs or shorthand (with /videos appended).
        output_json (str): The filename for the output JSON file.
    """
    video_data = {}

    for original_channel in channels:
        logging.info(f"Processing channel: {original_channel}")
        try:
            channel_url = original_channel
            # Convert shorthand to full URL if needed.
            if not channel_url.startswith("http"):
                channel_url = "https://www.youtube.com/" + channel_url
                logging.debug(f"Converted shorthand to full URL: {channel_url}")

            # Expected URL format: https://www.youtube.com/@ChannelName/videos
            parts = channel_url.split('/')
            channel_id = parts[3] if len(parts) > 3 else channel_url
            logging.info(f"Extracted channel ID: {channel_id}")

            # Use flat extraction to quickly retrieve the list of video IDs.
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

            filtered_video_ids = []
            logging.info(f"Found {len(info['entries'])} total videos in channel {channel_id}")

            for entry in info['entries']:
                if entry is None or 'id' not in entry:
                    continue
                video_id = entry['id']
                logging.info(f"Extracting metadata for video: {video_id}")
                video_info = get_video_metadata(video_id)
                if video_info:
                    if meets_criteria(video_info):
                        filtered_video_ids.append(video_id)
                        logging.info(f"Video {video_id} meets criteria.")
                    else:
                        logging.debug(f"Video {video_id} did not meet filtering criteria.")
                else:
                    logging.warning(f"Skipping video {video_id} due to metadata extraction failure.")

            video_data[channel_id] = filtered_video_ids
            logging.info(f"Channel {channel_id}: {len(filtered_video_ids)} videos passed filtering.")

        except yt_dlp.utils.DownloadError as e:
            logging.error(f"Download error for channel {original_channel}: {e}")
        except Exception as e:
            logging.error(f"An unexpected error occurred for channel {original_channel}: {e}")

    # Save the filtered video IDs to a JSON file.
    try:
        with open(output_json, 'w') as f:
            json.dump(video_data, f, indent=4)
        logging.info(f"Filtered video IDs saved to {output_json}")
    except Exception as e:
        logging.error(f"Error saving to JSON file {output_json}: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Fetch YouTube channel video IDs, filter by criteria, and save to JSON."
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
        "--output", type=str, default=YT_VIDEOS_JSON,
        help="Output JSON file name."
    )
    args = parser.parse_args()

    # Configure logging based on the provided log level.
    if args.log == "DEBUG":
        logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
    elif args.log == "INFO":
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    elif args.log == "NONE":
        logging.basicConfig(level=logging.CRITICAL, format='%(levelname)s: %(message)s')

    channels = args.channels if args.channels else YT_CHANNEL_LS
    logging.info("Starting YouTube channel video metadata extraction and filtering.")
    get_yt_channel_videos(channels, args.output)
    logging.info("Completed extraction and filtering.")

if __name__ == '__main__':
    main()
