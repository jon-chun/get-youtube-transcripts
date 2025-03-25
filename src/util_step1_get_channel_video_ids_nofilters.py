import yt_dlp
import json
import os
import logging
import argparse

# Default channel list and output filename.
YT_CHANNEL_LS = ['@StPatrickChurchColumbusOhio/videos', '@CovenantPresbyterian/videos']
YT_VIDEOS_JSON = 'yt_videos_by_channel.json'

def get_yt_channel_videos(channels=YT_CHANNEL_LS, output_json=YT_VIDEOS_JSON):
    """
    Fetches video IDs for all channels in the provided list and saves them to a JSON file.

    Args:
        channels (list): A list of YouTube channel URLs or shorthand (with /videos appended).
        output_json (str): The filename for the output JSON file.
    """
    video_data = {}

    for original_channel in channels:
        logging.info(f"Processing channel: {original_channel}")
        try:
            channel_url = original_channel
            # Convert shorthand to a full URL if needed.
            if not channel_url.startswith("http"):
                channel_url = "https://www.youtube.com/" + channel_url
                logging.debug(f"Converted shorthand to full URL: {channel_url}")

            # Extract the channel ID.
            # Expected URL format: https://www.youtube.com/@ChannelName/videos
            parts = channel_url.split('/')
            if len(parts) > 3:
                channel_id = parts[3]
            else:
                logging.warning(f"Unexpected URL format for {channel_url}. Using full URL as channel_id.")
                channel_id = channel_url

            logging.info(f"Extracted channel ID: {channel_id}")

            # Configure yt_dlp options.
            ydl_opts = {
                'quiet': True,           # Suppress non-error output.
                'extract_flat': True,    # Only extract a list of videos.
                'skip_download': True,   # Do not download video content.
                'forcejson': False,
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                logging.debug(f"Extracting info for channel: {channel_url}")
                info = ydl.extract_info(channel_url, download=False)

                if not info or 'entries' not in info:
                    logging.warning(f"Could not retrieve video information for {channel_url}. Skipping.")
                    continue

                video_ids = []
                for entry in info['entries']:
                    if entry and 'id' in entry:
                        video_id = entry['id']
                        video_ids.append(video_id)
                        logging.debug(f"Found video ID: {video_id}")

                video_data[channel_id] = video_ids
                logging.info(f"Processed {len(video_ids)} videos for channel {channel_id}")

        except yt_dlp.utils.DownloadError as e:
            logging.error(f"Download error for {original_channel}: {e}")
        except Exception as e:
            logging.error(f"An unexpected error occurred for {original_channel}: {e}")

    # Save the video IDs to a JSON file.
    try:
        with open(output_json, 'w') as f:
            json.dump(video_data, f, indent=4)
        logging.info(f"Video IDs saved to {output_json}")
    except (IOError, OSError) as e:
        logging.error(f"Error saving to {output_json}: {e}")
    except Exception as e:
        logging.error(f"Error saving to JSON file: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Fetch YouTube channel video IDs and save to JSON."
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
    logging.info("Starting YouTube channel video ID extraction.")
    get_yt_channel_videos(channels, args.output)
    logging.info("Completed extraction.")

if __name__ == '__main__':
    main()
