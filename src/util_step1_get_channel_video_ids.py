import yt_dlp
import json
import os

YT_CHANNEL_LS = ['@StPatrickChurchColumbusOhio/videos', '@CovenantPresbyterian/videos']
YT_VIDEOS_JSON = 'yt_videos_by_channel.json'

def get_yt_channel_videos(channels=YT_CHANNEL_LS, output_json=YT_VIDEOS_JSON):
    """
    Fetches video IDs for all channels in the provided list and saves them to a JSON file.

    Args:
        channels (list): A list of YouTube channel URLs or IDs (with /videos appended).
        output_json (str): The filename for the output JSON file.
    """

    video_data = {}

    for channel_url in channels:
        try:
            # Extract a more robust channel ID.
            channel_id = channel_url.split('/')[3]

            ydl_opts = {
                'quiet': True,  # Suppress console output (except errors)
                'extract_flat': True, # Get a list of videos without playlist info
                'skip_download': True, # Don't download any video content.
                'forcejson': False,   #  Don't force JSON, easier iteration.
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(channel_url, download=False)
                # Handle cases where extract_info might return None, or the wrong type.
                if not info or 'entries' not in info:
                    print(f"Warning: Could not retrieve video information for {channel_url}. Skipping.")
                    continue

                video_ids = [entry['id'] for entry in info['entries'] if entry and 'id' in entry]  # Filter out potential None entries
                video_data[channel_id] = video_ids

        except yt_dlp.utils.DownloadError as e:
            print(f"Error downloading info for {channel_url}: {e}")
        except Exception as e:
            print(f"An unexpected error occurred for {channel_url}: {e}")


    # Save to JSON file
    try:
        with open(output_json, 'w') as f:
            json.dump(video_data, f, indent=4)
        print(f"Video IDs saved to {output_json}")
    except (IOError, OSError) as e:
        print(f"Error saving to {output_json}: {e}")
    except Exception as e:
        print(f"Error saving to JSON File: {e}")



if __name__ == '__main__':
    get_yt_channel_videos()