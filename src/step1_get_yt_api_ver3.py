import os
import csv
import time
import json
import yaml
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any
from dotenv import load_dotenv

import googleapiclient.discovery
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
import yt_dlp
from fake_useragent import UserAgent

def load_environment():
    """Load sensitive information from .env file in root directory"""
    # Get the project root directory (one level up from src)
    root_dir = Path(__file__).parent.parent
    env_path = root_dir / ".env"
    
    # Load .env file
    load_dotenv(dotenv_path=env_path)
    
    # Return API key from env
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        logging.error("YouTube API key not found in .env file")
        raise ValueError("YouTube API key not found in .env file")
    
    return {"API_KEY": api_key}

def load_config() -> Dict[str, Any]:
    """Load configuration from YAML file"""
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_directories(base_dir: Path) -> Dict[str, Path]:
    """Setup required directories for the project"""
    # Define paths
    paths = {
        "transcripts_dir": base_dir.parent / "data" / "transcripts",
        "audio_dir": base_dir.parent / "data" / "audio",
        "output_dir": base_dir.parent / "data"
    }
    
    # Create directories
    for dir_path in paths.values():
        dir_path.mkdir(parents=True, exist_ok=True)
        logging.debug(f"Created directory: {dir_path}")
    
    return paths

def setup_logging(level: str) -> None:
    """Configure logging based on specified level"""
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")
    
    # Create logs directory
    logs_dir = Path(__file__).parent.parent / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(logs_dir / "youtube_transcript_scraper.log"),
            logging.StreamHandler()
        ]
    )

def power_law_pause(min_time: float = 0.5, alpha: float = 2.5) -> None:
    """
    Implement variable pauses following power law distribution with minimum time
    
    Args:
        min_time: Minimum pause time in seconds
        alpha: Power law exponent (higher = shorter pauses more common)
    """
    # Generate value from power law distribution (between 0 and 1)
    x = np.random.power(alpha)
    
    # Scale to create reasonable waiting time (0.5s to ~10s)
    wait_time = min_time + x * 9.5
    
    # Occasional longer pause (10% chance of 10-30s pause)
    if np.random.random() < 0.1:
        wait_time += np.random.uniform(10, 30)
    
    logging.debug(f"Waiting for {wait_time:.2f} seconds")
    time.sleep(wait_time)

def get_user_agent() -> str:
    """Get a random user agent using fake-useragent library"""
    ua = UserAgent()
    user_agent = ua.random
    logging.debug(f"Using user agent: {user_agent}")
    return user_agent

def parse_duration_to_minutes(duration: str) -> float:
    """Convert ISO 8601 duration to minutes"""
    # Basic implementation - would need enhancement for complex durations
    hours = 0
    minutes = 0
    seconds = 0
    
    # Extract hours
    if 'H' in duration:
        hours = int(duration.split('H')[0].split('T')[-1])
    
    # Extract minutes
    if 'M' in duration:
        if 'H' in duration:
            minutes = int(duration.split('H')[-1].split('M')[0])
        else:
            minutes = int(duration.split('T')[-1].split('M')[0])
    
    # Extract seconds
    if 'S' in duration:
        if 'M' in duration:
            seconds = int(duration.split('M')[-1].split('S')[0])
        else:
            seconds = int(duration.split('T')[-1].split('S')[0])
    
    total_minutes = hours * 60 + minutes + seconds / 60
    return total_minutes

def get_channel_id(youtube, channel_url: str) -> Optional[str]:
    """Extract channel ID from URL or handle custom URLs"""
    logging.info(f"Getting channel ID for: {channel_url}")
    
    if "@" in channel_url:
        channel_handle = channel_url.split('@')[-1].split('/')[0]
        request = youtube.search().list(
            part="snippet",
            q=channel_handle,
            type="channel",
            maxResults=1
        )
        response = request.execute()
        if response['items']:
            channel_id = response['items'][0]['id']['channelId']
            logging.debug(f"Found channel ID: {channel_id}")
            return channel_id
    
    logging.warning(f"Could not find channel ID for {channel_url}")
    return None

def get_video_details(youtube, config: Dict[str, Any], video_id: str) -> Optional[Dict[str, Any]]:
    """Get detailed video information including duration"""
    power_law_pause(config["MIN_WAIT_TIME"])
    
    # Update headers with new user agent
    youtube.videos().list.__func__.__globals__['service']._http.request_builder.headers["User-Agent"] = get_user_agent()
    
    request = youtube.videos().list(
        part="contentDetails,snippet",
        id=video_id
    )
    response = request.execute()
    
    if response['items']:
        logging.debug(f"Retrieved details for video {video_id}")
        return response['items'][0]
    
    logging.warning(f"Could not retrieve details for video {video_id}")
    return None

def passes_filters(config: Dict[str, Any], video_details: Dict[str, Any]) -> bool:
    """Check if video passes all filter criteria"""
    video_id = video_details['id']
    title = video_details['snippet']['title']
    published_at = video_details['snippet']['publishedAt']
    
    # 1. Duration check
    duration = video_details['contentDetails']['duration']  # ISO 8601 format
    duration_mins = parse_duration_to_minutes(duration)
    if not (config["MIN_LEN_MINS"] <= duration_mins <= config["MAX_LEN_MINS"]):
        logging.debug(f"Video {video_id} excluded: duration {duration_mins:.1f} mins outside range")
        return False
    
    # 2. Title keyword exclusion
    for exclude_term in config["FILTER_EXCLUDE_TITLE"]:
        if exclude_term.lower() in title.lower():
            logging.debug(f"Video {video_id} excluded: title contains '{exclude_term}'")
            return False
    
    # 3. Date range check
    if config["DATE_START"] or config["DATE_END"]:
        video_date = datetime.strptime(published_at, "%Y-%m-%dT%H:%M:%SZ")
        if config["DATE_START"]:
            start_date = datetime.strptime(config["DATE_START"], "%Y-%m-%d")
            if video_date < start_date:
                logging.debug(f"Video {video_id} excluded: published before {config['DATE_START']}")
                return False
        if config["DATE_END"]:
            end_date = datetime.strptime(config["DATE_END"], "%Y-%m-%d")
            if video_date > end_date:
                logging.debug(f"Video {video_id} excluded: published after {config['DATE_END']}")
                return False
    
    logging.debug(f"Video {video_id} passes all filters")
    return True

def get_all_videos(youtube, config: Dict[str, Any], channel_id: str) -> List[Dict[str, Any]]:
    """Get all videos for a channel respecting rate limits"""
    logging.info(f"Retrieving videos for channel {channel_id}")
    videos = []
    next_page_token = None
    
    while True:
        power_law_pause(config["MIN_WAIT_TIME"])
        
        # Update headers with new user agent for each request
        youtube.videos().list.__func__.__globals__['service']._http.request_builder.headers["User-Agent"] = get_user_agent()
        
        request = youtube.search().list(
            part="snippet",
            channelId=channel_id,
            maxResults=50,
            pageToken=next_page_token,
            type="video",
            order="date"
        )
        response = request.execute()
        
        logging.debug(f"Retrieved page with {len(response['items'])} videos")
        
        # Process video items
        for item in response['items']:
            video_id = item['id']['videoId']
            
            # Get full video details (including duration)
            video_details = get_video_details(youtube, config, video_id)
            if video_details and passes_filters(config, video_details):
                videos.append(video_details)
        
        next_page_token = response.get('nextPageToken')
        if not next_page_token:
            break
    
    logging.info(f"Found {len(videos)} matching videos for channel {channel_id}")
    return videos

def get_transcript_via_api(config: Dict[str, Any], video_id: str) -> Tuple[Optional[List[Dict]], str]:
    """Get transcript via YouTube Transcript API"""
    logging.info(f"Attempting to get transcript via API for video {video_id}")
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        
        # Try to get manual transcript first
        try:
            transcript = transcript_list.find_manually_created_transcript(config["LANG_INCL_LS"])
            logging.debug(f"Found manually created transcript for {video_id}")
            return transcript.fetch(), "manual_api"
        except:
            # Fall back to generated transcript
            try:
                transcript = transcript_list.find_generated_transcript(config["LANG_INCL_LS"])
                logging.debug(f"Found auto-generated transcript for {video_id}")
                return transcript.fetch(), "generated_api"
            except:
                logging.debug(f"No transcript found in specified languages for {video_id}")
                return None, "not_found_api"
    
    except TranscriptsDisabled:
        logging.debug(f"Transcripts are disabled for video {video_id}")
        return None, "disabled_api"
    except Exception as e:
        logging.warning(f"Error getting transcript via API for {video_id}: {str(e)}")
        return None, f"error_api: {str(e)}"

def get_transcript_via_ytdlp(config: Dict[str, Any], video_id: str) -> Tuple[Optional[List[Dict]], str]:
    """Get transcript via yt-dlp"""
    logging.info(f"Attempting to get transcript via yt-dlp for video {video_id}")
    
    # Set up temporary file for download
    temp_dir = Path(__file__).parent.parent / "temp"
    temp_dir.mkdir(exist_ok=True)
    temp_file = temp_dir / f"{video_id}.json"
    
    ydl_opts = {
        'skip_download': True,
        'writesubtitles': True,
        'writeautomaticsub': True,
        'subtitleslangs': config["LANG_INCL_LS"],
        'subtitlesformat': 'json',
        'quiet': True,
        'no_warnings': True,
        'outtmpl': str(temp_file),
        'user_agent': get_user_agent(),
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            ydl.download([f'https://www.youtube.com/watch?v={video_id}'])
            
            # Check if subtitle file was created
            subtitle_file = Path(f"{temp_file}.{config['LANG_INCL_LS'][0]}.json")
            if subtitle_file.exists():
                with open(subtitle_file, 'r', encoding='utf-8') as f:
                    transcript_data = json.load(f)
                
                # Delete temporary files
                subtitle_file.unlink()
                if temp_file.exists():
                    temp_file.unlink()
                
                logging.debug(f"Successfully extracted transcript via yt-dlp for {video_id}")
                return transcript_data, "ytdlp"
            else:
                logging.debug(f"No transcript found via yt-dlp for {video_id}")
                return None, "not_found_ytdlp"
                
        except Exception as e:
            logging.warning(f"Failed to extract transcript via yt-dlp for {video_id}: {str(e)}")
            return None, f"error_ytdlp: {str(e)}"

def download_audio(config: Dict[str, Any], video_id: str, output_path: Path) -> Tuple[bool, str]:
    """Download audio only at lower bitrate for whisper processing"""
    logging.info(f"Downloading audio for video {video_id}")
    
    ydl_opts = {
        'format': 'bestaudio[ext=m4a]/bestaudio',
        'outtmpl': str(output_path),
        'quiet': True,
        'no_warnings': True,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '64',
        }],
        'user_agent': get_user_agent(),
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            ydl.download([f'https://www.youtube.com/watch?v={video_id}'])
            logging.debug(f"Successfully downloaded audio for {video_id}")
            return True, "audio_downloaded"
        except Exception as e:
            logging.error(f"Failed to download audio for {video_id}: {str(e)}")
            return False, f"audio_error: {str(e)}"

def process_video_transcript(config: Dict[str, Any], video_id: str, title: str, channel: str,
                           paths: Dict[str, Path], success_data: Dict, fail_data: Dict) -> None:
    """Process video to get transcript or audio based on method order"""
    logging.info(f"Processing video: {title} (ID: {video_id})")
    
    # Skip if already processed successfully
    if video_id in success_data:
        logging.info(f"Skipping already processed video: {video_id}")
        return
    
    # Track which methods we've tried
    methods_tried = []
    
    transcript_data = None
    method_result = "all_methods_failed"
    
    # Try methods in the order specified
    for method in config["METHOD_ORDER_LS"]:
        methods_tried.append(method)
        
        if method == "api":
            transcript_data, method_result = get_transcript_via_api(config, video_id)
            if transcript_data:
                break
        
        elif method == "yt-dlp":
            transcript_data, method_result = get_transcript_via_ytdlp(config, video_id)
            if transcript_data:
                break
        
        elif method == "audio":
            audio_path = paths["audio_dir"] / f"{video_id}.mp3"
            success, method_result = download_audio(config, video_id, audio_path)
            if success:
                # Audio method doesn't return transcript data, but marks as success
                transcript_data = None
                break
    
    # Record success or failure
    if transcript_data or (method_result == "audio_downloaded"):
        # Save transcript if we have one
        if transcript_data:
            transcript_path = paths["transcripts_dir"] / f"{video_id}.json"
            with open(transcript_path, 'w', encoding='utf-8') as f:
                json.dump(transcript_data, f, ensure_ascii=False, indent=2)
            logging.info(f"Saved transcript for {video_id}")
        
        # Record success
        success_data[video_id] = {
            "title": title,
            "channel": channel,
            "method": method_result,
            "methods_tried": methods_tried,
            "timestamp": datetime.now().isoformat()
        }
    else:
        # Record failure
        fail_data[video_id] = {
            "title": title,
            "channel": channel,
            "method_result": method_result,
            "methods_tried": methods_tried,
            "timestamp": datetime.now().isoformat()
        }
    
    # Save success and failure data after each video
    with open(paths["output_dir"] / "yt_success.json", 'w', encoding='utf-8') as f:
        json.dump(success_data, f, ensure_ascii=False, indent=2)
    
    with open(paths["output_dir"] / "yt_fail.json", 'w', encoding='utf-8') as f:
        json.dump(fail_data, f, ensure_ascii=False, indent=2)
    
    # Respect rate limits and add human-like behavior
    power_law_pause(config["MIN_WAIT_TIME"] * 2)

def process_individual_videos(youtube, config: Dict[str, Any], paths: Dict[str, Path], 
                            success_data: Dict, fail_data: Dict) -> List[Tuple]:
    """Process individual videos from YT_VIDEO_CSV file"""
    logging.info("Starting to process individual videos")
    
    video_file_path = Path(__file__).parent / config["YT_VIDEO_CSV"]
    if not video_file_path.exists():
        logging.warning(f"Video list file {video_file_path} not found")
        return []
    
    # Read video IDs
    with open(video_file_path, 'r') as f:
        video_ids = [line.strip() for line in f if line.strip()]
    
    logging.info(f"Found {len(video_ids)} videos to process")
    
    # Process each video
    processed_videos = []
    for video_id in video_ids:
        # Skip videos we've already processed successfully
        if video_id in success_data:
            logging.info(f"Skipping already processed video: {video_id}")
            continue
        
        # Get video details
        video_details = get_video_details(youtube, config, video_id)
        if not video_details:
            logging.warning(f"Could not retrieve details for video {video_id}, skipping")
            fail_data[video_id] = {
                "title": "Unknown",
                "channel": "Unknown",
                "method_result": "video_details_not_found",
                "methods_tried": [],
                "timestamp": datetime.now().isoformat()
            }
            continue
        
        # Check if video passes filters
        if not passes_filters(config, video_details):
            logging.info(f"Video {video_id} did not pass filters, skipping")
            continue
        
        # Extract information
        title = video_details['snippet']['title']
        channel = video_details['snippet']['channelTitle']
        
        # Process video transcript
        process_video_transcript(config, video_id, title, channel, paths, success_data, fail_data)
        
        # Track processed video
        processed_videos.append((video_id, title, channel, "direct_input"))
    
    return processed_videos

def process_channel_videos(youtube, config: Dict[str, Any], paths: Dict[str, Path], 
                         success_data: Dict, fail_data: Dict) -> List[Tuple]:
    """Process videos from channels in YT_CHANNEL_CSV file"""
    logging.info("Starting to process channel videos")
    
    channel_file_path = Path(__file__).parent / config["YT_CHANNEL_CSV"]
    if not channel_file_path.exists():
        logging.warning(f"Channel list file {channel_file_path} not found")
        return []
    
    # Read channel URLs
    with open(channel_file_path, 'r') as f:
        channel_urls = [line.strip() for line in f if line.strip()]
    
    logging.info(f"Found {len(channel_urls)} channels to process")
    
    # Process each channel
    all_videos = []
    for channel_url in channel_urls:
        logging.info(f"Processing channel: {channel_url}")
        
        # Get channel ID
        channel_id = get_channel_id(youtube, channel_url)
        if not channel_id:
            logging.warning(f"Could not find channel ID for {channel_url}, skipping")
            continue
        
        # Get all videos passing filters
        videos = get_all_videos(youtube, config, channel_id)
        logging.info(f"Found {len(videos)} matching videos for {channel_url}")
        
        # Process each video from the channel
        for video in videos:
            video_id = video['id']
            title = video['snippet']['title']
            channel_title = video['snippet']['channelTitle']
            
            # Skip videos we've already processed successfully
            if video_id in success_data:
                logging.info(f"Skipping already processed video: {video_id}")
                continue
            
            # Process video transcript
            process_video_transcript(config, video_id, title, channel_title, paths, success_data, fail_data)
            
            # Add to list of processed videos
            all_videos.append((video_id, title, channel_title, channel_url))
        
        # Prevent overwhelming the API
        power_law_pause(config["MIN_WAIT_TIME"] * 10)  # Longer pause between channels
    
    return all_videos

def generate_summary_report(success_data: Dict, fail_data: Dict, paths: Dict[str, Path]) -> None:
    """Generate a human-readable summary report"""
    total_videos = len(success_data) + len(fail_data)
    success_rate = len(success_data) / total_videos * 100 if total_videos > 0 else 0
    
    # Count methods
    success_methods = {}
    for video_id, data in success_data.items():
        method = data["method"]
        success_methods[method] = success_methods.get(method, 0) + 1
    
    # Count failure reasons
    failure_reasons = {}
    for video_id, data in fail_data.items():
        reason = data["method_result"]
        failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
    
    # Create report
    report = [
        "YouTube Transcript Download Summary Report",
        "=" * 50,
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total videos processed: {total_videos}",
        f"Successful downloads: {len(success_data)} ({success_rate:.1f}%)",
        f"Failed downloads: {len(fail_data)} ({100-success_rate:.1f}%)",
        "",
        "Success Breakdown:",
        "-" * 20
    ]
    
    for method, count in success_methods.items():
        report.append(f"  {method}: {count} ({count/len(success_data)*100:.1f}% of successes)")
    
    report.extend([
        "",
        "Failure Breakdown:",
        "-" * 20
    ])
    
    for reason, count in failure_reasons.items():
        report.append(f"  {reason}: {count} ({count/len(fail_data)*100:.1f}% of failures)")
    
    report.extend([
        "",
        "YouTube API Quota Information:",
        "-" * 30,
        "The YouTube Data API v3 has a daily quota limit of 10,000 units.",
        "Each search.list request costs 100 units.",
        "Each videos.list request costs 1 unit.",
        "Using the youtube-transcript-api does not count against this quota.",
        "",
        "For more details, see: https://developers.google.com/youtube/v3/getting-started#quota"
    ])
    
    # Write report to file
    with open(paths["output_dir"] / "transcript_download_report.txt", 'w', encoding='utf-8') as f:
        f.write("\n".join(report))
    
    # Also log report
    for line in report:
        logging.info(line)

def main():
    # Get environment variables
    env_vars = load_environment()
    
    # Load configuration
    config = load_config()
    
    # Add API key from environment to config
    config["API_KEY"] = env_vars["API_KEY"]
    
    # Setup logging
    setup_logging(config["LOGGING_LEVEL"])
    
    logging.info("Starting YouTube transcript acquisition process")
    
    # Set up paths using pathlib for cross-platform compatibility
    base_dir = Path(__file__).parent
    paths = setup_directories(base_dir)
    
    output_file = paths["output_dir"] / config["OUTPUT_YT_VIDEO_CSV"]
    
    # Setup YouTube API
    youtube = googleapiclient.discovery.build(
        "youtube", "v3", 
        developerKey=config["API_KEY"],
        cache_discovery=False
    )
    
    # Load existing success and fail data if they exist
    success_data = {}
    fail_data = {}
    
    success_file = paths["output_dir"] / "yt_success.json"
    fail_file = paths["output_dir"] / "yt_fail.json"
    
    if success_file.exists():
        with open(success_file, 'r', encoding='utf-8') as f:
            success_data = json.load(f)
    
    if fail_file.exists():
        with open(fail_file, 'r', encoding='utf-8') as f:
            fail_data = json.load(f)
    
    # Process individual videos first (from YT_VIDEO_CSV)
    video_list = process_individual_videos(youtube, config, paths, success_data, fail_data)
    
    # Process channel videos (from YT_CHANNEL_CSV)
    channel_video_list = process_channel_videos(youtube, config, paths, success_data, fail_data)
    
    # Combine all videos
    all_videos = video_list + channel_video_list
    
    # Save complete list to output CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['video_id', 'title', 'channel', 'source'])
        writer.writerows(all_videos)
    
    # Generate summary report
    generate_summary_report(success_data, fail_data, paths)
    
    logging.info("YouTube transcript acquisition process completed")


if __name__ == "__main__":
    main()