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

import googleapiclient.discovery
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
import yt_dlp
from fake_useragent import UserAgent  # For rotating browser agents


def load_config() -> Dict[str, Any]:
    """Load configuration from YAML file"""
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_logging(level: str) -> None:
    """Configure logging based on specified level"""
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(Path(__file__).parent / "youtube_transcript_scraper.log"),
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
            title = item['snippet']['title']
            published_at = item['snippet']['publishedAt']
            
            # Get full video details (including duration)
            video_details = get_video_details(youtube, config, video_id)
            if video_details and passes_filters(config, video_details, title, published_at):
                videos.append(video_details)
        
        next_page_token = response.get('nextPageToken')
        if not next_page_token:
            break
    
    logging.info(f"Found {len(videos)} matching videos for channel {channel_id}")
    return videos


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


def passes_filters(config: Dict[str, Any], video_details: Dict[str, Any], 
                  title: str, published_at: str) -> bool:
    """Check if video passes all filter criteria"""
    video_id = video_details['id']
    
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


def get_transcript(config: Dict[str, Any], video_id: str) -> Tuple[Optional[List[Dict]], str]:
    """Get transcript for a video if available"""
    logging.info(f"Attempting to get transcript for video {video_id}")
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        
        # Try to get manual transcript first
        try:
            transcript = transcript_list.find_manually_created_transcript(config["LANG_INCL_LS"])
            logging.debug(f"Found manually created transcript for {video_id}")
            return transcript.fetch(), "manual"
        except:
            # Fall back to generated transcript
            try:
                transcript = transcript_list.find_generated_transcript(config["LANG_INCL_LS"])
                logging.debug(f"Found auto-generated transcript for {video_id}")
                return transcript.fetch(), "generated"
            except:
                logging.debug(f"No transcript found in specified languages for {video_id}")
                return None, "not_found"
    
    except TranscriptsDisabled:
        logging.debug(f"Transcripts are disabled for video {video_id}")
        return None, "disabled"
    except Exception as e:
        logging.warning(f"Error getting transcript for {video_id}: {str(e)}")
        return None, str(e)


def download_audio(config: Dict[str, Any], video_id: str, output_path: Path) -> bool:
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
            return True
        except Exception as e:
            logging.error(f"Failed to download audio for {video_id}: {str(e)}")
            return False


def main():
    # Load configuration
    config = load_config()
    
    # Setup logging
    setup_logging(config["LOGGING_LEVEL"])
    
    logging.info("Starting YouTube transcript acquisition process")
    
    # Set up paths using pathlib for cross-platform compatibility
    base_dir = Path(__file__).parent
    input_file = base_dir / config["INPUT_YT_CHANNEL_CSV"]
    output_file = base_dir / config["OUTPUT_YT_VIDEO_CSV"]
    transcripts_dir = base_dir / config["OUTPUT_TRANSCRIPTS_DIR"]
    
    # Create output directory
    transcripts_dir.mkdir(exist_ok=True)
    logging.debug(f"Created output directory: {transcripts_dir}")
    
    # Setup YouTube API
    youtube = googleapiclient.discovery.build(
        "youtube", "v3", 
        developerKey=config["API_KEY"],
        cache_discovery=False
    )
    
    # Read input channels
    with open(input_file, 'r') as f:
        channel_urls = [line.strip() for line in f if line.strip()]
    
    logging.info(f"Found {len(channel_urls)} channels to process")
    
    # Setup output CSV
    all_videos = []
    
    # Process each channel
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
        
        # Add to main list
        all_videos.extend([(
            video['id'], 
            video['snippet']['title'],
            video['snippet']['channelTitle'],
            channel_url
        ) for video in videos])
        
        # Incrementally save to CSV
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['video_id', 'title', 'channel', 'channel_url'])
            writer.writerows(all_videos)
        
        # Prevent overwhelming the API
        power_law_pause(config["MIN_WAIT_TIME"] * 10)  # Longer pause between channels
    
    logging.info(f"Processing transcripts for {len(all_videos)} videos")
    
    # Process transcripts for all videos
    for video_id, title, channel, _ in all_videos:
        logging.info(f"Getting transcript for: {title}")
        
        # Try to get transcript
        transcript_data, source = get_transcript(config, video_id)
        
        if transcript_data:
            # Save transcript
            transcript_path = transcripts_dir / f"{video_id}.json"
            with open(transcript_path, 'w', encoding='utf-8') as f:
                json.dump(transcript_data, f, ensure_ascii=False, indent=2)
            logging.info(f"Saved {source} transcript for {video_id}")
        else:
            # Download audio as fallback
            logging.info(f"No transcript available ({source}), downloading audio")
            audio_path = transcripts_dir / f"{video_id}.mp3"
            if download_audio(config, video_id, audio_path):
                logging.info(f"Downloaded audio for {video_id}")
            else:
                logging.error(f"Failed to download audio for {video_id}")
        
        # Respect rate limits and add human-like behavior
        power_law_pause(config["MIN_WAIT_TIME"] * 2)


if __name__ == "__main__":
    main()