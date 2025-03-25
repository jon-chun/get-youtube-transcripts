# YouTube Transcript Acquisition Strategy

I'll outline a comprehensive approach to ethically acquire YouTube transcripts from multiple channels while minimizing the risk of IP blocking.

## Approach Overview

The most reliable method involves using YouTube's official API combined with third-party libraries for transcript retrieval. This approach respects YouTube's systems while accomplishing your goals.

### Core Components

1. **YouTube Data API v3**: For channel and video metadata retrieval
2. **youtube-transcript-api**: For accessing available transcripts
3. **yt-dlp**: As a fallback for downloading audio when transcripts aren't available

## Implementation Strategy

```python
import os
import csv
import time
import random
import googleapiclient.discovery
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
import yt_dlp
from datetime import datetime

# Configuration
INPUT_YT_CHANNEL_CSV = "input_yt_channels.csv"
OUTPUT_YT_VIDEO_CSV = "output_yt_videos.csv"
OUTPUT_TRANSCRIPTS_DIR = "transcripts"
MIN_LEN_MINS = 20
MAX_LEN_MINS = 240
LANG_INCL_LS = ['en']
FILTER_EXCLUDE_TITLE = ['going deeper', 'sunday school']
DATE_START = ""  # Format: YYYY-MM-DD
DATE_END = ""    # Format: YYYY-MM-DD

# Setup YouTube API
API_KEY = "YOUR_YOUTUBE_API_KEY"  # Get from Google Cloud Console
youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=API_KEY)

def human_like_pause():
    """Implement variable pauses to mimic human behavior"""
    base_pause = random.uniform(1, 3)
    occasional_longer_pause = random.choices([0, 1], weights=[0.8, 0.2])[0] * random.uniform(5, 10)
    time.sleep(base_pause + occasional_longer_pause)

def get_channel_id(channel_url):
    """Extract channel ID from URL or handle custom URLs"""
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
            return response['items'][0]['id']['channelId']
    return None

def get_all_videos(channel_id):
    """Get all videos for a channel respecting rate limits"""
    videos = []
    next_page_token = None
    
    while True:
        human_like_pause()  # Respect YouTube's servers
        
        request = youtube.search().list(
            part="snippet",
            channelId=channel_id,
            maxResults=50,
            pageToken=next_page_token,
            type="video",
            order="date"
        )
        response = request.execute()
        
        # Process video items
        for item in response['items']:
            video_id = item['id']['videoId']
            title = item['snippet']['title']
            published_at = item['snippet']['publishedAt']
            
            # Get full video details (including duration)
            video_details = get_video_details(video_id)
            if video_details and passes_filters(video_details, title, published_at):
                videos.append(video_details)
        
        next_page_token = response.get('nextPageToken')
        if not next_page_token:
            break
    
    return videos

def get_video_details(video_id):
    """Get detailed video information including duration"""
    human_like_pause()
    
    request = youtube.videos().list(
        part="contentDetails,snippet",
        id=video_id
    )
    response = request.execute()
    
    if response['items']:
        return response['items'][0]
    return None

def passes_filters(video_details, title, published_at):
    """Check if video passes all filter criteria"""
    # 1. Duration check
    duration = video_details['contentDetails']['duration']  # ISO 8601 format
    duration_mins = parse_duration_to_minutes(duration)
    if not (MIN_LEN_MINS <= duration_mins <= MAX_LEN_MINS):
        return False
    
    # 2. Title keyword exclusion
    for exclude_term in FILTER_EXCLUDE_TITLE:
        if exclude_term.lower() in title.lower():
            return False
    
    # 3. Date range check
    if DATE_START or DATE_END:
        video_date = datetime.strptime(published_at, "%Y-%m-%dT%H:%M:%SZ")
        if DATE_START:
            start_date = datetime.strptime(DATE_START, "%Y-%m-%d")
            if video_date < start_date:
                return False
        if DATE_END:
            end_date = datetime.strptime(DATE_END, "%Y-%m-%d")
            if video_date > end_date:
                return False
    
    # 4. Language check will be done when getting transcripts
    return True

def get_transcript(video_id):
    """Get transcript for a video if available"""
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        
        # Try to get manual transcript first
        try:
            transcript = transcript_list.find_manually_created_transcript(LANG_INCL_LS)
            return transcript.fetch(), "manual"
        except:
            # Fall back to generated transcript
            try:
                transcript = transcript_list.find_generated_transcript(LANG_INCL_LS)
                return transcript.fetch(), "generated"
            except:
                return None, "not_found"
    
    except TranscriptsDisabled:
        return None, "disabled"
    except Exception as e:
        return None, str(e)

def download_audio(video_id, output_path):
    """Download audio only at lower bitrate for whisper processing"""
    ydl_opts = {
        'format': 'bestaudio[ext=m4a]/bestaudio',
        'outtmpl': output_path,
        'quiet': True,
        'no_warnings': True,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '64',
        }],
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            ydl.download([f'https://www.youtube.com/watch?v={video_id}'])
            return True
        except:
            return False

def main():
    # Create output directory
    os.makedirs(OUTPUT_TRANSCRIPTS_DIR, exist_ok=True)
    
    # Read input channels
    with open(INPUT_YT_CHANNEL_CSV, 'r') as f:
        channel_urls = [line.strip() for line in f if line.strip()]
    
    # Setup output CSV
    all_videos = []
    
    # Process each channel
    for channel_url in channel_urls:
        print(f"Processing channel: {channel_url}")
        
        # Get channel ID
        channel_id = get_channel_id(channel_url)
        if not channel_id:
            print(f"Could not find channel ID for {channel_url}")
            continue
        
        # Get all videos passing filters
        videos = get_all_videos(channel_id)
        print(f"Found {len(videos)} matching videos")
        
        # Add to main list
        all_videos.extend([(
            video['id'], 
            video['snippet']['title'],
            video['snippet']['channelTitle'],
            channel_url
        ) for video in videos])
        
        # Incrementally save to CSV
        with open(OUTPUT_YT_VIDEO_CSV, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['video_id', 'title', 'channel', 'channel_url'])
            writer.writerows(all_videos)
        
        # Prevent overwhelming the API
        time.sleep(random.uniform(5, 10))
    
    # Process transcripts for all videos
    for video_id, title, channel, _ in all_videos:
        print(f"Getting transcript for: {title}")
        
        # Try to get transcript
        transcript_data, source = get_transcript(video_id)
        
        if transcript_data:
            # Save transcript
            transcript_path = os.path.join(OUTPUT_TRANSCRIPTS_DIR, f"{video_id}.json")
            with open(transcript_path, 'w', encoding='utf-8') as f:
                json.dump(transcript_data, f, ensure_ascii=False, indent=2)
            print(f"Saved {source} transcript")
        else:
            # Download audio as fallback
            print(f"No transcript available ({source}), downloading audio")
            audio_path = os.path.join(OUTPUT_TRANSCRIPTS_DIR, f"{video_id}.mp3")
            if download_audio(video_id, audio_path):
                print(f"Downloaded audio for later processing")
            else:
                print(f"Failed to download audio")
        
        # Respect rate limits and add human-like behavior
        human_like_pause()

if __name__ == "__main__":
    main()
```

## Transcript Acquisition Methods: Pros and Cons

### 1. YouTube's Official API + youtube-transcript-api

**Pros:**
- Most ethical and compliant approach
- Minimal risk of IP blocking
- Can access both user-created and auto-generated transcripts
- Reliable rate limiting built into API usage

**Cons:**
- Requires API key with quota limitations (free quota is 10,000 units/day)
- Some channels may disable transcript access

### 2. yt-dlp + Manual Transcript Extraction

**Pros:**
- Can work when API access is limited
- Can download videos with disabled transcripts
- More flexible for handling edge cases

**Cons:**
- Higher risk of IP blocking
- More resource-intensive
- May violate YouTube's terms of service if not careful

### 3. Audio Extraction + Local Speech-to-Text

**Pros:**
- Works even when transcripts are disabled
- Higher quality transcription for unclear audio
- Complete control over transcript format

**Cons:**
- Significantly higher resource usage
- Longer processing time
- Lower accuracy than official transcripts
- Still requires downloading content

## Anti-Blocking Best Practices

1. **Random Time Intervals**: Implement variable delays between requests (1-5 seconds for API, 30-120 seconds for downloads)

2. **Rotate User Agents**: Cycle through different user agent strings that mimic common browsers:
   ```python
   USER_AGENTS = [
       "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
       "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.5 Safari/605.1.15",
       # Add more user agents
   ]
   ```

3. **Exponential Backoff**: Implement increasing delays on failed requests:
   ```python
   def backoff_request(func, max_retries=5):
       retries = 0
       while retries < max_retries:
           try:
               return func()
           except Exception as e:
               wait_time = (2 ** retries) + random.uniform(0, 1)
               time.sleep(wait_time)
               retries += 1
       raise Exception("Max retries exceeded")
   ```

4. **Distributed Requests**: If possible, use multiple IP addresses through proxy rotation or VPN services:
   ```python
   PROXIES = [
       "http://proxy1.example.com:8080",
       "http://proxy2.example.com:8080",
       # Add more proxies
   ]
   ```

5. **Respect Robots.txt**: Always check and honor robots.txt restrictions

## Recommended Implementation

The most balanced approach combines all three methods in a priority sequence:

1. First attempt: Official API + youtube-transcript-api
2. If that fails: Try yt-dlp for transcript extraction
3. Last resort: Download audio at minimal quality and process with Whisper locally

This tiered approach ensures you get transcripts while minimizing chances of IP blocking or other issues.

Would you like me to explore any specific aspect of this strategy in more detail?


project_root/
├── .env                             # Contains API keys
├── src/
│   ├── config.yaml                  # Configuration file
│   └── youtube_transcript_scraper.py # Main script
├── data/
│   ├── transcripts/                 # Saved transcripts
│   ├── audio/                       # Downloaded audio files
│   ├── yt_success.json              # Successful downloads
│   ├── yt_fail.json                 # Failed downloads
│   └── transcript_download_report.txt # Summary report
└── logs/
    └── youtube_transcript_scraper.log # Log file

    