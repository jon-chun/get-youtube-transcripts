# YouTube Transcript Scraper

A comprehensive tool for downloading and processing YouTube video transcripts using multiple methods (YouTube API, youtube-transcript-api, yt-dlp, and audio download + Whisper).

## Setup Instructions

1. Clone this repository
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the root directory with your YouTube API key:
   ```
   YOUTUBE_API_KEY=your_api_key_here
   ```
4. Configure the tool by editing `src/config.yaml`
5. Prepare input files:
   - Add video IDs to `src/videos.txt` for individual videos
   - Add channel URLs to `src/channels.txt` for channel processing

## Configuration Guide

All configuration is done through the `src/config.yaml` file. Here's a sample configuration with explanations:

```yaml
# Logging configuration
LOGGING_LEVEL: INFO  # Options: DEBUG, INFO, WARNING, ERROR

# Input/Output files
YT_VIDEO_CSV: "videos.txt"  # File containing individual video IDs
YT_CHANNEL_CSV: "channels.txt"  # File containing channel URLs
OUTPUT_YT_VIDEO_CSV: "all_videos.csv"  # Output file with video information

# API and request configuration
MIN_WAIT_TIME: 1.0  # Minimum wait time between requests in seconds
MAX_PAGES_PER_CHANNEL: 5  # Maximum number of result pages to process per channel (50 videos per page)
MAX_VIDEOS_PER_CHANNEL: 10  # Maximum number of videos to process per channel

# Processing methods
METHOD_ORDER_LS: ["api", "yt-dlp", "audio"]  # Order of methods to try
PROCESS_SST_AUDIO: false  # Whether to process downloaded audio to create transcripts (requires Whisper)

# Language filters
LANG_INCL_LS: ["en"]  # Language codes to include for transcripts

# Content filters
MIN_LEN_MINS: 1.0  # Minimum video duration in minutes
MAX_LEN_MINS: 60.0  # Maximum video duration in minutes
DATE_START: "2023-01-01"  # Process videos published after this date (format: YYYY-MM-DD)
DATE_END: "2025-03-25"  # Process videos published before this date (format: YYYY-MM-DD)
FILTER_EXCLUDE_TITLE: ["trailer", "teaser", "preview"]  # Exclude videos with these terms in title
```

## Usage

Run the script from the command line:

```bash
python src/step1_get_yt_all.py
```

The script will:
1. Process individual videos from `videos.txt`
2. Process channels from `channels.txt`
3. Download transcripts using the methods specified in the configuration
4. Generate a summary report

## Output Files

The script produces several output files in the `data` directory:

- `all_videos.csv`: CSV file with information about all processed videos
- `yt_success.json`: JSON file with details of successfully processed videos
- `yt_fail.json`: JSON file with details of failed video processing attempts
- `transcript_download_report.txt`: Summary report of the download process
- `transcripts/`: Directory containing downloaded transcript JSON files
- `audio/`: Directory containing downloaded audio files (if audio method is used)

## Troubleshooting

### Silent MP3 Files

If downloaded MP3 files are silent:

1. Check if the YouTube video has audio content
2. Ensure FFmpeg is properly installed (check with `ffmpeg -version`)
3. Try increasing the audio quality in the code (change `preferredquality` in the `download_audio` function)
4. Try a different download method by changing the `METHOD_ORDER_LS` in config.yaml

### Script Hanging

If the script seems to hang:

1. Check if `MAX_PAGES_PER_CHANNEL` and `MAX_VIDEOS_PER_CHANNEL` are set to reasonable values
2. Make sure your internet connection is stable
3. The YouTube API might be rate-limiting your requests - adjust the wait times
4. For very large channels, consider processing them in smaller batches

### API Quota Limits

The YouTube Data API has daily quota limits:
- Search.list: 100 units
- Videos.list: 1 unit
- Daily quota: 10,000 units

If you hit quota limits, consider:
- Processing fewer channels per day
- Using other methods (youtube-transcript-api and yt-dlp don't count against quota)
- Applying for increased quota from Google

## Dependencies

- FFmpeg: Required for audio processing
- OpenAI Whisper (optional): For converting audio to transcripts
- Python packages: See requirements.txt

## License

[Your license information here]