#!/usr/bin/env python
"""
Utility to convert audio files to text transcripts using OpenAI's Whisper model with GPU acceleration.
This script processes all MP3 files in the data/audio directory and creates corresponding transcripts
in the data/transcripts directory.


# Requirements:

OpenAI Whisper (pip install openai-whisper)
PyTorch with CUDA support (pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118)

# Output:
The script will:

Find all MP3 files in ../data/audio/
Process each file with Whisper
Save the transcripts as text files in ../data/transcripts/
Provide detailed logging of the process

For best performance, make sure your GPU drivers are up to date and you have enough VRAM for your chosen model size.


# Basic usage (uses "base" model and English language)
python util_audio2text.py

# Using a different model size
python util_audio2text.py --model small

# Force overwriting existing transcripts
python util_audio2text.py --force

# Specify a different language
python util_audio2text.py --language fr

# Force CPU usage even if GPU is available
python util_audio2text.py --cpu


Utility to convert audio files to text transcripts using OpenAI's Whisper model with GPU acceleration.
This script processes all MP3 files in the data/audio directory and creates corresponding transcripts
in the data/transcripts directory.
"""

import os
import sys
import time
import glob
import logging
import argparse
from pathlib import Path
import torch
from tqdm import tqdm
import math

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def setup_paths():
    """Setup paths relative to the script location to ensure it works from any directory"""
    # Get the directory of the script itself
    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    
    # Calculate paths relative to the parent directory
    parent_dir = script_dir.parent
    audio_dir = parent_dir / "data" / "audio"
    transcript_dir = parent_dir / "data" / "transcripts"
    
    # Ensure transcript directory exists
    os.makedirs(transcript_dir, exist_ok=True)
    
    return audio_dir, transcript_dir

def get_device_info():
    """Get information about the device and CUDA availability"""
    if not torch.cuda.is_available():
        logging.warning("CUDA is not available. Using CPU for transcription.")
        return "cpu", None
    
    device = "cuda"
    device_name = torch.cuda.get_device_name()
    device_count = torch.cuda.device_count()
    logging.info(f"Using GPU acceleration: {device_name}")
    logging.info(f"Number of available GPUs: {device_count}")
    
    # Print memory info
    memory_allocated = torch.cuda.memory_allocated() / (1024**2)
    memory_reserved = torch.cuda.memory_reserved() / (1024**2)
    logging.info(f"GPU memory allocated: {memory_allocated:.2f} MB")
    logging.info(f"GPU memory reserved: {memory_reserved:.2f} MB")
    
    return device, device_name

def load_whisper_model(model_size="base", device="cuda"):
    """Load the Whisper model with specified size and device"""
    try:
        import whisper
        logging.info(f"Loading Whisper {model_size} model on {device}...")
        start_time = time.time()
        model = whisper.load_model(model_size, device=device)
        elapsed_time = time.time() - start_time
        logging.info(f"Model loaded successfully in {elapsed_time:.2f} seconds")
        return model
    except ImportError:
        logging.error("OpenAI Whisper not installed. Install with: pip install openai-whisper")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error loading Whisper model: {str(e)}")
        sys.exit(1)

def get_audio_duration(file_path):
    """Get audio duration in seconds using Whisper's preprocessor if possible"""
    try:
        import whisper
        audio = whisper.load_audio(str(file_path))
        # Audio length in seconds
        return len(audio) / whisper.audio.SAMPLE_RATE
    except:
        # Fallback: estimate based on file size (very rough approximation)
        # ~10MB per minute at 128kbps
        size_mb = os.path.getsize(file_path) / (1024 * 1024)
        return size_mb * 6  # Approximate: 6 seconds per MB

def process_audio_file(file_path, transcript_path, model, device, args):
    """Process a single audio file with progress tracking"""
    file_name = file_path.stem
    
    try:
        # Estimate duration for progress bar
        duration = get_audio_duration(file_path)
        
        # Create a progress bar for this file
        with tqdm(
            total=100, 
            desc=f"Transcribing {file_name}",
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
            leave=False
        ) as pbar:
            
            start_time = time.time()
            
            # Function to update progress bar based on elapsed time vs duration
            def update_progress():
                elapsed = time.time() - start_time
                # Estimate progress percentage based on typical processing speed
                # (Whisper typically processes audio faster than real-time)
                if duration > 0:
                    progress = min(95, math.floor((elapsed / (duration * 0.5)) * 100))
                    pbar.n = progress
                    pbar.refresh()
            
            # Start a background thread to update progress
            import threading
            stop_flag = threading.Event()
            
            def progress_updater():
                while not stop_flag.is_set():
                    update_progress()
                    time.sleep(0.5)
            
            progress_thread = threading.Thread(target=progress_updater)
            progress_thread.daemon = True
            progress_thread.start()
            
            # Transcribe the audio
            result = model.transcribe(
                str(file_path),
                fp16=(device == "cuda"),  # Use half-precision for GPU
                language=args.language
            )
            
            # Stop the progress thread
            stop_flag.set()
            progress_thread.join(timeout=1.0)
            
            # Set progress to 100%
            pbar.n = 100
            pbar.refresh()
            
            # Get the full transcript
            transcript = result["text"].strip()
            
            # Save the transcript to a text file
            with open(transcript_path, 'w', encoding='utf-8') as f:
                f.write(transcript)
            
            elapsed_time = time.time() - start_time
            return True, elapsed_time
            
    except Exception as e:
        return False, str(e)

def process_audio_files(audio_dir, transcript_dir, model, device, args):
    """Process all MP3 files and create text transcripts with overall progress bar"""
    # Find all MP3 files
    audio_files = list(glob.glob(os.path.join(audio_dir, "*.mp3")))
    total_files = len(audio_files)
    
    if total_files == 0:
        logging.warning(f"No MP3 files found in {audio_dir}")
        return
    
    logging.info(f"Found {total_files} MP3 files to process")
    
    # Filter files that already have transcripts
    files_to_process = []
    for audio_file in audio_files:
        file_path = Path(audio_file)
        transcript_path = transcript_dir / f"{file_path.stem}.txt"
        if not transcript_path.exists() or args.force:
            files_to_process.append((file_path, transcript_path))
    
    skipped = total_files - len(files_to_process)
    if skipped > 0:
        logging.info(f"Skipping {skipped} files that already have transcripts")
    
    if not files_to_process:
        logging.info("All files already processed. Use --force to reprocess.")
        return
    
    # Process each file with an overall progress bar
    successful = 0
    failed = 0
    
    with tqdm(
        total=len(files_to_process),
        desc="Overall progress",
        unit="file",
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
    ) as pbar:
        for file_path, transcript_path in files_to_process:
            file_name = file_path.stem
            
            # Process the file
            success, result = process_audio_file(file_path, transcript_path, model, device, args)
            
            if success:
                logging.info(f"✓ Completed {file_name} in {result:.2f} seconds")
                successful += 1
                
                # Print memory usage if using GPU
                if device == "cuda":
                    memory_allocated = torch.cuda.memory_allocated() / (1024**2)
                    logging.debug(f"GPU memory allocated: {memory_allocated:.2f} MB")
                    
                    # Clear cache periodically
                    if successful % 5 == 0:
                        torch.cuda.empty_cache()
                        logging.debug("Cleared CUDA cache")
            else:
                logging.error(f"✗ Error processing {file_name}: {result}")
                failed += 1
            
            # Update overall progress bar
            pbar.update(1)
    
    # Print summary
    logging.info(f"Processing complete: {successful} successful, {failed} failed, {skipped} skipped")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Convert audio files to text transcripts using Whisper")
    parser.add_argument("--model", type=str, default="base", 
                        choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model size (default: base)")
    parser.add_argument("--force", action="store_true", 
                        help="Force overwrite of existing transcripts")
    parser.add_argument("--language", type=str, default="en",
                        help="Language code for transcription (default: en)")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU usage even if GPU is available")
    parser.add_argument("--no-progress", action="store_true",
                        help="Disable progress bars (useful for logging to file)")
    return parser.parse_args()

def main():
    """Main entry point"""
    args = parse_arguments()
    
    # Configure progress bar display
    if args.no_progress:
        # Disable tqdm progress bars
        tqdm.__init__ = lambda *_, **__: None
        tqdm.update = lambda *_, **__: None
    
    # Setup paths
    audio_dir, transcript_dir = setup_paths()
    logging.info(f"Audio directory: {audio_dir}")
    logging.info(f"Transcript directory: {transcript_dir}")
    
    # Get device info (CPU or GPU)
    device = "cpu" if args.cpu else get_device_info()[0]
    
    # Load Whisper model
    model = load_whisper_model(args.model, device)
    
    # Process audio files
    process_audio_files(audio_dir, transcript_dir, model, device, args)
    
    logging.info("Transcription completed")

if __name__ == "__main__":
    main()