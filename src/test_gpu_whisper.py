# test_gpu.py
import torch
import whisper
import time
import subprocess
import sys

def test_ffmpeg_gpu():
    print("\n--- Testing FFmpeg GPU Acceleration ---")
    try:
        result = subprocess.run(
            ["ffmpeg", "-encoders"], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True,
            check=False
        )
        
        if "nvenc" in result.stdout.lower():
            print("✅ NVIDIA NVENC hardware acceleration available")
        elif "qsv" in result.stdout.lower():
            print("✅ Intel QSV hardware acceleration available")
        elif "amf" in result.stdout.lower():
            print("✅ AMD AMF hardware acceleration available")
        else:
            print("❌ No hardware acceleration detected in FFmpeg")
    except Exception as e:
        print(f"❌ Error testing FFmpeg: {str(e)}")

def test_whisper_gpu():
    print("\n--- Testing Whisper GPU Acceleration ---")
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"✅ CUDA is available: {torch.cuda.get_device_name()}")
        print(f"   CUDA version: {torch.version.cuda}")
    else:
        print("❌ CUDA is not available")
        return
    
    # Test whisper loading
    try:
        print("Loading Whisper model on GPU (this may take a moment)...")
        start_time = time.time()
        model = whisper.load_model("base", device="cuda")
        load_time = time.time() - start_time
        print(f"✅ Whisper model loaded successfully in {load_time:.2f} seconds")
        
        # Print memory usage
        memory_allocated = torch.cuda.memory_allocated() / 1024**2
        memory_reserved = torch.cuda.memory_reserved() / 1024**2
        print(f"   GPU memory allocated: {memory_allocated:.2f} MB")
        print(f"   GPU memory reserved: {memory_reserved:.2f} MB")
        
    except Exception as e:
        print(f"❌ Error loading Whisper model: {str(e)}")

if __name__ == "__main__":
    print("GPU Acceleration Test")
    print("====================")
    
    test_ffmpeg_gpu()
    test_whisper_gpu()