#!/usr/bin/env python3
"""
FFmpeg-based script to resize AV1 videos with proper AV1 handling.
Forces software decoding to avoid hardware acceleration issues.
"""

import subprocess
import os
from pathlib import Path
import sys

def resize_videos_ffmpeg_av1():
    # Define paths
    input_dir = Path("combined_so101_follower_put_the_red_lego_block_in_the_black_cup_eps100_fps30/videos/chunk-000/observation.images.up_view")
    output_dir = Path("combined_so101_follower_put_the_red_lego_block_in_the_black_cup_eps100_fps30/videos/chunk-000/observation.images.up_view_resized")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all mp4 files
    video_files = sorted(list(input_dir.glob("*.mp4")))
    total_files = len(video_files)
    
    if total_files == 0:
        print(f"No MP4 files found in {input_dir}")
        return
    
    print(f"Found {total_files} video files to resize")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    
    successful = 0
    failed = 0
    
    for i, video_file in enumerate(video_files):
        try:
            print(f"Processing {i+1}/{total_files}: {video_file.name}")
            
            output_file = output_dir / video_file.name
            
            # FFmpeg command with AV1 specific options
            cmd = [
                'ffmpeg',
                '-y',  # Overwrite output files
                '-hwaccel', 'none',  # Force software decoding
                '-c:v', 'libaom-av1',  # Specify AV1 decoder
                '-i', str(video_file),
                '-vf', 'scale=640:480',  # Resize to 640x480
                '-c:v', 'libx264',  # Re-encode with H.264 for better compatibility
                '-preset', 'fast',  # Encoding speed
                '-crf', '23',  # Quality (lower = better quality)
                '-pix_fmt', 'yuv420p',  # Pixel format
                str(output_file)
            ]
            
            # Alternative command if the above fails
            cmd_fallback = [
                'ffmpeg',
                '-y',
                '-hwaccel', 'none',
                '-i', str(video_file),
                '-vf', 'scale=640:480',
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-crf', '23',
                '-pix_fmt', 'yuv420p',
                str(output_file)
            ]
            
            # Try main command first
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"  Main command failed, trying fallback...")
                print(f"  Error: {result.stderr}")
                
                # Try fallback command
                result = subprocess.run(cmd_fallback, capture_output=True, text=True)
                
                if result.returncode != 0:
                    print(f"  ✗ Failed to process {video_file.name}")
                    print(f"  Error: {result.stderr}")
                    failed += 1
                    continue
            
            print(f"  ✓ Successfully resized: {output_file.name}")
            successful += 1
            
        except Exception as e:
            print(f"  ✗ Error processing {video_file.name}: {e}")
            failed += 1
            continue
    
    print(f"\nCompleted! Successfully processed: {successful}, Failed: {failed}")

if __name__ == "__main__":
    resize_videos_ffmpeg_av1()
