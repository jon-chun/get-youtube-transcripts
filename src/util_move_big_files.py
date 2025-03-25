#!/usr/bin/env python
"""
Utility to move large files out of the git repository to prevent issues with GitHub push limits.
This script identifies large files in the data directory and moves them to a backup location.
"""

import os
import shutil
from pathlib import Path
import argparse
import logging
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def setup_parser():
    """Set up command line argument parser"""
    parser = argparse.ArgumentParser(
        description="Move large files out of the git repository"
    )
    parser.add_argument(
        "--size-limit", 
        type=int, 
        default=10, 
        help="Size limit in MB (default: 10MB)"
    )
    parser.add_argument(
        "--backup-dir", 
        type=str, 
        default=None, 
        help="Backup directory (default: ../backup_files)"
    )
    parser.add_argument(
        "--data-dir", 
        type=str, 
        default=None, 
        help="Data directory to scan (default: ./data)"
    )
    parser.add_argument(
        "--file-types", 
        type=str, 
        nargs="+", 
        default=[".mp3"], 
        help="File extensions to check (default: .mp3)"
    )
    parser.add_argument(
        "--dry-run", 
        action="store_true", 
        help="Only identify files, don't move them"
    )
    return parser

def find_large_files(data_dir, size_limit_mb, file_types):
    """Find files larger than size_limit_mb in the data directory"""
    size_limit_bytes = size_limit_mb * 1024 * 1024  # Convert MB to bytes
    large_files = []
    
    logging.info(f"Scanning {data_dir} for files larger than {size_limit_mb}MB...")
    
    for root, _, files in os.walk(data_dir):
        for file in files:
            if any(file.endswith(ext) for ext in file_types):
                file_path = os.path.join(root, file)
                file_size = os.path.getsize(file_path)
                
                if file_size > size_limit_bytes:
                    large_files.append({
                        'path': file_path,
                        'size_mb': file_size / (1024 * 1024),
                        'rel_path': os.path.relpath(file_path, data_dir)
                    })
    
    logging.info(f"Found {len(large_files)} files larger than {size_limit_mb}MB")
    return large_files

def move_files(large_files, backup_dir, dry_run=False):
    """Move large files to backup directory and create a manifest"""
    if not large_files:
        logging.info("No files to move")
        return
    
    # Create timestamp for this backup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_subdir = os.path.join(backup_dir, f"backup_{timestamp}")
    
    if not dry_run:
        os.makedirs(backup_subdir, exist_ok=True)
    
    # Track moved files in a manifest
    manifest = {
        'timestamp': timestamp,
        'files': []
    }
    
    for file_info in large_files:
        file_path = file_info['path']
        rel_path = file_info['rel_path']
        size_mb = file_info['size_mb']
        
        # Create subdirectories in backup to maintain structure
        backup_file_dir = os.path.join(backup_subdir, os.path.dirname(rel_path))
        backup_file_path = os.path.join(backup_subdir, rel_path)
        
        logging.info(f"{'Would move' if dry_run else 'Moving'} {file_path} ({size_mb:.2f}MB) to {backup_file_path}")
        
        if not dry_run:
            # Create directory structure
            os.makedirs(backup_file_dir, exist_ok=True)
            
            # Move the file
            try:
                shutil.move(file_path, backup_file_path)
                manifest['files'].append({
                    'original_path': file_path,
                    'backup_path': backup_file_path,
                    'size_mb': size_mb
                })
            except Exception as e:
                logging.error(f"Error moving {file_path}: {str(e)}")
    
    # Save the manifest
    if not dry_run and manifest['files']:
        manifest_path = os.path.join(backup_subdir, "manifest.json")
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        logging.info(f"Manifest saved to {manifest_path}")
    
    return manifest if not dry_run and manifest['files'] else None

def create_restore_script(manifest, backup_dir):
    """Create a restore script that can be used to move files back"""
    if not manifest or not manifest['files']:
        return
    
    timestamp = manifest['timestamp']
    restore_script_path = os.path.join(backup_dir, f"restore_{timestamp}.py")
    
    with open(restore_script_path, 'w') as f:
        f.write("""#!/usr/bin/env python
import os
import shutil
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load manifest
with open('{manifest_path}', 'r') as f:
    manifest = json.load(f)

# Restore files
for file_info in manifest['files']:
    original_path = file_info['original_path']
    backup_path = file_info['backup_path']
    
    # Create directory structure if needed
    os.makedirs(os.path.dirname(original_path), exist_ok=True)
    
    # Move the file back
    if os.path.exists(backup_path):
        logging.info(f"Restoring {{original_path}}")
        shutil.move(backup_path, original_path)
    else:
        logging.error(f"Backup file not found: {{backup_path}}")

logging.info("Restore complete")
""".format(manifest_path=os.path.join(backup_dir, f"backup_{timestamp}", "manifest.json")))
    
    # Make the script executable
    os.chmod(restore_script_path, 0o755)
    logging.info(f"Restore script created at {restore_script_path}")

def main():
    parser = setup_parser()
    args = parser.parse_args()
    
    # Set default directories if not provided
    script_dir = Path(__file__).parent
    
    if args.data_dir is None:
        data_dir = script_dir.parent / "data"
    else:
        data_dir = Path(args.data_dir)
    
    if args.backup_dir is None:
        backup_dir = script_dir.parent.parent / "backup_files"
    else:
        backup_dir = Path(args.backup_dir)
    
    # Ensure the backup directory exists
    if not args.dry_run:
        os.makedirs(backup_dir, exist_ok=True)
    
    # Find large files
    large_files = find_large_files(data_dir, args.size_limit, args.file_types)
    
    # Display found files
    if large_files:
        print("\nLarge files found:")
        for i, file_info in enumerate(large_files, 1):
            print(f"{i}. {file_info['path']} ({file_info['size_mb']:.2f} MB)")
    
    # Move files if not in dry run mode
    if not args.dry_run:
        confirm = input(f"\nMove {len(large_files)} files to {backup_dir}? (y/n): ")
        if confirm.lower() == 'y':
            manifest = move_files(large_files, backup_dir, args.dry_run)
            if manifest:
                create_restore_script(manifest, backup_dir)
        else:
            print("Operation cancelled")
    else:
        print("\nDry run completed. No files were moved.")

if __name__ == "__main__":
    main()