#!/bin/bash
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# This script extracts multiple compressed files (zip, rar, tgz) in a directory.
# Each archive is extracted to a subdirectory named after the archive file.
# Usage: ./extract_files.sh /path/to/compressed/files

# Check if directory argument is provided
if [ -z "$1" ]; then
  echo "Usage: $0 path_to_compressed_files"
  exit 1
fi

# Change to the target directory
cd "$1" || exit 1

# Extract each compressed file based on its extension
for file in *.{zip,rar,tgz}; do
  if [ -e "$file" ]; then
    # Strip extension to create target directory name
    base_name=$(basename "$file" .zip)
    base_name=$(basename "$base_name" .rar)
    base_name=$(basename "$base_name" .tgz)

    mkdir -p "$base_name"

    # Use appropriate extraction tool based on file extension
    case "$file" in
      *.zip)
        echo "Extracting $file to $base_name..."
        unzip "$file" -d "$base_name"
        ;;
      *.rar)
        echo "Extracting $file to $base_name..."
        unrar x "$file" "$base_name/"
        ;;
      *.tgz)
        echo "Extracting $file to $base_name..."
        tar -xzf "$file" -C "$base_name"
        ;;
    esac
  fi
done

echo "Extraction complete."
