#!/bin/bash

# ***************************************************************************************************
# Compatible TEXT sample downloader (Fixed to detect existing files correctly)
# ***************************************************************************************************

# URL to a raw text file containing the "Harvard Sentences" dataset (unaffiliated with this project)
SENTENCE_URL="https://raw.githubusercontent.com/tidyverse/stringr/master/data-raw/harvard-sentences.txt"

# Customizable (Change as needed)
DEST_DIR="benchmarks/data/text"
NUM_FILES=10   # **NOTE**: Change this to set number of sentences/files to save
TMP_FILE="tmp_sentences.txt"

mkdir -p "$DEST_DIR"

echo "Downloading sentences file..."
wget -q -O "$TMP_FILE" "$SENTENCE_URL"

if [[ ! -f "$TMP_FILE" ]]; then
    echo "Error: Download failed."
    exit 1
fi

echo "Download complete. Splitting into individual files..."

count=0
while IFS= read -r line || [[ -n "$line" ]]; do
    if [[ -z "$line" ]]; then
        continue
    fi

    if [[ $count -ge $NUM_FILES ]]; then
        break
    fi

    count=$((count + 1))
    FILE_NAME=$(printf "sentence_%04d.txt" "$count")
    DEST_PATH="${DEST_DIR}/${FILE_NAME}"

    if [[ -f "$DEST_PATH" ]]; then
        echo "Skipping existing file: $DEST_PATH"
    else
        echo "$line" > "$DEST_PATH"
        echo "Saved: $DEST_PATH"
    fi

done < "$TMP_FILE"

# Clean up
rm -f "$TMP_FILE"
echo "Done! Created $count files in $DEST_DIR"