#!/bin/bash

# ***************************************************************************************************
# Compatible IMAGE sample downloader (Fixed to detect existing files correctly)
# ***************************************************************************************************

# Base URL to a subset of ILSVRC (ImageNet) test images (1–1000) (unaffiliated with this project)
BASE_URL="https://www.isi.imi.i.u-tokyo.ac.jp/pattern/ilsvrc/test_images/all_mini/1-1000"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
DEST_DIR="${SCRIPT_DIR}/images"
NUM_IMAGES=10 # NOTE: CHANGE AS NEEDED

mkdir -p "$DEST_DIR"

echo "Saving images to: $DEST_DIR"

for i in $(seq -f "%08g" 1 "$NUM_IMAGES"); do
    FILENAME="ILSVRC2010_test_${i}.JPEG"
    DEST_PATH="${DEST_DIR}/${FILENAME}"

    if [[ -f "$DEST_PATH" ]]; then
        echo "Skipping existing file: $FILENAME"
    else
        echo "Downloading $FILENAME..."
        wget -q -O "$DEST_PATH" "${BASE_URL}/${FILENAME}"
    fi
done

echo "Done."