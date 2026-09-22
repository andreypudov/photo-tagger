#!/usr/bin/env bash
#
# Write generated metadata back into the photo files as IPTC/XMP tags.
#
# Usage: photo_embed_metadata.sh TAGS_JSON

set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: $(basename "$0") TAGS_JSON" >&2
    exit 2
fi

tags_json=$1

if [ ! -f "$tags_json" ]; then
    echo "Error: metadata file not found: $tags_json" >&2
    exit 2
fi

for tool in exiftool jq; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "Error: $tool is not installed (brew install $tool)" >&2
        exit 1
    fi
done

count=$(jq '.photos | length' "$tags_json")
for index in $(seq 0 $((count - 1))); do
    photo=$(jq -r ".photos[$index].file" "$tags_json")
    title=$(jq -r ".photos[$index].title" "$tags_json")
    description=$(jq -r ".photos[$index].description" "$tags_json")

    if [ ! -f "$photo" ]; then
        echo "Skipping missing file: $photo" >&2
        continue
    fi

    keyword_args=()
    while IFS= read -r keyword; do
        keyword_args+=("-IPTC:Keywords=$keyword" "-XMP-dc:Subject=$keyword")
    done < <(jq -r ".photos[$index].keywords[]" "$tags_json")

    # Stock sites and Lightroom read the IPTC title and caption, newer tools
    # the XMP ones, so both are written. IPTC is stored as UTF-8 so that
    # non-ASCII text survives.
    exiftool -overwrite_original \
        -charset iptc=UTF8 \
        -IPTC:CodedCharacterSet=UTF8 \
        -IPTC:ObjectName="$title" \
        -XMP-dc:Title="$title" \
        -IPTC:Caption-Abstract="$description" \
        -XMP-dc:Description="$description" \
        -EXIF:ImageDescription="$description" \
        -IPTC:Keywords= -XMP-dc:Subject= \
        "${keyword_args[@]}" \
        "$photo" >/dev/null

    echo "Tagged $photo"
done
