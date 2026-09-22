#!/usr/bin/env bash
#
# Downscale photos into a working directory before tagging.
#
# Usage: photo_resize.sh SOURCE_DIR DEST_DIR [MAX_DIMENSION]

set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "Usage: $(basename "$0") SOURCE_DIR DEST_DIR [MAX_DIMENSION]" >&2
    exit 2
fi

source_dir=$1
dest_dir=$2
max_dimension=${3:-2048}

if [ ! -d "$source_dir" ]; then
    echo "Error: source directory not found: $source_dir" >&2
    exit 2
fi

if ! command -v magick >/dev/null 2>&1; then
    echo "Error: ImageMagick is not installed (brew install imagemagick)" >&2
    exit 1
fi

mkdir -p "$dest_dir"

shopt -s nullglob nocaseglob
for photo in "$source_dir"/*.{jpg,jpeg,png,tif,tiff,webp}; do
    name=$(basename "${photo%.*}")
    magick "$photo" -auto-orient -resize "${max_dimension}x${max_dimension}>" \
        -quality 90 "$dest_dir/$name.jpg"
    echo "Resized $photo -> $dest_dir/$name.jpg"
done
