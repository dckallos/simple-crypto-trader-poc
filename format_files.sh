#!/bin/bash
# Usage: ./format_files.sh file1 file2 ...

# Check that at least one file is provided
if [ "$#" -lt 1 ]; then
  echo "Usage: $0 file1 [file2 ...]"
  exit 1
fi

output_file="output.md"
# Empty the output file if it already exists
> "$output_file"

# Process each file provided as an argument
for file in "$@"; do
  if [ ! -f "$file" ]; then
    echo "Warning: '$file' not found, skipping." >&2
    continue
  fi

  # Write the header for the file
  echo "## $(basename "$file")" >> "$output_file"
  echo "" >> "$output_file"

  # Write the opening code block
  echo '```' >> "$output_file"

  # Write the file's contents
  cat "$file" >> "$output_file"

  # Write the closing code block and a newline
  echo '```' >> "$output_file"
  echo "" >> "$output_file"
done
