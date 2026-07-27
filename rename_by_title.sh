#!/bin/bash
# Rename arXiv-ID-named PDFs (e.g. 2210.09897v1.pdf) in a folder to their paper titles.
# Usage: bash rename_by_title.sh [folder]   (default: ./reference)
set -u
DIR="${1:-/Users/Ryan/directional_market_making/reference}"
cd "$DIR" || exit 1
for f in *.pdf; do
  id=$(echo "$f" | grep -oE '^[0-9]{4}\.[0-9]{4,5}' )
  [ -z "$id" ] && { echo "skip (no arXiv id in name): $f"; continue; }
  title=$(curl -sL "https://export.arxiv.org/api/query?id_list=${id}" \
    | python3 -c "
import sys, re
xml = sys.stdin.read()
m = re.findall(r'<title>(.*?)</title>', xml, re.S)
t = re.sub(r'\s+', ' ', m[1]).strip() if len(m) > 1 else ''
t = re.sub(r'[:/]', ' -', t); t = re.sub(r'[?\"*<>|]', '', t)
print(t)
")
  if [ -n "$title" ]; then
    mv -n "$f" "${title}.pdf" && echo "renamed: $f -> ${title}.pdf"
  else
    echo "FAIL (no title found): $f"
  fi
  sleep 1
done
