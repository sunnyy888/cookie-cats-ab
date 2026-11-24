#!/usr/bin/env bash
# Download the public Cookie Cats A/B test dataset (originally on Kaggle) from a GitHub mirror and verify it.
# The same file is mirrored byte-for-byte in several repos; the SHA-256 below is the verified copy used here.
set -euo pipefail
cd "$(dirname "$0")"; mkdir -p data
curl -sSL -o data/cookie_cats.csv https://raw.githubusercontent.com/hbilu/CookieCats_AB_Testing/HEAD/datasets/cookie_cats.csv
echo "5ab54d761fbddcd50de7b88e4eaf7837cba4569474f50c043a4d17ee342c46bd  data/cookie_cats.csv" | shasum -a 256 -c -
