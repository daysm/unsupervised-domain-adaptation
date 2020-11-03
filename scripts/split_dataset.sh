#!/bin/bash

# Based on https://unix.stackexchange.com/questions/466593/moving-random-files-using-shuf-and-mv-argument-list-too-long
# Based on https://www.gnu.org/software/coreutils/manual/html_node/Random-sources.html#Random-sources

SOURCE_DIR=$1
TARGET_DIR=$2
TEST_SIZE=$3
RANDOM_SEED=$4

get_seeded_random()
{
  seed="$1"
  echo $seed
  openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt </dev/zero 2>/dev/null
}

mkdir -p $TARGET_DIR

NUM_SOURCE_FILES=$(find $SOURCE_DIR -mindepth 1 -maxdepth 1 ! -name '.*' | grep -c /)
NUM_SAMPLES=$(echo "x = $NUM_SOURCE_FILES*$TEST_SIZE; scale = 0; x / 1" | bc)
echo "Num source files: $NUM_SOURCE_FILES"
echo "Num sample files: $NUM_SAMPLES"

find $SOURCE_DIR -mindepth 1 -maxdepth 1 ! -name '.*' -print0 | shuf --random-source=<(get_seeded_random $RANDOM_SEED) -n $NUM_SAMPLES -z | xargs -0 -I{} mv {} $TARGET_DIR
