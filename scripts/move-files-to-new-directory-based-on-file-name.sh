#!/bin/bash

START=$1
END=$2
for file in *; do dir=$(echo $file | cut -c $START-$END); mkdir -p $dir; mv "$file" "$dir"; done
