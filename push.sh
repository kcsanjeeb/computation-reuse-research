#!/bin/bash

# Get current timestamp for commit message
TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")

# Git operations
git add .
git commit -m "Auto commit: $TIMESTAMP"
git push

clear