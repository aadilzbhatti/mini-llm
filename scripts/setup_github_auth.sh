#!/bin/bash

# Exit on errors
set -e

# Ensure Git is installed
if ! command -v git &> /dev/null; then
    echo "Installing Git..."
    sudo apt update && sudo apt install -y git
else
    echo "Git is already installed."
fi

# Ensure GitHub CLI is installed
if ! command -v gh &> /dev/null; then
    echo "Installing GitHub CLI..."
    sudo apt update && sudo apt install -y gh
else
    echo "GitHub CLI is already installed."
fi

# Authenticate using GitHub CLI (fully automated)
echo "Starting GitHub authentication..."
gh auth login --with-token || gh auth login --web

# Configure Git to use GitHub CLI for authentication
echo "Configuring Git to use GitHub CLI credentials..."
git config --global credential.helper "gh auth login"

# Verify authentication
echo "Checking authentication status..."
gh auth status

echo "GitHub authentication setup complete! 🎉"
