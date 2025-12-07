#!/bin/bash
# Git operations for latent-diffusion repository

echo "=== Git Repository Operations ==="

# Clone from original repository
clone_original() {
    echo "Cloning from CompVis repository..."
    git clone https://github.com/CompVis/latent-diffusion.git
    cd latent-diffusion
    echo "Repository cloned to: $(pwd)"
}

# Clone from fork
clone_fork() {
    echo "Cloning from fork..."
    read -p "Enter your GitHub username: " username
    git clone https://github.com/${username}/latent-diffusion.git
    cd latent-diffusion
}

# Update repository
update_repo() {
    echo "Updating repository..."
    cd latent-diffusion
    git pull origin main
    echo "Repository updated to latest version"
}

# Check commit information
check_commit_info() {
    cd latent-diffusion
    echo "Current commit: $(git rev-parse HEAD)"
    echo "Branch: $(git branch --show-current)"
    echo "Last 5 commits:"
    git log --oneline -5
}

# Main menu
echo "Select operation:"
echo "1) Clone original repository"
echo "2) Clone your fork"
echo "3) Update existing repository"
echo "4) Check commit information"
read -p "Choice [1-4]: " choice

case $choice in
    1) clone_original ;;
    2) clone_fork ;;
    3) update_repo ;;
    4) check_commit_info ;;
    *) echo "Invalid choice" ;;
esac
