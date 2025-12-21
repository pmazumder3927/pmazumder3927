"""
Generate a unique raccoon image based on GitHub commit history.
Uses the trained diffusion model with a deterministic seed derived from commits.

Usage:
    python generate_raccoon.py

Environment variables:
    GITHUB_USERNAME: GitHub username to fetch commits from
    GITHUB_TOKEN: GitHub token for API access
    MODEL_PATH: Path to trained model (default: ./output_model/raccoon_model.pt)
"""

import hashlib
import os
import subprocess
from datetime import datetime
from pathlib import Path

import requests
import torch
from PIL import Image

from raccoon_diffusion.model import TinyUNet
from raccoon_diffusion.diffusion import GaussianDiffusion


def get_device():
    """Get the best available device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def get_commit_seed_from_git():
    """
    Get a deterministic seed from the local git history.
    Uses all commit hashes to create a unique seed.
    """
    try:
        # Get all commit hashes
        result = subprocess.run(
            ["git", "log", "--format=%H", "--all"],
            capture_output=True,
            text=True,
            check=True,
        )
        commits = result.stdout.strip()

        # Add today's date to make it change daily
        today = datetime.now().strftime("%Y-%m-%d")
        seed_string = commits + today

        # Hash to get a consistent integer seed
        hash_bytes = hashlib.sha256(seed_string.encode()).digest()
        seed = int.from_bytes(hash_bytes[:4], byteorder="big")

        return seed, len(commits.split("\n"))
    except Exception as e:
        print(f"Warning: Could not get git commits: {e}")
        return None, 0


def get_commit_seed_from_api(username: str, token: str):
    """
    Get a deterministic seed from GitHub API.
    Fetches recent events and commit data.
    """
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "Authorization": f"token {token}",
    }

    all_data = []

    # Fetch user events
    try:
        response = requests.get(
            f"https://api.github.com/users/{username}/events",
            headers=headers,
        )
        if response.status_code == 200:
            events = response.json()
            for event in events:
                if event["type"] == "PushEvent":
                    commits = event.get("payload", {}).get("commits", [])
                    for commit in commits:
                        all_data.append(commit.get("sha", ""))
    except Exception as e:
        print(f"Warning: Could not fetch events: {e}")

    # Fetch user repos and their commits
    try:
        response = requests.get(
            f"https://api.github.com/users/{username}/repos?per_page=10&sort=pushed",
            headers=headers,
        )
        if response.status_code == 200:
            repos = response.json()
            for repo in repos[:5]:  # Top 5 most recently pushed
                repo_name = repo["full_name"]
                commits_response = requests.get(
                    f"https://api.github.com/repos/{repo_name}/commits?per_page=20",
                    headers=headers,
                )
                if commits_response.status_code == 200:
                    commits = commits_response.json()
                    for commit in commits:
                        all_data.append(commit.get("sha", ""))
    except Exception as e:
        print(f"Warning: Could not fetch repos: {e}")

    # Add today's date to make it change daily
    today = datetime.now().strftime("%Y-%m-%d")
    seed_string = "".join(all_data) + today

    # Hash to get a consistent integer seed
    hash_bytes = hashlib.sha256(seed_string.encode()).digest()
    seed = int.from_bytes(hash_bytes[:4], byteorder="big")

    return seed, len(all_data)


def generate_raccoon(model_path: str, output_path: str, seed: int, device: torch.device):
    """Generate a raccoon image using the trained model."""
    # Load model
    model = TinyUNet(in_channels=3, out_channels=3, time_emb_dim=128).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Setup diffusion
    diffusion = GaussianDiffusion(timesteps=1000, device=device)

    # Generate with seed
    print(f"Generating raccoon with seed: {seed}")
    with torch.no_grad():
        # Use DDIM for faster sampling
        samples = diffusion.sample_ddim(
            model,
            shape=(1, 3, 64, 64),
            seed=seed,
            steps=50,
        )

        # Denormalize from [-1, 1] to [0, 255]
        samples = (samples + 1) / 2
        samples = samples.clamp(0, 1)
        samples = (samples * 255).byte()

        # Convert to PIL and save
        img_tensor = samples[0].cpu()
        img_array = img_tensor.permute(1, 2, 0).numpy()
        img = Image.fromarray(img_array)
        img.save(output_path)

    print(f"Saved raccoon to {output_path}")
    return img


def main():
    # Configuration
    model_path = os.getenv("MODEL_PATH", "./output_model/raccoon_model.pt")
    output_path = os.getenv("OUTPUT_PATH", "./raccoon.png")
    username = os.getenv("GITHUB_USERNAME")
    token = os.getenv("GITHUB_TOKEN")

    # Check if model exists
    if not Path(model_path).exists():
        print(f"Error: Model not found at {model_path}")
        print("Please train the model first with: python train.py")
        return

    # Get seed from commits
    seed = None
    num_commits = 0

    # Try local git first
    seed, num_commits = get_commit_seed_from_git()

    # Fall back to API if no local git or in CI
    if seed is None or os.getenv("CI"):
        if username and token:
            seed, num_commits = get_commit_seed_from_api(username, token)
        else:
            print("Warning: No git history and no API credentials")
            print("Using date-based seed only")
            today = datetime.now().strftime("%Y-%m-%d")
            seed = int(hashlib.sha256(today.encode()).hexdigest()[:8], 16)

    print(f"Commit data points: {num_commits}")
    print(f"Generated seed: {seed}")

    # Get device
    device = get_device()
    print(f"Using device: {device}")

    # Generate raccoon
    generate_raccoon(model_path, output_path, seed, device)
    print("Raccoon image generated successfully!")


if __name__ == "__main__":
    main()
