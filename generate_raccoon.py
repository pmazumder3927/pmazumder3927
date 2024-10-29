import requests
import json
from PIL import Image, ImageDraw, ImageFont
import os

def main():
    # Get the GitHub username and token from environment variables
    username = os.getenv('GITHUB_USERNAME')
    token = os.getenv('GITHUB_TOKEN')

    if not username:
        print("Error: GITHUB_USERNAME environment variable not set.")
        return

    if not token:
        print("Error: GITHUB_TOKEN environment variable not set.")
        return

    # Fetch contribution data from GitHub API with authentication
    headers = {
        'Accept': 'application/vnd.github.v3+json',
        'Authorization': f'token {token}'
    }
    response = requests.get(f'https://api.github.com/users/{username}/events', headers=headers)

    if response.status_code != 200:
        print(f"Failed to fetch data: HTTP {response.status_code}")
        return

    data = response.json()

    # Calculate total contributions (e.g., number of PushEvents)
    total_contributions = sum(1 for event in data if event['type'] == 'PushEvent')

    # Define image size and background color based on contributions
    img_size = (500, 500)
    bg_color = ((total_contributions * 5) % 256, 150, 150)  # Multiply to add variance

    # Create image
    img = Image.new('RGB', img_size, color=bg_color)
    draw = ImageDraw.Draw(img)

    # Add raccoon emoji or image
    font = ImageFont.load_default()
    text = '🦝'

    # Calculate text size using textbbox
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    text_position = ((img_size[0] - text_width) / 2, (img_size[1] - text_height) / 2)

    # Draw text
    draw.text(text_position, text, font=font, fill=(255, 255, 255))

    # Save image
    img.save('raccoon.png')

    print("Raccoon image generated successfully.")

if __name__ == '__main__':
    main()
