import requests
import json
from PIL import Image, ImageDraw, ImageFont
import os

def main():
    username = os.getenv('GITHUB_USERNAME')
    token = os.getenv('GITHUB_TOKEN')

    if not username:
        print("Error: GITHUB_USERNAME environment variable not set.")
        return

    if not token:
        print("Error: GITHUB_TOKEN environment variable not set.")
        return

    # contrib data
    headers = {
        'Accept': 'application/vnd.github.v3+json',
        'Authorization': f'token {token}'
    }
    response = requests.get(f'https://api.github.com/users/{username}/events', headers=headers)

    if response.status_code != 200:
        print(f"Failed to fetch data: HTTP {response.status_code}")
        return

    data = response.json()
  
    total_contributions = sum(1 for event in data if event['type'] == 'PushEvent')
    img_size = (500, 500)
    bg_color = ((total_contributions * 5) % 256, 150, 150)  # Multiply to add variance

    # Create image
    img = Image.new('RGB', img_size, color=bg_color)
    draw = ImageDraw.Draw(img)

    # temp raccoon
    font = ImageFont.load_default()
    text = '🦝'
    text_size = draw.textsize(text, font=font)
    text_position = ((img_size[0] - text_size[0]) / 2, (img_size[1] - text_size[1]) / 2)
    draw.text(text_position, text, font=font, fill=(255, 255, 255))

    img.save('raccoon.png')

    print("Raccoon image generated successfully.")

if __name__ == '__main__':
    main()
