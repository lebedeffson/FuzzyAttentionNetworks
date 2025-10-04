#!/usr/bin/env python3
"""
Download more Hateful Memes images (50 samples) for comprehensive testing
"""

import requests
import json
from pathlib import Path
from datasets import load_dataset
import os
from PIL import Image
import io
import time
import random

def download_image(url, save_path):
    """Download image from URL and save locally"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # Open image and save
        img = Image.open(io.BytesIO(response.content))
        img.save(save_path)
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

def main():
    # Set token
    os.environ['HF_TOKEN'] = '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'
    
    print('Loading neuralcatcher/hateful_memes dataset...')
    dataset = load_dataset('neuralcatcher/hateful_memes', split='train')
    print(f'✅ Dataset loaded: {len(dataset)} samples')
    
    # Create data directory
    data_dir = Path('./data/hateful_memes')
    data_dir.mkdir(parents=True, exist_ok=True)
    img_dir = data_dir / 'img'
    img_dir.mkdir(exist_ok=True)
    
    # Base URL for images
    base_url = "https://huggingface.co/datasets/neuralcatcher/hateful_memes/resolve/main/"
    
    # Select 50 random samples
    num_samples = 50
    selected_indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    
    print(f'Downloading {num_samples} samples...')
    downloaded_count = 0
    hateful_count = 0
    non_hateful_count = 0
    
    with open(data_dir / 'train.jsonl', 'w') as f:
        for i, idx in enumerate(selected_indices):
            item = dataset[idx]
            
            # Create image filename
            img_filename = f'{item["id"]}.png'
            img_path = img_dir / img_filename
            
            # Try to download image
            img_url = base_url + item['img']
            print(f'Downloading {i+1}/{num_samples}: {img_url}')
            
            if download_image(img_url, img_path):
                downloaded_count += 1
                print(f'✅ Downloaded: {img_filename}')
            else:
                print(f'❌ Failed to download: {img_filename}')
                # Create a placeholder image
                placeholder = Image.new('RGB', (224, 224), color=(128, 128, 128))
                placeholder.save(img_path)
                print(f'📝 Created placeholder: {img_filename}')
            
            # Count labels
            if item['label'] == 1:
                hateful_count += 1
            else:
                non_hateful_count += 1
            
            # Create JSONL entry
            jsonl_entry = {
                'id': item['id'],
                'img': f'img/{img_filename}',
                'label': item['label'],
                'text': item['text']
            }
            f.write(json.dumps(jsonl_entry) + '\n')
            print(f'   Text: {item["text"][:60]}... (label: {item["label"]})')
            
            # Small delay to be nice to the server
            time.sleep(0.3)
    
    print(f'\n✅ Download completed!')
    print(f'   Downloaded: {downloaded_count}/{num_samples} images')
    print(f'   Hateful memes: {hateful_count}')
    print(f'   Non-hateful memes: {non_hateful_count}')
    print(f'   Hateful ratio: {hateful_count/num_samples:.1%}')
    print(f'   Data directory: {data_dir}')
    print(f'   Images directory: {img_dir}')
    print(f'   JSONL file: {data_dir / "train.jsonl"}')
    
    # Verify the files
    print(f'\n📁 Verifying files...')
    jsonl_file = data_dir / 'train.jsonl'
    if jsonl_file.exists():
        with open(jsonl_file, 'r') as f:
            lines = f.readlines()
        print(f'   JSONL entries: {len(lines)}')
        
        # Check image files
        img_files = list(img_dir.glob('*.png'))
        print(f'   Image files: {len(img_files)}')
        
        # Show sample statistics
        labels = []
        for line in lines[:10]:
            data = json.loads(line)
            labels.append(data['label'])
        
        print(f'   Sample labels: {labels}')
        print(f'   Sample hateful ratio: {sum(labels)/len(labels):.1%}')

if __name__ == "__main__":
    main()

