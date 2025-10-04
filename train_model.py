#!/usr/bin/env python3
"""
Train Fuzzy Attention Networks model on Hateful Memes dataset
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
import os
from pathlib import Path
import json
import time
from tqdm import tqdm

# Add src to path
src_path = os.path.join(os.path.dirname(__file__), 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Add experiments to path
experiments_path = os.path.join(os.path.dirname(__file__), 'experiments')
if experiments_path not in sys.path:
    sys.path.insert(0, experiments_path)

from multimodal_fuzzy_attention import VQAFuzzyModel
from evaluation_framework import HatefulMemesLocalDataset
from utils import setup_logging

def train_model(model, dataloader, optimizer, criterion, device, num_epochs=5):
    """Train the model"""
    model.train()
    total_loss = 0
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        correct_predictions = 0
        total_samples = 0
        
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 50)
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training")):
            question_tokens = batch['question_tokens'].to(device)
            image_features = batch['image_features'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            result = model(question_tokens, image_features, return_explanations=True)
            logits = result['answer_logits']
            
            # Calculate loss (using cross-entropy for binary classification)
            # Convert logits to binary classification
            binary_logits = logits.mean(dim=-1)  # Average over vocabulary
            binary_logits = binary_logits.unsqueeze(-1)  # [batch_size, 1]
            
            # Add a second output for non-hateful
            binary_output = torch.cat([1 - binary_logits, binary_logits], dim=-1)
            
            loss = criterion(binary_output, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Calculate accuracy
            predictions = torch.argmax(binary_output, dim=-1)
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)
            
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}: Loss = {loss.item():.4f}, Accuracy = {correct_predictions/total_samples:.4f}")
        
        avg_loss = epoch_loss / len(dataloader)
        accuracy = correct_predictions / total_samples
        
        print(f"Epoch {epoch + 1} Summary:")
        print(f"  Average Loss: {avg_loss:.4f}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Correct: {correct_predictions}/{total_samples}")
    
    return model

def evaluate_model(model, dataloader, device):
    """Evaluate the model"""
    model.eval()
    correct_predictions = 0
    total_samples = 0
    predictions_list = []
    labels_list = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            question_tokens = batch['question_tokens'].to(device)
            image_features = batch['image_features'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            result = model(question_tokens, image_features, return_explanations=True)
            logits = result['answer_logits']
            
            # Convert to binary classification
            binary_logits = logits.mean(dim=-1)
            binary_logits = binary_logits.unsqueeze(-1)
            binary_output = torch.cat([1 - binary_logits, binary_logits], dim=-1)
            
            predictions = torch.argmax(binary_output, dim=-1)
            
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)
            
            predictions_list.extend(predictions.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())
    
    accuracy = correct_predictions / total_samples
    return accuracy, predictions_list, labels_list

def save_model(model, optimizer, epoch, loss, accuracy, save_path):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
        'timestamp': time.time()
    }
    
    torch.save(checkpoint, save_path)
    print(f"Model saved to {save_path}")

def load_model(model, checkpoint_path, device):
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded from {checkpoint_path}")
    print(f"Epoch: {checkpoint['epoch']}, Accuracy: {checkpoint['accuracy']:.4f}")
    return model

def main():
    """Main training function"""
    print("🚀 FUZZY ATTENTION NETWORKS - MODEL TRAINING")
    print("=" * 60)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create models directory
    models_dir = Path('./models')
    models_dir.mkdir(exist_ok=True)
    
    # Load dataset
    print("\n📊 Loading Hateful Memes dataset...")
    dataset = HatefulMemesLocalDataset(
        root_dir='./data/hateful_memes',
        split='train',
        max_samples=50,  # Use all 50 samples
        device=device
    )
    
    print(f"Dataset loaded: {len(dataset)} samples")
    
    # Create data loader with custom collate function
    def collate_fn(batch):
        # Pad sequences to same length
        max_len = max(len(item['question_tokens']) for item in batch)
        
        padded_tokens = []
        padded_images = []
        labels = []
        
        for item in batch:
            # Pad question tokens
            tokens = item['question_tokens']
            if len(tokens) < max_len:
                padding = torch.zeros(max_len - len(tokens), dtype=tokens.dtype)
                tokens = torch.cat([tokens, padding])
            padded_tokens.append(tokens)
            
            # Image features are already the same size
            padded_images.append(item['image_features'])
            labels.append(item['label'])
        
        return {
            'question_tokens': torch.stack(padded_tokens),
            'image_features': torch.stack(padded_images),
            'label': torch.tensor(labels)
        }
    
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    
    # Create model
    print("\n🤖 Creating model...")
    model = VQAFuzzyModel(
        vocab_size=10000,
        answer_vocab_size=1000,
        text_dim=256,
        image_dim=2048,
        hidden_dim=256,
        n_heads=4,
        n_layers=2
    ).to(device)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Setup training
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Train model
    print("\n🎯 Starting training...")
    trained_model = train_model(model, dataloader, optimizer, criterion, device, num_epochs=10)
    
    # Evaluate model
    print("\n📊 Evaluating model...")
    accuracy, predictions, labels = evaluate_model(trained_model, dataloader, device)
    
    print(f"\n📈 Final Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Correct: {sum(p == l for p, l in zip(predictions, labels))}/{len(predictions)}")
    
    # Save model
    model_path = models_dir / 'fuzzy_attention_trained.pth'
    save_model(trained_model, optimizer, 10, 0.0, accuracy, model_path)
    
    # Save predictions for analysis
    results = {
        'predictions': [int(p) for p in predictions],
        'labels': [int(l) for l in labels],
        'accuracy': float(accuracy),
        'total_samples': len(predictions)
    }
    
    results_path = models_dir / 'training_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {results_path}")
    
    # Show some examples
    print(f"\n📝 Sample Predictions:")
    for i in range(min(10, len(predictions))):
        pred = "Hateful" if predictions[i] == 1 else "Non-hateful"
        actual = "Hateful" if labels[i] == 1 else "Non-hateful"
        correct = "✅" if predictions[i] == labels[i] else "❌"
        print(f"  Sample {i+1}: Predicted={pred}, Actual={actual} {correct}")
    
    print(f"\n🎉 Training completed!")
    print(f"Model saved to: {model_path}")
    print(f"Results saved to: {results_path}")

if __name__ == "__main__":
    main()
