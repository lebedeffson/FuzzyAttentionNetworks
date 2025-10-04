#!/usr/bin/env python3
"""
Train Improved Fuzzy Attention Networks model with better parameters
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
import numpy as np

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

def create_improved_model(vocab_size=10000, answer_vocab_size=1000, device='cpu'):
    """Create improved model with better parameters"""
    model = VQAFuzzyModel(
        vocab_size=vocab_size,
        answer_vocab_size=answer_vocab_size,
        text_dim=512,  # Increased from 256
        image_dim=2048,
        hidden_dim=512,  # Increased from 256
        n_heads=8,  # Increased from 4
        n_layers=4,  # Increased from 2
        dropout=0.1  # Added dropout
    ).to(device)
    
    return model

def train_improved_model(model, dataloader, optimizer, criterion, device, num_epochs=20):
    """Train the improved model with better techniques"""
    model.train()
    best_accuracy = 0.0
    patience = 5
    patience_counter = 0
    
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
            binary_logits = logits.mean(dim=-1)
            binary_logits = binary_logits.unsqueeze(-1)
            binary_output = torch.cat([1 - binary_logits, binary_logits], dim=-1)
            
            loss = criterion(binary_output, labels)
            
            # Add regularization
            l2_reg = 0.001
            l2_loss = sum(p.pow(2.0).sum() for p in model.parameters())
            loss = loss + l2_reg * l2_loss
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
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
        
        # Early stopping
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break
    
    return model, best_accuracy

def evaluate_improved_model(model, dataloader, device):
    """Evaluate the improved model"""
    model.eval()
    correct_predictions = 0
    total_samples = 0
    predictions_list = []
    labels_list = []
    confidence_scores = []
    
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
            confidence = torch.softmax(binary_output, dim=-1).max(dim=-1)[0]
            
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)
            
            predictions_list.extend(predictions.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())
            confidence_scores.extend(confidence.cpu().numpy())
    
    accuracy = correct_predictions / total_samples
    avg_confidence = np.mean(confidence_scores)
    
    return accuracy, predictions_list, labels_list, confidence_scores

def save_improved_model(model, optimizer, epoch, loss, accuracy, save_path):
    """Save improved model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
        'timestamp': time.time(),
        'model_config': {
            'text_dim': 512,
            'image_dim': 2048,
            'hidden_dim': 512,
            'n_heads': 8,
            'n_layers': 4
        }
    }
    
    torch.save(checkpoint, save_path)
    print(f"Improved model saved to {save_path}")

def main():
    """Main training function for improved model"""
    print("🚀 IMPROVED FUZZY ATTENTION NETWORKS - MODEL TRAINING")
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
        max_samples=200,  # Use all 200 samples
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
    
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
    
    # Create improved model
    print("\n🤖 Creating improved model...")
    model = create_improved_model(device=device)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Setup training with better optimizer
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    criterion = nn.CrossEntropyLoss()
    
    # Train model
    print("\n🎯 Starting improved training...")
    trained_model, best_accuracy = train_improved_model(
        model, dataloader, optimizer, criterion, device, num_epochs=20
    )
    
    # Evaluate model
    print("\n📊 Evaluating improved model...")
    accuracy, predictions, labels, confidence_scores = evaluate_improved_model(trained_model, dataloader, device)
    
    print(f"\n📈 Final Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Best Accuracy: {best_accuracy:.4f}")
    print(f"  Average Confidence: {np.mean(confidence_scores):.4f}")
    print(f"  Correct: {sum(p == l for p, l in zip(predictions, labels))}/{len(predictions)}")
    
    # Save improved model
    model_path = models_dir / 'fuzzy_attention_improved.pth'
    save_improved_model(trained_model, optimizer, 20, 0.0, accuracy, model_path)
    
    # Save detailed results
    results = {
        'predictions': [int(p) for p in predictions],
        'labels': [int(l) for l in labels],
        'confidence_scores': [float(c) for c in confidence_scores],
        'accuracy': float(accuracy),
        'best_accuracy': float(best_accuracy),
        'total_samples': len(predictions),
        'model_config': {
            'text_dim': 512,
            'image_dim': 2048,
            'hidden_dim': 512,
            'n_heads': 8,
            'n_layers': 4
        }
    }
    
    results_path = models_dir / 'improved_training_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {results_path}")
    
    # Show some examples
    print(f"\n📝 Sample Predictions:")
    for i in range(min(10, len(predictions))):
        pred = "Hateful" if predictions[i] == 1 else "Non-hateful"
        actual = "Hateful" if labels[i] == 1 else "Non-hateful"
        confidence = confidence_scores[i]
        correct = "✅" if predictions[i] == labels[i] else "❌"
        print(f"  Sample {i+1}: Predicted={pred}, Actual={actual}, Confidence={confidence:.3f} {correct}")
    
    print(f"\n🎉 Improved training completed!")
    print(f"Model saved to: {model_path}")
    print(f"Results saved to: {results_path}")

if __name__ == "__main__":
    main()
