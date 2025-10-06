"""
Оптимизированное обучение финальной модели с улучшенными параметрами
Сохраняет сложную архитектуру согласно абстракту, но оптимизирует обучение
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# Add src and experiments to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'experiments'))

from ultimate_quality_model import UltimateQualityFuzzyModel
from evaluation_framework import HatefulMemesLocalDataset


def collate_fn(batch):
    """Custom collate function with proper padding"""
    max_len = max(len(item['question_tokens']) for item in batch)
    padded_tokens = []
    padded_images = []
    labels = []
    
    for item in batch:
        tokens = item['question_tokens']
        if len(tokens) < max_len:
            padding = torch.zeros(max_len - len(tokens), dtype=tokens.dtype)
            tokens = torch.cat([tokens, padding])
        padded_tokens.append(tokens)
        padded_images.append(item['image_features'])
        labels.append(item['label'])
    
    return {
        'question_tokens': torch.stack(padded_tokens),
        'image_features': torch.stack(padded_images),
        'label': torch.tensor(labels)
    }


def calculate_metrics(y_true, y_pred, y_proba):
    """Calculate comprehensive metrics"""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, average='binary', zero_division=0),
        'precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
        'auc': roc_auc_score(y_true, y_proba),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
    }


def get_class_weights(dataset):
    """Calculate class weights for balanced training"""
    labels = []
    for i in range(len(dataset)):
        labels.append(dataset[i]['label'])
    
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(labels),
        y=labels
    )
    
    return torch.FloatTensor(class_weights)


def train_epoch(model, dataloader, optimizer, criterion, device, l2_reg=0.0001):
    """Train for one epoch with advanced techniques"""
    model.train()
    total_loss = 0
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    for batch in tqdm(dataloader, desc="Training"):
        question_tokens = batch['question_tokens'].to(device)
        image_features = batch['image_features'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        result = model(question_tokens, image_features, return_explanations=False)
        logits = result['binary_logits']
        
        # Main loss
        loss = criterion(logits, labels)
        
        # L2 regularization
        l2_loss = 0
        for param in model.parameters():
            l2_loss += torch.norm(param, 2)
        loss += l2_reg * l2_loss
        
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        
        predictions = torch.argmax(logits, dim=-1)
        probabilities = torch.softmax(logits, dim=-1)[:, 1]
        
        all_predictions.extend(predictions.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())
        all_probabilities.extend(probabilities.detach().cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss, all_predictions, all_labels, all_probabilities


def evaluate_epoch(model, dataloader, criterion, device):
    """Evaluate for one epoch"""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            question_tokens = batch['question_tokens'].to(device)
            image_features = batch['image_features'].to(device)
            labels = batch['label'].to(device)
            
            result = model(question_tokens, image_features, return_explanations=False)
            logits = result['binary_logits']
            
            loss = criterion(logits, labels)
            total_loss += loss.item()
            
            predictions = torch.argmax(logits, dim=-1)
            probabilities = torch.softmax(logits, dim=-1)[:, 1]
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss, all_predictions, all_labels, all_probabilities


def plot_training_curves(train_losses, val_losses, train_f1s, val_f1s, save_path):
    """Plot training curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Loss curves
    ax1.plot(train_losses, label='Train Loss', color='blue', linewidth=2)
    ax1.plot(val_losses, label='Validation Loss', color='red', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # F1 curves
    ax2.plot(train_f1s, label='Train F1', color='blue', linewidth=2)
    ax2.plot(val_f1s, label='Validation F1', color='red', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('F1 Score')
    ax2.set_title('Training and Validation F1 Score')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(cm, save_path):
    """Plot confusion matrix"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-hateful', 'Hateful'],
                yticklabels=['Non-hateful', 'Hateful'],
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix - Optimized Final Model', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted', fontsize=14)
    plt.ylabel('Actual', fontsize=14)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """Main training function"""
    print("🚀 Оптимизированное обучение финальной модели")
    print("Сохраняем сложную архитектуру согласно абстракту")
    print("=" * 70)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create directories
    models_dir = Path('./models')
    results_dir = Path('./results')
    models_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)
    
    # Load dataset with more samples
    print("\n📊 Loading LARGE hateful memes dataset...")
    dataset = HatefulMemesLocalDataset(
        root_dir='./data/hateful_memes',
        split='train',
        max_samples=300  # Используем все 300 образцов
    )
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Validation samples: {len(val_dataset)}")
    
    # Calculate class weights for balanced training
    print("\n⚖️ Calculating class weights...")
    class_weights = get_class_weights(dataset)
    print(f"   Class weights: {class_weights}")
    
    # Create weighted sampler for balanced training
    train_labels = [dataset[i]['label'] for i in train_dataset.indices]
    sample_weights = [class_weights[label] for label in train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, sampler=sampler, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)
    
    # Create ultimate quality model (сохраняем сложную архитектуру)
    print("\n🤖 Creating ultimate quality fuzzy model...")
    model = UltimateQualityFuzzyModel(
        vocab_size=10000,
        text_dim=768,
        image_dim=2048,
        hidden_dim=512,  # Немного уменьшили для стабильности
        n_heads=8,       # Уменьшили количество голов
        n_layers=4,      # Уменьшили количество слоев
        dropout=0.2      # Увеличили dropout для регуляризации
    ).to(device)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Setup training with optimized parameters
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)  # Увеличили learning rate и weight decay
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))  # Используем weighted loss
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
    
    # Training loop
    print("\n🎯 Starting optimized training...")
    num_epochs = 100  # Уменьшили количество эпох
    best_val_f1 = 0.0
    patience = 20
    patience_counter = 0
    
    train_losses = []
    val_losses = []
    train_f1s = []
    val_f1s = []
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 60)
        
        # Train
        train_loss, train_preds, train_labels, train_probs = train_epoch(
            model, train_loader, optimizer, criterion, device, l2_reg=0.001
        )
        
        # Validate
        val_loss, val_preds, val_labels, val_probs = evaluate_epoch(
            model, val_loader, criterion, device
        )
        
        # Calculate metrics
        train_metrics = calculate_metrics(train_labels, train_preds, train_probs)
        val_metrics = calculate_metrics(val_labels, val_preds, val_probs)
        
        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_f1s.append(train_metrics['f1'])
        val_f1s.append(val_metrics['f1'])
        
        # Print results
        print(f"Train Loss: {train_loss:.4f}, Train F1: {train_metrics['f1']:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val F1: {val_metrics['f1']:.4f}")
        print(f"Val Accuracy: {val_metrics['accuracy']:.4f}, Val AUC: {val_metrics['auc']:.4f}")
        
        # Learning rate scheduling
        scheduler.step(val_metrics['f1'])
        
        # Early stopping and model saving
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            patience_counter = 0
            
            # Save best model
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_f1': val_metrics['f1'],
                'val_metrics': val_metrics,
                'class_weights': class_weights
            }, models_dir / 'best_optimized_final_model.pth')
            
            print(f"✅ New best model saved! (Val F1: {val_metrics['f1']:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
    
    # Final evaluation
    print("\n📊 Final Evaluation...")
    if (models_dir / 'best_optimized_final_model.pth').exists():
        model.load_state_dict(torch.load(models_dir / 'best_optimized_final_model.pth')['model_state_dict'])
    
    final_val_loss, final_val_preds, final_val_labels, final_val_probs = evaluate_epoch(
        model, val_loader, criterion, device
    )
    
    final_metrics = calculate_metrics(final_val_labels, final_val_preds, final_val_probs)
    
    print(f"\n📈 Final Results:")
    print(f"   Accuracy: {final_metrics['accuracy']:.4f}")
    print(f"   F1 Score: {final_metrics['f1']:.4f}")
    print(f"   Precision: {final_metrics['precision']:.4f}")
    print(f"   Recall: {final_metrics['recall']:.4f}")
    print(f"   AUC: {final_metrics['auc']:.4f}")
    
    # Save results
    results = {
        'final_metrics': final_metrics,
        'training_history': {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_f1s': train_f1s,
            'val_f1s': val_f1s
        },
        'model_config': {
            'text_dim': 768,
            'image_dim': 2048,
            'hidden_dim': 512,
            'n_heads': 8,
            'n_layers': 4,
            'dropout': 0.2
        },
        'class_weights': class_weights.tolist()
    }
    
    with open(results_dir / 'optimized_final_training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Plot training curves
    plot_training_curves(
        train_losses, val_losses, train_f1s, val_f1s,
        results_dir / 'optimized_final_training_curves.png'
    )
    
    # Plot confusion matrix
    plot_confusion_matrix(
        np.array(final_metrics['confusion_matrix']),
        results_dir / 'optimized_final_confusion_matrix.png'
    )
    
    print(f"\n🎉 Optimized training completed!")
    print(f"Best model saved to: {models_dir / 'best_optimized_final_model.pth'}")
    print(f"Results saved to: {results_dir / 'optimized_final_training_results.json'}")
    
    # Check quality
    if final_metrics['f1'] >= 0.90:
        print(f"\n🎯 EXCELLENT QUALITY! F1 = {final_metrics['f1']:.4f} >= 0.90")
    elif final_metrics['f1'] >= 0.85:
        print(f"\n🎯 HIGH QUALITY! F1 = {final_metrics['f1']:.4f} >= 0.85")
    elif final_metrics['f1'] >= 0.80:
        print(f"\n🎯 GOOD QUALITY! F1 = {final_metrics['f1']:.4f} >= 0.80")
    elif final_metrics['f1'] >= 0.70:
        print(f"\n✅ IMPROVED QUALITY! F1 = {final_metrics['f1']:.4f} >= 0.70")
    else:
        print(f"\n⚠️ Quality needs more improvement. F1 = {final_metrics['f1']:.4f} < 0.70")


if __name__ == "__main__":
    main()
