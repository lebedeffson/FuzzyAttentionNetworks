"""
Main entry point for Human-Centered Differentiable Neuro-Fuzzy Architectures
"""

import torch
import argparse
import logging
import sys
import os
from pathlib import Path

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Fuzzy Attention Networks')

    parser.add_argument('--mode', choices=['train', 'evaluate', 'demo', 'web', 'download_hateful_memes', 'train_improved'], 
                       default='demo', help='Run mode')
    parser.add_argument('--dataset', choices=['VQA-X', 'e-SNLI-VE', 'HatefulMemes', 'Beans'], default='Beans')
    parser.add_argument('--max_samples', type=int, default=200)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--model_path', type=str, help='Path to trained model')
    parser.add_argument('--data_path', type=str, default='./data', help='Data directory')
    parser.add_argument('--output_dir', type=str, default='./results', help='Output directory')
    parser.add_argument('--user_id', type=str, default='demo_user', help='User ID for demo')

    return parser.parse_args()

def main():
    """Main function"""
    args = parse_arguments()

    # Add src to Python path
    src_path = os.path.join(os.path.dirname(__file__), 'src')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    # Now we can import our modules
    try:
        from config import config
        from utils import set_seed, setup_logging
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Ensure you're running from the project root directory")
        return

    # Setup
    set_seed(config.seed)
    logger = setup_logging()
    logger.info("🚀 Starting Human-Centered Fuzzy Attention Networks")

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.mode == 'demo':
        run_demo(args, logger)
    elif args.mode == 'evaluate':
        run_evaluation(args, logger)
    elif args.mode == 'web':
        logger.info("Web interface mode (not implemented yet)")
    elif args.mode == 'train':
        run_training(args, logger)
    elif args.mode == 'download_hateful_memes':
        run_download_hateful_memes(args, logger)
    elif args.mode == 'train_improved':
        run_train_improved(args, logger)

def run_demo(args, logger):
    """Run interactive demo"""
    logger.info("🎯 Running interactive demo")

    try:
        from fuzzy_attention import MultiHeadFuzzyAttention
        from utils import FuzzyOperators

        # Initialize demo components
        d_model, n_heads, seq_len = 128, 4, 10

        logger.info(f"📊 Creating Fuzzy Attention layer: d_model={d_model}, n_heads={n_heads}")

        # Create fuzzy attention layer
        fuzzy_layer = MultiHeadFuzzyAttention(d_model, n_heads)

        # Create demo data
        demo_input = torch.randn(1, seq_len, d_model)

        logger.info(f"📊 Processing input shape: {demo_input.shape}")

        # Forward pass with attention
        with torch.no_grad():
            output, attention_info = fuzzy_layer(
                demo_input, demo_input, demo_input, 
                return_attention=True
            )

        logger.info(f"✅ Output shape: {output.shape}")

        if attention_info:
            avg_attention = attention_info['avg_attention']
            logger.info(f"📈 Average attention shape: {avg_attention.shape}")
            logger.info(f"🎯 Max attention weight: {avg_attention.max().item():.4f}")
            logger.info(f"📉 Min attention weight: {avg_attention.min().item():.4f}")

            # Analyze fuzzy attention patterns
            attention_entropy = -(avg_attention * torch.log(avg_attention + 1e-8)).sum(dim=-1).mean()
            logger.info(f"🔍 Attention entropy: {attention_entropy.item():.4f}")

        print("\n" + "="*50)
        print("🎉 FUZZY ATTENTION DEMO SUCCESSFUL!")
        print("="*50)
        print(f"✅ Processed sequence length: {seq_len}")
        print(f"✅ Model dimension: {d_model}")
        print(f"✅ Number of fuzzy heads: {n_heads}")
        print(f"✅ Output generated successfully")

        if attention_info:
            print(f"✅ Attention analysis completed")
            print(f"   📈 Max attention: {avg_attention.max().item():.4f}")
            print(f"   📉 Min attention: {avg_attention.min().item():.4f}")
            print(f"   🔍 Entropy: {attention_entropy.item():.4f}")

        print("="*50)

        # Test fuzzy operators
        print("\n🧠 Testing Fuzzy Logic Operators:")
        a, b = torch.tensor(0.7), torch.tensor(0.3)
        print(f"   Product t-norm T({a:.1f}, {b:.1f}) = {FuzzyOperators.product_tnorm(a, b):.3f}")
        print(f"   Minimum t-norm T({a:.1f}, {b:.1f}) = {FuzzyOperators.minimum_tnorm(a, b):.3f}")
        print(f"   Łukasiewicz t-norm T({a:.1f}, {b:.1f}) = {FuzzyOperators.lukasiewicz_tnorm(a, b):.3f}")

        logger.info("🎯 Demo completed successfully!")

    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        print(f"Error in demo: {e}")
        import traceback
        traceback.print_exc()

def run_evaluation(args, logger):
    """Run evaluation"""
    logger.info("📊 Evaluation mode")
    try:
        from experiments.evaluation_framework import (
            FuzzyAttentionEvaluator, 
            VQAXDataset, ESNLIVEDataset, BeansHFDataset
        )
        from src.multimodal_fuzzy_attention import VQAFuzzyModel
    except Exception as e:
        logger.error(f"❌ Import error in evaluation: {e}")
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = VQAFuzzyModel().to(device)
    evaluator = FuzzyAttentionEvaluator(device=device)

    if args.dataset == 'VQA-X':
        dataset = VQAXDataset(args.data_path, split='train', max_samples=args.max_samples)
    elif args.dataset == 'e-SNLI-VE':
        dataset = ESNLIVEDataset(args.data_path, split='train', max_samples=args.max_samples)
    elif args.dataset == 'Beans':
        dataset = BeansHFDataset(split='train', max_samples=args.max_samples, device=device)
    else:
        dataset = BeansHFDataset(split='train', max_samples=args.max_samples, device=device)

    evaluator.evaluate_model(model, dataset, args.dataset)

def run_training(args, logger):
    """Run training"""
    logger.info("🏋️ Training mode")
    try:
        from experiments.evaluation_framework import BeansHFDataset
        from src.multimodal_fuzzy_attention import VQAFuzzyModel
        from src.utils import save_model_checkpoint
    except Exception as e:
        logger.error(f"❌ Import error in training: {e}")
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = VQAFuzzyModel().to(device)
    dataset = BeansHFDataset(split='train', max_samples=args.max_samples, device=device)

    from torch.utils.data import DataLoader
    def collate_fn(batch_list):
        max_len = max(len(b['question_tokens']) for b in batch_list)
        padded = []
        for b in batch_list:
            q = b['question_tokens']
            if len(q) < max_len:
                q = torch.cat([q, torch.zeros(max_len - len(q), dtype=torch.long)])
            padded.append(q)
        question_tokens = torch.stack(padded)
        image_features = torch.stack([b['image_features'] for b in batch_list])
        labels = torch.tensor([b['label'] for b in batch_list], dtype=torch.long)
        return question_tokens, image_features, labels

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    model.train()
    global_step = 0
    for epoch in range(args.epochs):
        for question_tokens, image_features, labels in loader:
            question_tokens = question_tokens.to(device)
            image_features = image_features.to(device)
            labels = labels.to(device)

            optim.zero_grad()
            out = model(question_tokens, image_features, return_explanations=False)
            loss = loss_fn(out['answer_logits'], labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            if global_step % 10 == 0:
                logger.info(f"step {global_step} | loss {loss.item():.4f}")
            if global_step % 100 == 0 and global_step > 0:
                os.makedirs(args.output_dir, exist_ok=True)
                ckpt_path = os.path.join(args.output_dir, f"model_step_{global_step}.pt")
                save_model_checkpoint(model, optim, epoch, loss.item(), ckpt_path)
                logger.info(f"💾 Saved checkpoint to {ckpt_path}")
            global_step += 1

    logger.info("✅ Training finished")

def run_download_hateful_memes(args, logger):
    logger.info("⬇️ Download hateful_memes to local data folder")
    try:
        from utils.download_more_memes import main as dl_main
    except Exception as e:
        logger.error(f"❌ Failed to import download script: {e}")
        return
    # Forward CLI via env vars (token read from env or user passes)
    os.environ.setdefault('HF_DATASETS_OFFLINE', '0')
    dl_main()

def run_train_improved(args, logger):
    """Run improved model training"""
    logger.info("🏋️ Training improved model")
    try:
        import subprocess
        result = subprocess.run([
            'python', 'train_improved_model.py',
            '--epochs', str(args.epochs),
            '--batch_size', str(args.batch_size),
            '--lr', str(args.lr)
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("✅ Improved model training completed successfully")
            logger.info(f"Output: {result.stdout}")
        else:
            logger.error(f"❌ Training failed: {result.stderr}")
    except Exception as e:
        logger.error(f"❌ Error in improved training: {e}")

if __name__ == "__main__":
    main()
