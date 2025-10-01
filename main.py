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

    parser.add_argument('--mode', choices=['train', 'evaluate', 'demo', 'web'], 
                       default='demo', help='Run mode')
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
    logger.info("📊 Evaluation mode (basic implementation)")
    print("Evaluation functionality will be implemented in experiments/")

def run_training(args, logger):
    """Run training"""
    logger.info("🏋️ Training mode (not implemented yet)")
    print("Training functionality will be implemented in experiments/")

if __name__ == "__main__":
    main()
