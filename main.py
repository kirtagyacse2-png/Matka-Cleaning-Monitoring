@@ -0,0 +1,48 @@
#!/usr/bin/env python3
"""
Matka Cleaning Monitoring System - Main Entry Point
"""

import argparse
import logging
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from scripts.train import main as train_main
from scripts.infer import main as infer_main
from utils.preprocessing import check_dependencies

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Matka Cleaning Monitoring System")
    parser.add_argument('--mode', choices=['train', 'infer'], required=True,
                       help='Mode: train or infer')
    parser.add_argument('--source', type=str, help='Path to image or webcam for inference')
    parser.add_argument('--model_path', type=str, default='models/matka_ir_cleaning.keras',
                        help='Path to trained model')

    args = parser.parse_args()

    # Check dependencies
    if not check_dependencies():
        logger.error("Dependencies not satisfied. Please install requirements.")
        return

    if args.mode == 'train':
        logger.info("Starting training...")
        train_main()
    elif args.mode == 'infer':
        if not args.source:
            logger.error("Source required for inference. Use --source path/to/image or 'webcam'")
            return
        logger.info(f"Starting inference with source: {args.source}")
        infer_main(args.source, args.model_path)

if __name__ == "__main__":
    main()
