#!/usr/bin/env python3
"""
Video Embedding Extraction Script

This script extracts video embeddings from all videos in a directory using the 
TempMe model for efficient video representation learning.

Usage:
    python extract_video_embeddings.py --video_dir /path/to/videos --output_dir /path/to/embeddings
    
Features:
- Batch processing of all videos in a directory
- Efficient video embedding extraction using TempMe
- Configurable output formats (individual files or consolidated)
- GPU acceleration support
"""

import os
import sys
import argparse
import logging
import torch
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from unified_model import create_unified_model
from config import UnifiedModelConfig

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Supported video extensions
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.m4v', '.webm'}


def get_video_files(video_dir: str) -> List[str]:
    """
    Get all video files from the specified directory.
    
    Args:
        video_dir: Directory containing video files
        
    Returns:
        List of video file paths
    """
    video_files = []
    video_dir = Path(video_dir)
    
    if not video_dir.exists():
        raise FileNotFoundError(f"Video directory not found: {video_dir}")
    
    for file_path in video_dir.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in VIDEO_EXTENSIONS:
            video_files.append(str(file_path))
    
    logger.info(f"Found {len(video_files)} video files in {video_dir}")
    return sorted(video_files)


def extract_single_video_embedding(model, video_path: str, device: str) -> np.ndarray:
    """
    Extract embedding for a single video.
    
    Args:
        model: Loaded unified model
        video_path: Path to video file
        device: Device to run inference on
        
    Returns:
        Video embedding as numpy array
    """
    try:
        # Extract video embedding using the model
        with torch.no_grad():
            embedding = model.extract_video_embedding(video_path)
            
        # Convert to numpy if it's a tensor
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.cpu().numpy()
            
        return embedding
        
    except Exception as e:
        logger.error(f"Failed to extract embedding for {video_path}: {e}")
        raise


def save_embedding(embedding: np.ndarray, video_path: str, output_dir: str, 
                  save_format: str = 'npy') -> str:
    """
    Save video embedding to file.
    
    Args:
        embedding: Video embedding array
        video_path: Original video path
        output_dir: Output directory
        save_format: Save format ('npy', 'json', or 'both')
        
    Returns:
        Saved file path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create output filename based on video filename
    video_name = Path(video_path).stem
    
    saved_files = []
    
    if save_format in ['npy', 'both']:
        npy_path = output_dir / f"{video_name}_embedding.npy"
        np.save(npy_path, embedding)
        saved_files.append(str(npy_path))
        
    if save_format in ['json', 'both']:
        json_path = output_dir / f"{video_name}_embedding.json"
        embedding_data = {
            'video_path': video_path,
            'embedding': embedding.tolist(),
            'shape': embedding.shape,
            'dtype': str(embedding.dtype)
        }
        with open(json_path, 'w') as f:
            json.dump(embedding_data, f, indent=2)
        saved_files.append(str(json_path))
    
    return saved_files[0] if len(saved_files) == 1 else saved_files


def extract_directory_embeddings(video_dir: str, output_dir: str, config_path: str = None,
                                device: str = 'cuda', save_format: str = 'npy',
                                consolidated: bool = False) -> Dict:
    """
    Extract embeddings for all videos in a directory.
    
    Args:
        video_dir: Directory containing videos
        output_dir: Directory to save embeddings
        config_path: Path to model configuration
        device: Device to run inference on
        save_format: Format to save embeddings ('npy', 'json', 'both')
        consolidated: Whether to save all embeddings in one file
        
    Returns:
        Dictionary with processing results
    """
    # Get all video files
    video_files = get_video_files(video_dir)
    
    if not video_files:
        logger.warning(f"No video files found in {video_dir}")
        return {'status': 'no_videos', 'processed': 0, 'failed': 0}
    
    # Load model
    logger.info("Loading unified model for embedding extraction...")
    model = create_unified_model(config_path=config_path, device=device)
    model.to(device)
    model.eval()
    
    # Process videos
    results = []
    failed_videos = []
    
    logger.info(f"Processing {len(video_files)} videos...")
    
    for i, video_path in enumerate(video_files):
        logger.info(f"Processing {i+1}/{len(video_files)}: {Path(video_path).name}")
        
        try:
            start_time = time.time()
            
            # Extract embedding
            embedding = extract_single_video_embedding(model, video_path, device)
            
            # Save embedding
            if not consolidated:
                saved_path = save_embedding(embedding, video_path, output_dir, save_format)
                logger.info(f"Saved embedding to: {saved_path}")
            
            processing_time = time.time() - start_time
            
            results.append({
                'video_path': video_path,
                'embedding_shape': embedding.shape,
                'processing_time': processing_time,
                'status': 'success'
            })
            
        except Exception as e:
            logger.error(f"Failed to process {video_path}: {e}")
            failed_videos.append(video_path)
            results.append({
                'video_path': video_path,
                'status': 'failed',
                'error': str(e)
            })
    
    # Save consolidated embeddings if requested
    if consolidated:
        logger.info("Saving consolidated embeddings...")
        consolidated_data = {
            'embeddings': {},
            'metadata': {
                'total_videos': len(video_files),
                'successful': len([r for r in results if r['status'] == 'success']),
                'failed': len(failed_videos),
                'processing_date': time.strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        
        for result in results:
            if result['status'] == 'success':
                video_path = result['video_path']
                embedding = extract_single_video_embedding(model, video_path, device)
                video_name = Path(video_path).stem
                consolidated_data['embeddings'][video_name] = {
                    'embedding': embedding.tolist(),
                    'shape': embedding.shape,
                    'video_path': video_path
                }
        
        consolidated_path = Path(output_dir) / 'consolidated_embeddings.json'
        with open(consolidated_path, 'w') as f:
            json.dump(consolidated_data, f, indent=2)
        logger.info(f"Consolidated embeddings saved to: {consolidated_path}")
    
    # Summary
    successful = len([r for r in results if r['status'] == 'success'])
    logger.info(f"Processing complete: {successful}/{len(video_files)} videos processed successfully")
    
    if failed_videos:
        logger.warning(f"Failed videos: {failed_videos}")
    
    return {
        'status': 'complete',
        'total_videos': len(video_files),
        'processed': successful,
        'failed': len(failed_videos),
        'results': results,
        'failed_videos': failed_videos
    }


def main():
    """Main entry point for the embedding extraction script."""
    parser = argparse.ArgumentParser(
        description="Extract video embeddings from all videos in a directory"
    )
    parser.add_argument(
        '--video_dir',
        type=str,
        required=True,
        help='Directory containing video files to process'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Directory to save extracted embeddings'
    )
    parser.add_argument(
        '--config_path',
        type=str,
        default='./configs/unified_model_config.json',
        help='Path to model configuration file'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        choices=['cuda', 'cpu'],
        help='Device to run inference on'
    )
    parser.add_argument(
        '--save_format',
        type=str,
        default='npy',
        choices=['npy', 'json', 'both'],
        help='Format to save embeddings'
    )
    parser.add_argument(
        '--consolidated',
        action='store_true',
        help='Save all embeddings in a single consolidated file'
    )
    parser.add_argument(
        '--summary',
        action='store_true',
        help='Save processing summary to JSON file'
    )
    
    args = parser.parse_args()
    
    try:
        # Extract embeddings
        results = extract_directory_embeddings(
            video_dir=args.video_dir,
            output_dir=args.output_dir,
            config_path=args.config_path,
            device=args.device,
            save_format=args.save_format,
            consolidated=args.consolidated
        )
        
        # Save summary if requested
        if args.summary:
            summary_path = Path(args.output_dir) / 'processing_summary.json'
            with open(summary_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Processing summary saved to: {summary_path}")
        
        # Print final summary
        print("\n" + "="*60)
        print("EMBEDDING EXTRACTION RESULTS")
        print("="*60)
        print(f"Video directory: {args.video_dir}")
        print(f"Output directory: {args.output_dir}")
        print(f"Total videos: {results['total_videos']}")
        print(f"Successfully processed: {results['processed']}")
        print(f"Failed: {results['failed']}")
        
        if results['failed'] > 0:
            print(f"Failed videos: {results['failed_videos']}")
        
        print("="*60)
        
    except Exception as e:
        logger.error(f"Embedding extraction failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()