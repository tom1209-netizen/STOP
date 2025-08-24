#!/usr/bin/env python3
"""
Inference Script for Unified TempMe-STOP Model

This script provides the main entry point for temporal video event retrieval
using the unified TempMe-STOP model.

Usage:
    python inference.py --video_path path/to/video.mp4 --query "The dog is running"
    
API:
    retrieve_event(video_path: str, query: str) -> List[Frames]
"""

import os
import sys
import argparse
import logging
import torch
import json
from typing import List, Dict, Any
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from unified_model import UnifiedTempMeSTOPModel, create_unified_model

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def retrieve_event(video_path: str, query: str, model: UnifiedTempMeSTOPModel = None, 
                  top_k: int = 5, device: str = 'cuda', config_path: str = None) -> List[int]:
    """
    Main API function for temporal video event retrieval.
    
    Args:
        video_path: Path to input video file
        query: Natural language query describing the temporal event
        model: Pre-loaded unified model (optional)
        top_k: Number of top relevant frames to return
        device: Device to run inference on
        
    Returns:
        List of frame indices corresponding to the query
        
    Example:
        >>> frames = retrieve_event("video.mp4", "The dog is running")
        >>> print(f"Relevant frames: {frames}")
    """
    # Load model if not provided
    if model is None:
        logger.info("Loading unified TempMe-STOP model...")
        model = create_unified_model(config_path=config_path, device=device)
        model.to(device)
        model.eval()
    
    # Validate inputs
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    if not query.strip():
        raise ValueError("Query cannot be empty")
    
    logger.info(f"Processing video: {video_path}")
    logger.info(f"Query: '{query}'")
    
    # Run inference
    start_time = time.time()
    try:
        relevant_frames = model.retrieve_event(video_path, query, top_k=top_k)
        inference_time = time.time() - start_time
        
        logger.info(f"Inference completed in {inference_time:.2f} seconds")
        logger.info(f"Found {len(relevant_frames)} relevant frames: {relevant_frames}")
        
        return relevant_frames
        
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        raise


def batch_retrieve_events(video_queries: List[Dict[str, str]], model: UnifiedTempMeSTOPModel = None,
                         top_k: int = 5, device: str = 'cuda', config_path: str = None) -> List[Dict[str, Any]]:
    """
    Batch inference for multiple video-query pairs.
    
    Args:
        video_queries: List of dicts with 'video_path' and 'query' keys
        model: Pre-loaded unified model (optional)
        top_k: Number of top relevant frames to return per query
        device: Device to run inference on
        
    Returns:
        List of results with 'video_path', 'query', 'frames', 'inference_time'
    """
    # Load model once for batch processing
    if model is None:
        logger.info("Loading unified TempMe-STOP model for batch processing...")
        model = create_unified_model(config_path=config_path, device=device)
        model.to(device)
        model.eval()
    
    results = []
    
    for i, item in enumerate(video_queries):
        video_path = item['video_path']
        query = item['query']
        
        logger.info(f"Processing batch item {i+1}/{len(video_queries)}")
        
        try:
            start_time = time.time()
            frames = model.retrieve_event(video_path, query, top_k=top_k)
            inference_time = time.time() - start_time
            
            results.append({
                'video_path': video_path,
                'query': query,
                'frames': frames,
                'inference_time': inference_time,
                'status': 'success'
            })
            
        except Exception as e:
            logger.error(f"Error processing {video_path} with query '{query}': {e}")
            results.append({
                'video_path': video_path,
                'query': query,
                'frames': [],
                'inference_time': 0.0,
                'status': 'error',
                'error': str(e)
            })
    
    return results


def demo_inference():
    """
    Demonstration of the unified model with sample inputs and outputs.
    """
    print("=" * 60)
    print("UNIFIED TEMPME-STOP MODEL DEMO")
    print("=" * 60)
    
    # Sample video paths and queries (these would be real files in practice)
    sample_data = [
        {
            'video_path': './sample_videos/dog_running.mp4',
            'query': 'The dog is running in the park'
        },
        {
            'video_path': './sample_videos/person_walking.mp4', 
            'query': 'A person walking down the street'
        },
        {
            'video_path': './sample_videos/car_driving.mp4',
            'query': 'A car driving on the highway'
        }
    ]
    
    print("Sample inputs:")
    for i, item in enumerate(sample_data, 1):
        print(f"  {i}. Video: {item['video_path']}")
        print(f"     Query: '{item['query']}'")
    
    print("\nSample expected outputs:")
    sample_outputs = [
        [45, 46, 47, 48, 49],  # Frame indices where dog is running
        [23, 24, 25, 26, 27],  # Frame indices where person is walking  
        [67, 68, 69, 70, 71],  # Frame indices where car is driving
    ]
    
    for i, frames in enumerate(sample_outputs, 1):
        print(f"  {i}. Relevant frames: {frames}")
    
    print("\nNOTE: This is a demonstration. To run with real videos,")
    print("      use: python inference.py --video_path <path> --query <query>")
    
    print("\nPipeline Overview:")
    print("  Input Video → Frame Sampling → TempMe Compression (N→12) → STOP Retrieval → Output Frames")
    
    print("\nArchitecture Details:")
    print("  1. Video Preprocessing: Extract frames from raw video")
    print("  2. TempMe Compression: Reduce N frames to 12 representatives") 
    print("  3. STOP Retrieval: Query-based similarity computation")
    print("  4. Output: Top-k most relevant frame indices")


def main():
    """Main entry point for the inference script."""
    parser = argparse.ArgumentParser(
        description="Temporal Video Event Retrieval using Unified TempMe-STOP Model"
    )
    parser.add_argument(
        '--video_path', 
        type=str, 
        help='Path to input video file'
    )
    parser.add_argument(
        '--query', 
        type=str, 
        help='Natural language query describing the temporal event'
    )
    parser.add_argument(
        '--top_k', 
        type=int, 
        default=5, 
        help='Number of top relevant frames to return (default: 5)'
    )
    parser.add_argument(
        '--device', 
        type=str, 
        default='cuda' if torch.cuda.is_available() else 'cpu',
        choices=['cuda', 'cpu'],
        help='Device to run inference on (default: cuda if available)'
    )
    parser.add_argument(
        '--batch_file',
        type=str,
        help='JSON file with batch of video-query pairs'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        help='Output file to save results (JSON format)'
    )
    parser.add_argument(
        '--config_path',
        type=str,
        default='./configs/unified_model_config.json',
        help='Path to model configuration file'
    )
    parser.add_argument(
        '--demo',
        action='store_true',
        help='Run demonstration with sample inputs/outputs'
    )
    
    args = parser.parse_args()
    
    # Run demo mode
    if args.demo:
        demo_inference()
        return
    
    # Batch processing mode
    if args.batch_file:
        if not os.path.exists(args.batch_file):
            logger.error(f"Batch file not found: {args.batch_file}")
            return
        
        with open(args.batch_file, 'r') as f:
            video_queries = json.load(f)
        
        logger.info(f"Processing {len(video_queries)} video-query pairs from {args.batch_file}")
        
        results = batch_retrieve_events(video_queries, top_k=args.top_k, device=args.device, config_path=args.config_path)
        
        # Save results
        output_file = args.output_file or 'inference_results.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {output_file}")
        
        # Print summary
        successful = sum(1 for r in results if r['status'] == 'success')
        logger.info(f"Batch processing complete: {successful}/{len(results)} successful")
        
        return
    
    # Single inference mode
    if not args.video_path or not args.query:
        logger.error("Both --video_path and --query are required for single inference")
        parser.print_help()
        return
    
    try:
        # Run single inference
        frames = retrieve_event(
            video_path=args.video_path,
            query=args.query, 
            top_k=args.top_k,
            device=args.device,
            config_path=args.config_path
        )
        
        # Print results
        print("\n" + "="*50)
        print("INFERENCE RESULTS")
        print("="*50)
        print(f"Video: {args.video_path}")
        print(f"Query: '{args.query}'")
        print(f"Relevant frames: {frames}")
        print(f"Number of frames: {len(frames)}")
        
        # Save results if output file specified
        if args.output_file:
            result = {
                'video_path': args.video_path,
                'query': args.query,
                'frames': frames,
                'top_k': args.top_k
            }
            with open(args.output_file, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"Results saved to: {args.output_file}")
        
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()