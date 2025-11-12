#!/usr/bin/env python3
"""
Download image generation model from HuggingFace for Triton deployment
"""

import argparse
from pathlib import Path

def download_model(model_name: str, output_dir: str):
    """Download Stable Diffusion XL model from HuggingFace"""
    try:
        from huggingface_hub import snapshot_download # type: ignore
    except ImportError:
        print("Error: Required packages not installed")
        print("Install with: pip install huggingface-hub")
        exit(1)
    
    print(f"Downloading model: {model_name}")
    print(f"Output directory: {output_dir}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("\nDownloading model files...")
    print("This may take several minutes depending on model size...")
    print("Note: Files are downloaded without loading into memory to avoid OOM errors")
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                print(f"\nRetry attempt {attempt + 1}/{max_retries}...")
            
            # Use snapshot_download() instead of DiffusionPipeline.from_pretrained()
            # This downloads files directly without loading the model into RAM.
            # Loading a 7GB model with from_pretrained() requires 15-20GB RAM and causes OOM.
            # snapshot_download() only needs ~100MB for the download buffer.
            snapshot_download(
                repo_id=model_name,
                local_dir=output_dir
            )
            
            print("✓ Model downloaded")
            break  # Success, exit retry loop
            
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"\n⚠ Download error: {str(e)[:100]}...")
                print("Retrying in 5 seconds...")
                import time
                time.sleep(5)
            else:
                print(f"\nError: Failed to download after {max_retries} attempts")
                print("\nTroubleshooting:")
                print("1. Check your internet connection")
                print("2. Try again later (HuggingFace servers may be busy)")
                print("3. Clear cache: rm -rf ~/.cache/huggingface/")
                raise e
    
    # Print summary
    model_size = sum(f.stat().st_size for f in output_path.rglob('*') if f.is_file()) / 1e9
    
    print("\n" + "="*50)
    print("Download Complete!")
    print("="*50)
    print(f"Model location: {output_dir}")
    print(f"Model size: {model_size:.2f} GB")

def main():
    parser = argparse.ArgumentParser(description="Download image generation model for Triton")
    parser.add_argument(
        "--model",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="HuggingFace model name (default: stabilityai/stable-diffusion-xl-base-1.0)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./base_model",
        help="Output directory (default: ./base_model)"
    )
    
    args = parser.parse_args()
    
    download_model(args.model, args.output)


if __name__ == "__main__":
    main()

