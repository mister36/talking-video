"""
Setup script for downloading InfiniteTalk models
Run this script to download all required model weights
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n📦 {description}")
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Success")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        return False

def check_huggingface_cli():
    """Check if huggingface-cli is available"""
    try:
        subprocess.run(["huggingface-cli", "--version"], check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def install_huggingface_hub():
    """Install huggingface_hub if not available"""
    print("📦 Installing huggingface_hub...")
    return run_command([sys.executable, "-m", "pip", "install", "huggingface_hub"], "Installing huggingface_hub")

def download_models():
    """Download all required models"""
    
    # Create weights directory
    weights_dir = Path("weights")
    weights_dir.mkdir(exist_ok=True)
    
    print("🚀 Starting model download process...")
    print("This may take a while depending on your internet connection.")
    
    # Check if huggingface-cli is available
    if not check_huggingface_cli():
        print("⚠️  huggingface-cli not found, installing huggingface_hub...")
        if not install_huggingface_hub():
            print("❌ Failed to install huggingface_hub")
            return False
    
    models_to_download = [
        {
            "repo": "Wan-AI/Wan2.1-I2V-14B-480P",
            "local_dir": "weights/Wan2.1-I2V-14B-480P",
            "description": "Downloading Wan2.1-I2V-14B-480P base model"
        },
        {
            "repo": "TencentGameMate/chinese-wav2vec2-base",
            "local_dir": "weights/chinese-wav2vec2-base",
            "description": "Downloading Chinese Wav2Vec2 audio encoder"
        },
        {
            "repo": "MeiGen-AI/InfiniteTalk",
            "local_dir": "weights/InfiniteTalk",
            "description": "Downloading InfiniteTalk weights"
        }
    ]
    
    success_count = 0
    
    for model in models_to_download:
        if Path(model["local_dir"]).exists():
            print(f"⏭️  Skipping {model['repo']} (already exists)")
            success_count += 1
            continue
            
        cmd = [
            "huggingface-cli", "download",
            model["repo"],
            "--local-dir", model["local_dir"]
        ]
        
        if run_command(cmd, model["description"]):
            success_count += 1
        else:
            print(f"❌ Failed to download {model['repo']}")
    
    # Download specific model.safetensors for wav2vec2
    wav2vec_file = "weights/chinese-wav2vec2-base/model.safetensors"
    if not Path(wav2vec_file).exists():
        cmd = [
            "huggingface-cli", "download",
            "TencentGameMate/chinese-wav2vec2-base",
            "model.safetensors",
            "--revision", "refs/pr/1",
            "--local-dir", "weights/chinese-wav2vec2-base"
        ]
        run_command(cmd, "Downloading Wav2Vec2 safetensors file")
    
    print(f"\n📊 Download Summary: {success_count}/{len(models_to_download)} models downloaded")
    
    if success_count == len(models_to_download):
        print("🎉 All models downloaded successfully!")
        return True
    else:
        print("⚠️  Some models failed to download. Please check the errors above.")
        return False

def verify_installation():
    """Verify that all required models are present"""
    print("\n🔍 Verifying installation...")
    
    required_paths = [
        "weights/Wan2.1-I2V-14B-480P",
        "weights/chinese-wav2vec2-base",
        "weights/InfiniteTalk"
    ]
    
    all_present = True
    for path in required_paths:
        if Path(path).exists():
            print(f"✅ {path}")
        else:
            print(f"❌ {path}")
            all_present = False
    
    return all_present

def main():
    """Main setup function"""
    print("🚀 InfiniteTalk Model Setup")
    print("=" * 40)
    
    # Download models
    if download_models():
        # Verify installation
        if verify_installation():
            print("\n🎉 Setup completed successfully!")
            print("\nYou can now run the FastAPI server with:")
            print("python main.py")
        else:
            print("\n❌ Setup incomplete. Some models are missing.")
            sys.exit(1)
    else:
        print("\n❌ Setup failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
