# InfiniteTalk Lip-Sync Video Generation API

A FastAPI server for generating lip-sync videos using the InfiniteTalk model. This API takes an image and audio file as input and generates a video where the character in the image lip-syncs to the provided audio.

## Features

-   🎬 **Image-to-Video Generation**: Create talking videos from a single image and audio
-   🎯 **Accurate Lip Synchronization**: Advanced lip-sync using InfiniteTalk model
-   ⚡ **Fast API**: RESTful API with automatic documentation
-   🔧 **Configurable**: Adjustable resolution, quality, and generation parameters
-   🚀 **RunPod Ready**: Optimized for deployment on RunPod GPU instances

## Requirements

-   Python 3.10+
-   CUDA-compatible GPU (12GB+ VRAM recommended)
-   FFmpeg
-   50GB+ storage for model weights

## Quick Start

### 1. Clone and Setup

```bash
git clone <your-repo>
cd talking-video
```

### 2. Install Dependencies

```bash
# Install PyTorch with CUDA support
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121

# Install XFormers
pip install -U xformers==0.0.28 --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install -r requirements.txt

# Install FFmpeg (Ubuntu/Debian)
sudo apt update && sudo apt install -y ffmpeg

# Or using conda
conda install -c conda-forge ffmpeg
```

### 3. Download Models

```bash
python setup.py
```

This will download all required models (~30GB total):

-   Wan2.1-I2V-14B-480P (base video generation model)
-   chinese-wav2vec2-base (audio encoder)
-   InfiniteTalk weights (lip-sync conditioning)

### 4. Start the Server

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

## API Usage

### Generate Video Endpoint

**POST** `/generate-video`

Upload an image and audio file to generate a lip-sync video.

#### Parameters

-   `image` (file): Input image (JPG, PNG, WebP)
-   `audio` (file): Input audio (WAV, MP3, FLAC)
-   `resolution` (form): Output resolution ("480p" or "720p", default: "480p")
-   `sample_steps` (form): Denoising steps (1-100, default: 40)
-   `audio_cfg_scale` (form): Audio guidance scale (1.0-10.0, default: 4.0)
-   `max_duration` (form): Maximum video duration in seconds (1-300, default: 40)

#### Example using curl

```bash
curl -X POST "http://localhost:8000/generate-video" \
  -F "image=@photo.jpg" \
  -F "audio=@speech.wav" \
  -F "resolution=720p" \
  -F "sample_steps=50" \
  -F "audio_cfg_scale=4.5" \
  --output result.mp4
```

#### Example using Python

```python
import requests

url = "http://localhost:8000/generate-video"

files = {
    'image': open('photo.jpg', 'rb'),
    'audio': open('speech.wav', 'rb')
}

data = {
    'resolution': '720p',
    'sample_steps': 40,
    'audio_cfg_scale': 4.0,
    'max_duration': 30
}

response = requests.post(url, files=files, data=data)

with open('output.mp4', 'wb') as f:
    f.write(response.content)
```

## Configuration

### Environment Variables

-   `PORT`: Server port (default: 8000)
-   `HOST`: Server host (default: 0.0.0.0)

### Model Parameters

Edit the `MODEL_CONFIG` in `app.py` to adjust default settings:

```python
MODEL_CONFIG = {
    "size": "infinitetalk-480",  # or "infinitetalk-720"
    "sample_steps": 40,
    "sample_text_guide_scale": 5.0,
    "sample_audio_guide_scale": 4.0,
    "max_frame_num": 1000,  # ~40 seconds at 25fps
    "num_persistent_param_in_dit": 0,  # For low VRAM
}
```

## Performance Optimization

### Memory Optimization

For lower VRAM usage:

-   Use `480p` resolution instead of `720p`
-   Set `num_persistent_param_in_dit: 0`
-   Reduce `sample_steps` to 20-30

### Speed Optimization

-   Use LoRA weights for faster inference (4-8 steps)
-   Enable TeaCache acceleration
-   Use quantized models (FP8/INT8)

## RunPod Deployment

### 1. Create RunPod Template

Use these settings:

-   **Container Image**: `pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel`
-   **GPU**: RTX 4090 or A100 (24GB+ VRAM recommended)
-   **Disk Space**: 100GB+

### 2. Setup Script

```bash
#!/bin/bash
cd /workspace
git clone <your-repo> talking-video
cd talking-video

# Install dependencies
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install -U xformers==0.0.28 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# Install FFmpeg
apt update && apt install -y ffmpeg

# Download models
python setup.py

# Start server
uvicorn app:app --host 0.0.0.0 --port 8000
```

### 3. Access the API

The API will be accessible through RunPod's public endpoint on port 8000.

## API Documentation

Once the server is running, visit:

-   **Interactive Docs**: `http://localhost:8000/docs`
-   **OpenAPI Spec**: `http://localhost:8000/openapi.json`

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**

    - Reduce resolution to 480p
    - Lower sample_steps
    - Set `num_persistent_param_in_dit: 0`

2. **Model Download Fails**

    - Check internet connection
    - Ensure sufficient disk space (50GB+)
    - Try downloading models manually

3. **FFmpeg Not Found**

    - Install FFmpeg: `sudo apt install ffmpeg`
    - Or use conda: `conda install -c conda-forge ffmpeg`

4. **Slow Generation**
    - Use GPU with more VRAM
    - Consider using LoRA weights
    - Enable quantization

### Performance Monitoring

Check GPU usage:

```bash
nvidia-smi
```

Monitor server logs:

```bash
tail -f /var/log/infinitetalk.log
```

## File Structure

```
talking-video/
├── app.py                     # FastAPI server
├── generate_infinitetalk.py   # Video generation script
├── setup.py                   # Model download script
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── uploads/                   # Temporary upload directory
├── outputs/                   # Generated video outputs
├── temp/                      # Temporary files
└── weights/                   # Model weights (created by setup.py)
    ├── Wan2.1-I2V-14B-480P/
    ├── chinese-wav2vec2-base/
    └── InfiniteTalk/
```

## License

This project uses the InfiniteTalk model which is licensed under Apache 2.0. See the original [InfiniteTalk repository](https://github.com/MeiGen-AI/InfiniteTalk) for more details.

## Citation

If you use this API in your research, please cite the original InfiniteTalk paper:

```bibtex
@misc{yang2025infinitetalkaudiodrivenvideogeneration,
    title={InfiniteTalk: Audio-driven Video Generation for Sparse-Frame Video Dubbing},
    author={Shaoshu Yang and Zhe Kong and Feng Gao and Meng Cheng and Xiangyu Liu and Yong Zhang and Zhuoliang Kang and Wenhan Luo and Xunliang Cai and Ran He and Xiaoming Wei},
    year={2025},
    eprint={2508.14033},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2508.14033},
}
```
