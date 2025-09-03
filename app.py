"""
FastAPI server for InfiniteTalk lip-sync video generation
Designed for RunPod deployment
"""

import os
import json
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Dict, List
import uuid
import subprocess
import asyncio
import logging
from datetime import datetime, timezone
from enum import Enum
import sys
import threading

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import InfiniteTalk dependencies (these will be available after model setup)
import torch
import torchaudio
import librosa
import numpy as np
from PIL import Image
import cv2

app = FastAPI(
    title="InfiniteTalk Lip-Sync API",
    description="Generate lip-sync videos using InfiniteTalk model",
    version="1.0.0"
)

# Add CORS middleware for web frontend compatibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Job system models
class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class JobRequest(BaseModel):
    resolution: str = Field(default="480p", description="Output resolution: 480p or 720p")
    sample_steps: int = Field(default=4, description="Number of sampling steps (4 for lightx2v LoRA)")
    audio_cfg_scale: float = Field(default=4.0, description="Audio CFG scale")
    max_duration: int = Field(default=60, description="Maximum video duration in seconds")

class JobResponse(BaseModel):
    job_id: str = Field(description="Unique job identifier")
    status: JobStatus = Field(description="Current job status")
    message: str = Field(description="Status message")
    created_at: datetime = Field(description="Job creation timestamp")
    completed_at: Optional[datetime] = Field(default=None, description="Job completion timestamp")
    download_url: Optional[str] = Field(default=None, description="Download URL when completed")
    error_message: Optional[str] = Field(default=None, description="Error message if failed")
    progress: Optional[float] = Field(default=None, description="Processing progress (0-100)")

class JobInfo(BaseModel):
    job_id: str
    status: JobStatus
    image_path: str
    audio_path: str
    output_path: str
    request_params: JobRequest
    created_at: datetime
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    progress: Optional[float] = None

def is_lora_enabled():
    """Check if LoRA is enabled based on configuration"""
    return (
        MODEL_CONFIG.get("lora_dir") and 
        os.path.exists(MODEL_CONFIG["lora_dir"]) and 
        MODEL_CONFIG.get("lora_scale", 0) > 0
    )

def get_optimal_guide_scales():
    """Get optimal guide scale values based on whether LoRA is enabled"""
    if is_lora_enabled():
        return {
            "sample_text_guide_scale": 1.0,
            "sample_audio_guide_scale": 2.0
        }
    else:
        return {
            "sample_text_guide_scale": 5.0,
            "sample_audio_guide_scale": 4.0
        }

def update_guide_scales():
    """Update guide scale values in MODEL_CONFIG based on LoRA usage"""
    optimal_scales = get_optimal_guide_scales()
    MODEL_CONFIG.update(optimal_scales)

def update_model_paths_for_cache():
    """Update model paths to point to cache locations if they exist"""
    global MODEL_CONFIG
    
    # Check if we should use cache paths
    hf_home = os.environ.get('HF_HOME')
    if not hf_home:
        return  # Use default local paths
    
    logger.info("Detected HF_HOME environment variable, checking for models in cache...")
    
    # Map of local paths to cache locations
    cache_mappings = [
        {
            "local_key": "ckpt_dir",
            "local_default": "InfiniteTalk/weights/Wan2.1-I2V-14B-480P",
            "repo": "Wan-AI/Wan2.1-I2V-14B-480P"
        },
        {
            "local_key": "wav2vec_dir", 
            "local_default": "InfiniteTalk/weights/chinese-wav2vec2-base",
            "repo": "TencentGameMate/chinese-wav2vec2-base"
        },
        {
            "local_key": "infinitetalk_dir",
            "local_default": "InfiniteTalk/weights/InfiniteTalk/single/single/infinitetalk.safetensors",
            "repo": "MeiGen-AI/InfiniteTalk",
            "filename": "single/single/infinitetalk.safetensors"
        }
    ]
    
    for mapping in cache_mappings:
        local_path = mapping["local_default"]
        cache_path = get_hf_cache_path(mapping["repo"], mapping.get("filename"))
        
        # If cache path exists and local path doesn't, use cache path
        if os.path.exists(cache_path) and not os.path.exists(local_path):
            MODEL_CONFIG[mapping["local_key"]] = cache_path
            logger.info(f"Using cache path for {mapping['local_key']}: {cache_path}")
        elif os.path.exists(cache_path) and os.path.exists(local_path):
            # Both exist, prefer local if it's not a broken symlink
            if os.path.islink(local_path) and not os.path.exists(os.readlink(local_path)):
                MODEL_CONFIG[mapping["local_key"]] = cache_path
                logger.info(f"Local path is broken symlink, using cache for {mapping['local_key']}: {cache_path}")


# Global variables for model paths and configurations
MODEL_CONFIG = {
    "ckpt_dir": "InfiniteTalk/weights/Wan2.1-I2V-14B-480P",
    "wav2vec_dir": "InfiniteTalk/weights/chinese-wav2vec2-base", 
    "infinitetalk_dir": "InfiniteTalk/weights/InfiniteTalk/single/single/infinitetalk.safetensors",
    "lora_dir": "InfiniteTalk/weights/Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors",
    "lora_scale": 1.0,
    "size": "infinitetalk-480",  # or infinitetalk-720
    "sample_steps": 4,  # Reduced to 4 for lightx2v LoRA
    "mode": "streaming",
    "motion_frame": 9,
    "stext_guide_scale": 5.0,  # Will be updated based on LoRA usage
    "sample_audio_guide_scale": 4.0,  # Will be updated based on LoRA usage
    "max_frame_num": 1500,  # 60 seconds at 25fps
    "keep_models_loaded": True,  # Set to False to save VRAM by offloading models after each generation
    # Removed num_persistent_param_in_dit - A100 has enough VRAM to keep everything on GPU
}

# Directory setup
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
TEMP_DIR = Path("temp")
JOBS_DIR = Path("jobs")

for dir_path in [UPLOAD_DIR, OUTPUT_DIR, TEMP_DIR, JOBS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Simple job storage - in a real system you'd use a database
jobs = {}

class JobManager:
    """Simple job management system"""
    
    @staticmethod
    def create_job(request_params: JobRequest, image_path: str, audio_path: str, output_path: str) -> str:
        """Create a new job and return job ID"""
        job_id = str(uuid.uuid4())
        
        job_data = {
            "id": job_id,
            "status": "queued",
            "type": "video",
            "image_path": image_path,
            "audio_path": audio_path,
            "output_path": output_path,
            "request_params": request_params.model_dump(),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "completed_at": None,
            "error": None,
            "progress": 0.0
        }
        
        jobs[job_id] = job_data
        
        # Save job to disk for persistence
        job_file = JOBS_DIR / f"{job_id}.json"
        with open(job_file, "w") as f:
            json.dump(job_data, f, indent=2)
        
        return job_id
    
    @staticmethod
    def update_job_status(job_id: str, status: str, error: str = None, progress: float = None):
        """Update job status"""
        if job_id not in jobs:
            return False
        
        jobs[job_id]["status"] = status
        if error:
            jobs[job_id]["error"] = error
        if progress is not None:
            jobs[job_id]["progress"] = progress
        if status == "completed":
            jobs[job_id]["completed_at"] = datetime.now(timezone.utc).isoformat()
        
        # Save updated job to disk
        job_file = JOBS_DIR / f"{job_id}.json"
        with open(job_file, "w") as f:
            json.dump(jobs[job_id], f, indent=2)
        
        return True
    
    @staticmethod
    def get_job(job_id: str) -> dict:
        """Get job details"""
        return jobs.get(job_id)
    
    @staticmethod
    def load_jobs_from_disk():
        """Load existing jobs from disk on startup"""
        for job_file in JOBS_DIR.glob("*.json"):
            try:
                with open(job_file, "r") as f:
                    job_data = json.load(f)
                    jobs[job_data["id"]] = job_data
            except Exception as e:
                logger.warning(f"Could not load job file {job_file}: {e}")

# In-memory job storage (for backward compatibility)
active_jobs: Dict[str, JobInfo] = {}
job_storage_file = JOBS_DIR / "jobs.json"

# Load existing jobs from storage
if job_storage_file.exists():
    try:
        with open(job_storage_file, 'r') as f:
            jobs_data = json.load(f)
            for job_id, job_data in jobs_data.items():
                # Convert datetime strings back to datetime objects
                job_data['created_at'] = datetime.fromisoformat(job_data['created_at'])
                if job_data.get('completed_at'):
                    job_data['completed_at'] = datetime.fromisoformat(job_data['completed_at'])
                active_jobs[job_id] = JobInfo(**job_data)
    except Exception as e:
        print(f"Warning: Failed to load existing jobs: {e}")

def save_jobs_to_storage():
    """Save current jobs to persistent storage"""
    try:
        jobs_data = {}
        for job_id, job_info in active_jobs.items():
            job_dict = job_info.model_dump()
            # Convert datetime objects to strings for JSON serialization
            job_dict['created_at'] = job_info.created_at.isoformat()
            if job_info.completed_at:
                job_dict['completed_at'] = job_info.completed_at.isoformat()
            jobs_data[job_id] = job_dict
        
        with open(job_storage_file, 'w') as f:
            json.dump(jobs_data, f, indent=2)
    except Exception as e:
        print(f"Warning: Failed to save jobs to storage: {e}")


def process_video_generation(job_id: str):
    """Process video generation job"""
    try:
        JobManager.update_job_status(job_id, "processing", progress=0.0)
        logger.info(f"Starting video generation for job {job_id}")
        
        job = JobManager.get_job(job_id)
        if not job:
            logger.error(f"Job {job_id} not found")
            return
        
        # Update progress
        JobManager.update_job_status(job_id, "processing", progress=10.0)
        
        # Generate video using the existing generator
        result_path = generator.generate_video(
            job["image_path"],
            job["audio_path"],
            job["output_path"]
        )
        
        JobManager.update_job_status(job_id, "processing", progress=90.0)
        
        # Verify output file exists
        if not os.path.exists(result_path):
            raise Exception("Video generation completed but output file not found")
        
        # Update job status to completed
        JobManager.update_job_status(job_id, "completed", progress=100.0)
        logger.info(f"Video generation completed for job {job_id}")
        
        # Clean up uploaded files
        try:
            if os.path.exists(job["image_path"]) and UPLOAD_DIR.name in job["image_path"]:
                os.unlink(job["image_path"])
            if os.path.exists(job["audio_path"]) and UPLOAD_DIR.name in job["audio_path"]:
                os.unlink(job["audio_path"])
        except:
            pass  # Don't fail if cleanup fails
            
    except Exception as e:
        error_msg = f"Video generation failed: {str(e)}"
        logger.error(f"Job {job_id} failed: {error_msg}")
        JobManager.update_job_status(job_id, "failed", error=error_msg)
        
        # Clean up files on error
        job = JobManager.get_job(job_id)
        if job:
            try:
                for file_path in [job["image_path"], job["audio_path"], job["output_path"]]:
                    if os.path.exists(file_path):
                        os.unlink(file_path)
            except:
                pass

async def process_video_generation_legacy(job_id: str):
    """Legacy background task to process video generation (for backward compatibility)"""
    job_info = active_jobs.get(job_id)
    if not job_info:
        return
    
    try:
        # Update job status to processing
        job_info.status = JobStatus.PROCESSING
        job_info.progress = 0.0
        save_jobs_to_storage()
        
        # Generate video using the existing generator
        job_info.progress = 10.0
        save_jobs_to_storage()
        
        result_path = generator.generate_video(
            job_info.image_path,
            job_info.audio_path,
            job_info.output_path
        )
        
        job_info.progress = 90.0
        save_jobs_to_storage()
        
        # Verify output file exists
        if not os.path.exists(result_path):
            raise Exception("Video generation completed but output file not found")
        
        # Update job status to completed
        job_info.status = JobStatus.COMPLETED
        job_info.completed_at = datetime.now(timezone.utc)
        job_info.progress = 100.0
        save_jobs_to_storage()
        
        # Clean up uploaded files
        try:
            if os.path.exists(job_info.image_path):
                os.unlink(job_info.image_path)
            if os.path.exists(job_info.audio_path):
                os.unlink(job_info.audio_path)
        except:
            pass  # Don't fail if cleanup fails
            
    except Exception as e:
        # Update job status to failed
        job_info.status = JobStatus.FAILED
        job_info.error_message = str(e)
        job_info.completed_at = datetime.now(timezone.utc)
        save_jobs_to_storage()
        
        # Clean up files on error
        try:
            for file_path in [job_info.image_path, job_info.audio_path, job_info.output_path]:
                if os.path.exists(file_path):
                    os.unlink(file_path)
        except:
            pass

class InfiniteTalkGenerator:
    """Wrapper class for InfiniteTalk model inference with persistent pipeline"""
    
    def __init__(self):
        self.model_loaded = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.local_rank = 0  # Single device setup
        self.rank = 0
        self.pipeline = None
        self.wav2vec_feature_extractor = None
        self.audio_encoder = None
        
    def load_model(self):
        """Load the InfiniteTalk model pipeline directly into memory"""
        if self.model_loaded:
            return
        
        # Update model paths for cache before checking if models exist
        update_model_paths_for_cache()
            
        # Check if model weights exist
        if not check_models_exist():
            # Check if models are currently downloading
            with download_lock:
                status = download_status["status"]
            
            if status == "downloading":
                raise FileNotFoundError("Models are currently downloading/fixing. Please wait for completion. Check /model-status for progress.")
            elif status == "failed":
                with download_lock:
                    error = download_status["error"]
                raise FileNotFoundError(f"Model download/fix failed: {error}. Please check server logs.")
            else:
                # Try immediate fix first, then background download if that fails
                logger.info("Models not found during load_model, attempting immediate fix...")
                try:
                    if fix_broken_models():
                        logger.info("Models fixed successfully during load_model")
                        # Update paths again after fix
                        update_model_paths_for_cache()
                    else:
                        raise Exception("Immediate fix failed")
                except Exception as e:
                    logger.warning(f"Immediate fix failed: {e}")
                    logger.info("Starting background download/fix...")
                    start_background_download()
                    raise FileNotFoundError("Model download/fix started in background. Please wait for completion. Check /model-status for progress.")
        
        # Double-check that models now exist
        required_paths = [
            MODEL_CONFIG["ckpt_dir"],
            MODEL_CONFIG["wav2vec_dir"],
            MODEL_CONFIG["infinitetalk_dir"],
            MODEL_CONFIG["lora_dir"]
        ]
        
        for path in required_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Model weights not found at: {path}")
        
        logger.info(f"Loading InfiniteTalk pipeline directly on {self.device}...")
        
        # Import InfiniteTalk dependencies
        try:
            import sys
            sys.path.append('InfiniteTalk')
            import wan
            from wan.configs import WAN_CONFIGS
            from transformers import Wav2Vec2FeatureExtractor
            from src.audio_analysis.wav2vec2 import Wav2Vec2Model
        except ImportError as e:
            raise ImportError(f"Failed to import InfiniteTalk dependencies: {e}")
        
        # Initialize pipeline configuration
        task = "infinitetalk-14B"
        cfg = WAN_CONFIGS[task]
        
        # Create the persistent InfiniteTalk pipeline
        self.pipeline = wan.InfiniteTalkPipeline(
            config=cfg,
            checkpoint_dir=MODEL_CONFIG["ckpt_dir"],
            quant_dir=None,
            device_id=self.local_rank,
            rank=self.rank,
            t5_fsdp=False,
            dit_fsdp=False,
            use_usp=False,
            t5_cpu=False,
            lora_dir=[MODEL_CONFIG["lora_dir"]] if MODEL_CONFIG["lora_dir"] else None,
            lora_scales=[MODEL_CONFIG["lora_scale"]],
            quant=None,
            dit_path=None,
            infinitetalk_dir=MODEL_CONFIG["infinitetalk_dir"]
        )
        
        # Enable VRAM management if configured
        if MODEL_CONFIG.get("num_persistent_param_in_dit") is not None:
            self.pipeline.vram_management = True
            self.pipeline.enable_vram_management(
                num_persistent_param_in_dit=MODEL_CONFIG["num_persistent_param_in_dit"]
            )
        
        # Initialize audio processing components
        self.wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            MODEL_CONFIG["wav2vec_dir"], local_files_only=True
        )
        self.audio_encoder = Wav2Vec2Model.from_pretrained(
            MODEL_CONFIG["wav2vec_dir"], local_files_only=True
        ).to('cpu')  # Keep on CPU as in original
        self.audio_encoder.feature_extractor._freeze_parameters()
        
        logger.info("InfiniteTalk pipeline loaded successfully and ready for inference")
        logger.info(f"Model paths: ckpt_dir={MODEL_CONFIG['ckpt_dir']}, wav2vec_dir={MODEL_CONFIG['wav2vec_dir']}, infinitetalk_dir={MODEL_CONFIG['infinitetalk_dir']}")
        
        self.model_loaded = True
    
    def _loudness_norm(self, audio_array, sr=16000, lufs=-23):
        """Normalize audio loudness"""
        import pyloudnorm as pyln
        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(audio_array)
        if abs(loudness) > 100:
            return audio_array
        normalized_audio = pyln.normalize.loudness(audio_array, loudness, lufs)
        return normalized_audio
    
    def _audio_prepare_single(self, audio_path, sample_rate=16000):
        """Prepare single audio file for processing"""
        import librosa
        human_speech_array, sr = librosa.load(audio_path, sr=sample_rate)
        human_speech_array = self._loudness_norm(human_speech_array, sr)
        return human_speech_array
    
    def _get_embedding(self, speech_array, sr=16000, device='cpu'):
        """Extract audio embeddings using wav2vec"""
        from einops import rearrange
        
        audio_duration = len(speech_array) / sr
        video_length = audio_duration * 25  # Assume the video fps is 25
        
        # wav2vec_feature_extractor
        audio_feature = np.squeeze(
            self.wav2vec_feature_extractor(speech_array, sampling_rate=sr).input_values
        )
        audio_feature = torch.from_numpy(audio_feature).float().to(device=device)
        audio_feature = audio_feature.unsqueeze(0)
        
        # audio encoder
        with torch.no_grad():
            embeddings = self.audio_encoder(audio_feature, seq_len=int(video_length), output_hidden_states=True)
        
        if len(embeddings) == 0:
            raise RuntimeError("Failed to extract audio embedding")
        
        audio_emb = torch.stack(embeddings.hidden_states[1:], dim=1).squeeze(0)
        audio_emb = rearrange(audio_emb, "b s d -> s b d")
        audio_emb = audio_emb.cpu().detach()
        return audio_emb
        
    def generate_video(self, image_path: str, audio_path: str, output_path: str) -> str:
        """Generate lip-sync video from image and audio using persistent pipeline"""
        if not self.model_loaded:
            self.load_model()
        
        # Update guide scales based on LoRA usage before generation
        update_guide_scales()
        
        import soundfile as sf
        import random
        from datetime import datetime
        
        logger.info(f"Starting direct pipeline generation for: {image_path} -> {output_path}")
        
        # Create temporary directory for audio processing
        audio_save_dir = TEMP_DIR / f"audio_{uuid.uuid4().hex}"
        audio_save_dir.mkdir(exist_ok=True)
        
        try:
            # Prepare audio
            human_speech = self._audio_prepare_single(audio_path)
            
            # Save processed audio
            sum_audio = audio_save_dir / 'sum_all.wav'
            sf.write(str(sum_audio), human_speech, 16000)
            
            # Extract audio embeddings
            audio_embedding = self._get_embedding(human_speech)
            emb_path = audio_save_dir / '1.pt'
            torch.save(audio_embedding, str(emb_path))
            
            # Prepare input for pipeline
            prompt = "A person is speaking with natural facial expressions and lip movements, captured in high quality with good lighting and clear details."
            
            input_clip = {
                'prompt': prompt,
                'cond_video': image_path,
                'cond_audio': {
                    'person1': str(emb_path)
                },
                'video_audio': str(sum_audio)
            }
            
            # Create args object for the pipeline (required for teacache and APG settings)
            class ExtraArgs:
                def __init__(self):
                    self.use_teacache = False  # Disable teacache for now
                    self.teacache_thresh = 0.2
                    self.use_apg = False  # Disable APG for now  
                    self.apg_momentum = -0.75
                    self.apg_norm_threshold = 55
                    self.size = MODEL_CONFIG["size"]
            
            extra_args = ExtraArgs()
            
            # Generate video using the persistent pipeline
            logger.info("Generating video with persistent pipeline...")
            video = self.pipeline.generate_infinitetalk(
                input_clip,
                size_buckget=MODEL_CONFIG["size"],
                motion_frame=MODEL_CONFIG["motion_frame"],
                frame_num=81,  # Default frame_num from original script
                shift=7 if MODEL_CONFIG["size"] == 'infinitetalk-480' else 11,  # Default sample_shift
                sampling_steps=MODEL_CONFIG["sample_steps"],
                text_guide_scale=MODEL_CONFIG["sample_text_guide_scale"],
                audio_guide_scale=MODEL_CONFIG["sample_audio_guide_scale"],
                seed=random.randint(0, 99999999),  # Random seed for generation
                offload_model=not MODEL_CONFIG["keep_models_loaded"],
                max_frames_num=MODEL_CONFIG["max_frame_num"],
                color_correction_strength=1.0,
                extra_args=extra_args,
            )
            
            # Save the generated video
            import sys
            sys.path.append('InfiniteTalk')
            from wan.utils.multitalk_utils import save_video_ffmpeg
            
            save_video_ffmpeg(video, output_path.replace('.mp4', ''), [str(sum_audio)], high_quality_save=False)
            
            logger.info(f"Video generation completed: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Video generation failed: {str(e)}")
            raise RuntimeError(f"Video generation failed: {str(e)}")
        finally:
            # Clean up temporary audio processing files
            try:
                import shutil
                if audio_save_dir.exists():
                    shutil.rmtree(audio_save_dir)
            except Exception as e:
                logger.warning(f"Failed to clean up temporary audio files: {e}")


def check_models_exist():
    """Check if all required models exist and are accessible (not broken symlinks)"""
    required_paths = [
        MODEL_CONFIG["ckpt_dir"],
        MODEL_CONFIG["wav2vec_dir"],
        MODEL_CONFIG["infinitetalk_dir"],
        MODEL_CONFIG["lora_dir"]
    ]
    
    for path in required_paths:
        if not os.path.exists(path):
            logger.warning(f"Model path missing: {path}")
            return False
        
        # Check if it's a broken symlink
        if os.path.islink(path):
            if not os.path.exists(os.readlink(path)):
                logger.warning(f"Broken symlink detected: {path} -> {os.readlink(path)}")
                return False
        
        # For directories, check if they contain files
        if os.path.isdir(path):
            if not any(os.listdir(path)):
                logger.warning(f"Empty directory: {path}")
                return False
    
    return True


def clone_infinitetalk_repo():
    """Clone the InfiniteTalk repository if it doesn't exist"""
    repo_dir = Path("InfiniteTalk")
    
    if repo_dir.exists():
        logger.info("InfiniteTalk repository already exists")
        return True
    
    logger.info("Cloning InfiniteTalk repository...")
    try:
        cmd = [
            "git", "clone", 
            "https://github.com/MeiGen-AI/InfiniteTalk.git",
            "InfiniteTalk"
        ]
        
        result = subprocess.run(cmd, capture_output=False, text=True, check=True)
        logger.info("InfiniteTalk repository cloned successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to clone InfiniteTalk repository: {e}")
        logger.error(f"Git stderr: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error cloning repository: {e}")
        return False


def download_lightx2v_lora():
    """Download the lightx2v LoRA file if it doesn't exist"""
    lora_path = Path(MODEL_CONFIG["lora_dir"])
    
    if lora_path.exists():
        logger.info("LightX2V LoRA already exists")
        return True
    
    # Ensure the weights directory exists
    lora_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info("Downloading LightX2V LoRA...")
    try:
        lora_url = "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors"
        
        # Use wget or curl depending on what's available
        cmd = ["wget", "-O", str(lora_path), lora_url]
        
        try:
            result = subprocess.run(cmd, capture_output=False, text=True, check=True)
            logger.info("LightX2V LoRA downloaded successfully with wget")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Try with curl if wget fails
            logger.info("wget failed, trying with curl...")
            cmd = ["curl", "-L", "-o", str(lora_path), lora_url]
            result = subprocess.run(cmd, capture_output=False, text=True, check=True)
            logger.info("LightX2V LoRA downloaded successfully with curl")
            return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to download LightX2V LoRA: {e}")
        logger.error(f"Command stderr: {e.stderr}")
        # Clean up partial download
        if lora_path.exists():
            lora_path.unlink()
        return False
    except Exception as e:
        logger.error(f"Unexpected error downloading LoRA: {e}")
        # Clean up partial download
        if lora_path.exists():
            lora_path.unlink()
        return False

def create_symlink_if_not_exists(src, dst):
    """Create a symlink from src to dst if dst doesn't exist"""
    dst_path = Path(dst)
    src_path = Path(src)
    
    # Create parent directories if they don't exist
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    
    # If destination already exists and is a working symlink or real file/dir, skip
    if dst_path.exists():
        return True
    
    # Remove broken symlinks
    if dst_path.is_symlink():
        dst_path.unlink()
    
    # Create symlink if source exists
    if src_path.exists():
        try:
            dst_path.symlink_to(src_path.resolve())
            logger.info(f"Created symlink: {dst} -> {src}")
            return True
        except Exception as e:
            logger.warning(f"Failed to create symlink {dst} -> {src}: {e}")
            return False
    
    return False


def get_hf_cache_path(repo_id, filename=None):
    """Get the Hugging Face cache path for a given repo"""
    import hashlib
    
    # Get HF cache directory from environment or default
    hf_home = os.environ.get('HF_HOME', '/workspace/.cache/huggingface')
    hub_cache = os.path.join(hf_home, 'hub')
    
    # Create repo cache directory name
    repo_cache_name = f"models--{repo_id.replace('/', '--')}"
    repo_cache_path = os.path.join(hub_cache, repo_cache_name)
    
    if filename:
        # For specific files, look in snapshots
        snapshots_dir = os.path.join(repo_cache_path, 'snapshots')
        if os.path.exists(snapshots_dir):
            # Get the latest snapshot (most recent directory)
            snapshots = [d for d in os.listdir(snapshots_dir) if os.path.isdir(os.path.join(snapshots_dir, d))]
            if snapshots:
                latest_snapshot = max(snapshots, key=lambda x: os.path.getctime(os.path.join(snapshots_dir, x)))
                return os.path.join(snapshots_dir, latest_snapshot, filename)
    
    # For directories, try to find in snapshots
    snapshots_dir = os.path.join(repo_cache_path, 'snapshots')
    if os.path.exists(snapshots_dir):
        snapshots = [d for d in os.listdir(snapshots_dir) if os.path.isdir(os.path.join(snapshots_dir, d))]
        if snapshots:
            latest_snapshot = max(snapshots, key=lambda x: os.path.getctime(os.path.join(snapshots_dir, x)))
            return os.path.join(snapshots_dir, latest_snapshot)
    
    return repo_cache_path


def fix_broken_models():
    """Fix broken or missing models by re-downloading them and creating proper symlinks"""
    logger.info("Fixing broken or missing models...")
    
    try:
        # First ensure InfiniteTalk repo is cloned
        if not clone_infinitetalk_repo():
            logger.error("Failed to clone InfiniteTalk repository")
            return False
        
        # Check each model path and fix if needed
        models_to_check = [
            {
                "path": MODEL_CONFIG["ckpt_dir"],
                "repo": "Wan-AI/Wan2.1-I2V-14B-480P",
                "name": "Wan2.1-I2V-14B-480P"
            },
            {
                "path": MODEL_CONFIG["wav2vec_dir"],
                "repo": "TencentGameMate/chinese-wav2vec2-base",
                "name": "chinese-wav2vec2-base"
            },
            {
                "path": Path(MODEL_CONFIG["infinitetalk_dir"]).parent,  # Download to parent dir
                "repo": "MeiGen-AI/InfiniteTalk",
                "name": "InfiniteTalk"
            }
        ]
        
        for model in models_to_check:
            model_path = Path(model["path"])
            needs_download = False
            
            if not model_path.exists():
                logger.info(f"Model directory missing: {model['name']}")
                needs_download = True
            elif model_path.is_dir() and not any(model_path.iterdir()):
                logger.info(f"Model directory empty: {model['name']}")
                needs_download = True
            elif model["name"] == "InfiniteTalk":
                # Special check for InfiniteTalk - check if the specific file exists
                specific_file = Path(MODEL_CONFIG["infinitetalk_dir"])
                if not specific_file.exists() or (specific_file.is_symlink() and not specific_file.resolve().exists()):
                    logger.info(f"InfiniteTalk model file missing or broken: {specific_file}")
                    needs_download = True
            
            if needs_download:
                logger.info(f"Re-downloading {model['name']}...")
                cmd = [
                    "huggingface-cli", "download",
                    model["repo"],
                    "--local-dir", str(model_path)
                ]
                
                result = subprocess.run(cmd, capture_output=False, text=True, check=True)
                logger.info(f"Successfully re-downloaded {model['name']}")
        
        # After downloading, try to create symlinks from cache to expected locations
        logger.info("Creating symlinks from HF cache to expected model paths...")
        
        # Create symlinks for each model if they exist in cache
        cache_to_local_mappings = [
            {
                "cache_path": get_hf_cache_path("Wan-AI/Wan2.1-I2V-14B-480P"),
                "local_path": MODEL_CONFIG["ckpt_dir"],
                "name": "Wan2.1-I2V-14B-480P"
            },
            {
                "cache_path": get_hf_cache_path("TencentGameMate/chinese-wav2vec2-base"),
                "local_path": MODEL_CONFIG["wav2vec_dir"],
                "name": "chinese-wav2vec2-base"
            },
            {
                "cache_path": get_hf_cache_path("MeiGen-AI/InfiniteTalk"),
                "local_path": Path(MODEL_CONFIG["infinitetalk_dir"]).parent,
                "name": "InfiniteTalk"
            }
        ]
        
        for mapping in cache_to_local_mappings:
            cache_path = mapping["cache_path"]
            local_path = mapping["local_path"]
            name = mapping["name"]
            
            if os.path.exists(cache_path):
                logger.info(f"Found {name} in cache: {cache_path}")
                if not os.path.exists(local_path) or not any(Path(local_path).iterdir()):
                    create_symlink_if_not_exists(cache_path, local_path)
            else:
                logger.warning(f"Cache path not found for {name}: {cache_path}")
        
        # Special handling for InfiniteTalk model file
        infinitetalk_file = Path(MODEL_CONFIG["infinitetalk_dir"])
        if not infinitetalk_file.exists():
            # Try to find the model file in the cache
            cache_base = get_hf_cache_path("MeiGen-AI/InfiniteTalk")
            potential_files = [
                os.path.join(cache_base, "single", "single", "infinitetalk.safetensors"),
                os.path.join(cache_base, "single", "infinitetalk.safetensors"),
                os.path.join(cache_base, "infinitetalk.safetensors"),
            ]
            
            for potential_file in potential_files:
                if os.path.exists(potential_file):
                    logger.info(f"Found InfiniteTalk model in cache: {potential_file}")
                    create_symlink_if_not_exists(potential_file, infinitetalk_file)
                    break
        
        # Also download the LoRA if missing
        if not download_lightx2v_lora():
            logger.warning("Failed to download LightX2V LoRA")
        
        # Download specific wav2vec2 safetensors if needed
        wav2vec_file = Path("InfiniteTalk/weights/chinese-wav2vec2-base/model.safetensors")
        if not wav2vec_file.exists():
            logger.info("Downloading Wav2Vec2 safetensors file...")
            cmd = [
                "huggingface-cli", "download",
                "TencentGameMate/chinese-wav2vec2-base",
                "model.safetensors",
                "--revision", "refs/pr/1",
                "--local-dir", "InfiniteTalk/weights/chinese-wav2vec2-base"
            ]
            subprocess.run(cmd, capture_output=False, text=True, check=True)
        
        # Verify models are now present
        if check_models_exist():
            logger.info("All models verified successfully after fix")
            return True
        else:
            logger.error("Models still missing after attempting to fix")
            # Log what's actually missing for debugging
            required_paths = [
                MODEL_CONFIG["ckpt_dir"],
                MODEL_CONFIG["wav2vec_dir"],
                MODEL_CONFIG["infinitetalk_dir"],
                MODEL_CONFIG["lora_dir"]
            ]
            
            for path in required_paths:
                if not os.path.exists(path):
                    logger.error(f"Still missing: {path}")
                    # Try to find alternatives in cache
                    if "Wan2.1-I2V-14B-480P" in path:
                        cache_path = get_hf_cache_path("Wan-AI/Wan2.1-I2V-14B-480P")
                        logger.info(f"Check cache: {cache_path} exists: {os.path.exists(cache_path)}")
                    elif "chinese-wav2vec2-base" in path:
                        cache_path = get_hf_cache_path("TencentGameMate/chinese-wav2vec2-base")
                        logger.info(f"Check cache: {cache_path} exists: {os.path.exists(cache_path)}")
                    elif "InfiniteTalk" in path:
                        cache_path = get_hf_cache_path("MeiGen-AI/InfiniteTalk")
                        logger.info(f"Check cache: {cache_path} exists: {os.path.exists(cache_path)}")
            
            return False
            
    except subprocess.CalledProcessError as e:
        logger.error(f"Model fix failed: {e}")
        logger.error(f"Command stderr: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during model fix: {e}")
        return False


def download_models_automatically():
    """Automatically download models using setup.py"""
    logger.info("Models not found. Starting automatic download...")
    
    try:
        # First ensure InfiniteTalk repo is cloned
        if not clone_infinitetalk_repo():
            logger.error("Failed to clone InfiniteTalk repository")
            return False
        
        # Run setup.py to download models
        setup_script = Path(__file__).parent / "setup.py"
        if not setup_script.exists():
            logger.error("setup.py not found!")
            return False
        
        logger.info("Running setup.py to download models...")
        result = subprocess.run(
            [sys.executable, str(setup_script)],
            capture_output=True,
            text=True,
            check=True
        )
        
        logger.info("Model download completed successfully")
        logger.info(f"Setup output: {result.stdout}")
        
        # Verify models are now present
        if check_models_exist():
            logger.info("All models verified successfully")
            return True
        else:
            logger.error("Models still missing after download")
            return False
            
    except subprocess.CalledProcessError as e:
        logger.error(f"Model download failed: {e}")
        logger.error(f"Setup stderr: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during model download: {e}")
        return False


def download_models_background():
    """Background thread function to download models"""
    global download_status
    
    with download_lock:
        download_status["status"] = "downloading"
        download_status["progress"] = 0.0
        download_status["message"] = "Starting model download and verification..."
        download_status["error"] = None
    
    logger.info("Background model download/fix started")
    
    try:
        # First ensure InfiniteTalk repo is cloned
        with download_lock:
            download_status["progress"] = 5.0
            download_status["message"] = "Cloning InfiniteTalk repository..."
        
        if not clone_infinitetalk_repo():
            with download_lock:
                download_status["status"] = "failed"
                download_status["error"] = "Failed to clone InfiniteTalk repository"
            logger.error("Failed to clone InfiniteTalk repository")
            return
        
        with download_lock:
            download_status["progress"] = 20.0
            download_status["message"] = "Checking and fixing broken models..."
        
        # Use the new fix function instead of setup.py
        logger.info("Checking and fixing broken or missing models...")
        if fix_broken_models():
            with download_lock:
                download_status["status"] = "completed"
                download_status["progress"] = 100.0
                download_status["message"] = "All models downloaded and verified successfully"
            logger.info("All models verified successfully")
        else:
            # Fallback to setup.py if fix_broken_models fails
            with download_lock:
                download_status["progress"] = 50.0
                download_status["message"] = "Fallback: Running setup.py to download models..."
            
            setup_script = Path(__file__).parent / "setup.py"
            if setup_script.exists():
                logger.info("Fallback: Running setup.py to download models...")
                result = subprocess.run(
                    [sys.executable, str(setup_script)],
                    capture_output=False,
                    text=True,
                    check=True
                )
                
                with download_lock:
                    download_status["progress"] = 90.0
                    download_status["message"] = "Verifying downloaded models..."
                
                logger.info("Setup.py completed successfully")
                
                # Verify models are now present
                if check_models_exist():
                    with download_lock:
                        download_status["status"] = "completed"
                        download_status["progress"] = 100.0
                        download_status["message"] = "All models downloaded and verified successfully"
                    logger.info("All models verified successfully")
                else:
                    with download_lock:
                        download_status["status"] = "failed"
                        download_status["error"] = "Models still missing after download"
                    logger.error("Models still missing after download")
            else:
                with download_lock:
                    download_status["status"] = "failed"
                    download_status["error"] = "Both fix_broken_models and setup.py failed"
                logger.error("Both fix_broken_models and setup.py failed")
            
    except subprocess.CalledProcessError as e:
        with download_lock:
            download_status["status"] = "failed"
            download_status["error"] = f"Model download failed: {e}"
        logger.error(f"Model download failed: {e}")
        logger.error("Check the console output above for detailed error information")
    except Exception as e:
        with download_lock:
            download_status["status"] = "failed"
            download_status["error"] = f"Unexpected error during model download: {e}"
        logger.error(f"Unexpected error during model download: {e}")


def start_background_download():
    """Start background model download in a separate thread"""
    download_thread = threading.Thread(target=download_models_background, daemon=True)
    download_thread.start()
    logger.info("Background model download thread started")


# Global download status tracking
download_status = {
    "status": "not_started",  # not_started, downloading, completed, failed
    "progress": 0.0,
    "message": "",
    "error": None
}
download_lock = threading.Lock()

# Global model instance
generator = InfiniteTalkGenerator()


@app.on_event("startup")
async def startup_event():
    """Initialize the model and load jobs on startup"""
    global download_status
    
    # Install flash-attn with --no-build-isolation
    try:
        logger.info("Installing flash-attn with --no-build-isolation...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "flash-attn", "--no-build-isolation"],
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )
        if result.returncode == 0:
            logger.info("flash-attn installed successfully")
        else:
            logger.warning(f"flash-attn installation failed: {result.stderr}")
    except subprocess.TimeoutExpired:
        logger.warning("flash-attn installation timed out")
    except Exception as e:
        logger.warning(f"Error installing flash-attn: {e}")
    
    # Load existing jobs from disk
    JobManager.load_jobs_from_disk()
    logger.info(f"Loaded {len(jobs)} existing jobs")
    
    # First ensure InfiniteTalk repository is cloned
    if not clone_infinitetalk_repo():
        logger.warning("Failed to clone InfiniteTalk repository on startup")
    
    # Download LightX2V LoRA if not present
    if not download_lightx2v_lora():
        logger.warning("Failed to download LightX2V LoRA on startup")
    
    # Update model paths to use cache if available
    update_model_paths_for_cache()
    
    # Update guide scales based on LoRA availability at startup
    update_guide_scales()
    lora_status = "enabled" if is_lora_enabled() else "disabled"
    logger.info(f"LoRA is {lora_status}. Guide scales: text={MODEL_CONFIG['sample_text_guide_scale']}, audio={MODEL_CONFIG['sample_audio_guide_scale']}")
    
    # Check if models exist, start background download/fix if not
    if not check_models_exist():
        logger.info("Models missing or broken, attempting immediate fix...")
        
        # Try to fix models immediately on startup (quick attempt)
        try:
            if fix_broken_models():
                logger.info("Models fixed successfully on startup")
                with download_lock:
                    download_status["status"] = "completed"
                    download_status["progress"] = 100.0
                    download_status["message"] = "All models verified and fixed on startup"
            else:
                raise Exception("Immediate fix failed")
        except Exception as e:
            logger.warning(f"Immediate model fix failed: {e}")
            logger.info("Starting background download/fix...")
            with download_lock:
                download_status["status"] = "not_started"
            start_background_download()
            logger.info("Server will be available immediately. Models are downloading/fixing in background.")
            logger.info("Check /model-status for download progress.")
    else:
        logger.info("All required models found and verified")
        with download_lock:
            download_status["status"] = "completed"
            download_status["progress"] = 100.0
            download_status["message"] = "All models already present"
        
        # Try to load models immediately if they exist
        try:
            generator.load_model()
            logger.info("InfiniteTalk model loaded successfully")
        except Exception as e:
            logger.warning(f"Model loading failed: {e}")
            logger.warning("Model will be loaded on first request")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    with download_lock:
        download_info = download_status.copy()
    
    return {
        "status": "healthy",
        "model_loaded": generator.model_loaded,
        "device": generator.device if hasattr(generator, 'device') else "unknown",
        "models_available": check_models_exist(),
        "lora_enabled": is_lora_enabled(),
        "guide_scales": {
            "sample_text_guide_scale": MODEL_CONFIG["sample_text_guide_scale"],
            "sample_audio_guide_scale": MODEL_CONFIG["sample_audio_guide_scale"]
        },
        "download_status": download_info
    }


@app.get("/model-status")
async def model_download_status():
    """
    Get the current status of model download
    
    Returns information about whether models are available and download progress
    """
    with download_lock:
        status_info = download_status.copy()
    
    status_info["models_available"] = check_models_exist()
    status_info["model_loaded"] = generator.model_loaded
    
    return JSONResponse(content=status_info)


@app.post("/model-persistence")
async def set_model_persistence(keep_loaded: bool = True):
    """
    Configure whether models should stay loaded in memory between generations
    
    Args:
        keep_loaded: If True, models stay in VRAM for faster subsequent generations.
                    If False, models are offloaded to CPU to save VRAM.
                    
    Returns:
        Current model persistence configuration
    """
    MODEL_CONFIG["keep_models_loaded"] = keep_loaded
    return JSONResponse(content={
        "keep_models_loaded": MODEL_CONFIG["keep_models_loaded"],
        "message": f"Model persistence {'enabled' if keep_loaded else 'disabled'}. "
                  f"Models will {'stay loaded' if keep_loaded else 'be offloaded'} after generation."
    })


@app.get("/model-persistence")
async def get_model_persistence():
    """
    Get current model persistence configuration
    
    Returns:
        Current model persistence setting
    """
    return JSONResponse(content={
        "keep_models_loaded": MODEL_CONFIG["keep_models_loaded"],
        "description": "If True, models stay in VRAM for faster generation. If False, models are offloaded to save VRAM."
    })


def validate_image(file: UploadFile) -> bool:
    """Validate uploaded image file"""
    allowed_types = ["image/jpeg", "image/jpg", "image/png", "image/webp"]
    return file.content_type in allowed_types


def validate_audio(file: UploadFile) -> bool:
    """Validate uploaded audio file"""
    allowed_types = ["audio/wav", "audio/mpeg", "audio/mp3", "audio/flac"]
    return file.content_type in allowed_types or file.filename.endswith(('.wav', '.mp3', '.flac', '.m4a'))


@app.post("/generate-video")
async def generate_video(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(..., description="Input image file (JPG, PNG, WebP)"),
    audio: UploadFile = File(..., description="Input audio file (WAV, MP3, FLAC)"),
    resolution: str = Form("480p", description="Output resolution: 480p or 720p"),
    sample_steps: int = Form(4, description="Number of sampling steps (default: 4 for lightx2v LoRA)"),
    audio_cfg_scale: float = Form(4.0, description="Audio CFG scale (3-5 recommended)"),
    max_duration: int = Form(60, description="Maximum video duration in seconds")
):
    """
    Submit a job to generate a lip-sync video from an image and audio file
    
    Returns a job ID immediately. Use /job/{job_id} to check status and /download/{job_id} to get the video.
    
    - **image**: Input image showing the person's face
    - **audio**: Audio file for lip-sync generation
    - **resolution**: Output video resolution (480p or 720p)
    - **sample_steps**: Number of denoising steps (higher = better quality, slower)
    - **audio_cfg_scale**: Audio guidance scale (3-5 for best lip sync)
    - **max_duration**: Maximum video length in seconds
    
    Returns: Job information with job_id for tracking progress
    """
    
    # Validate file types
    if not validate_image(image):
        raise HTTPException(status_code=400, detail="Invalid image format. Supported: JPG, PNG, WebP")
    
    if not validate_audio(audio):
        raise HTTPException(status_code=400, detail="Invalid audio format. Supported: WAV, MP3, FLAC")
    
    # Validate parameters
    if resolution not in ["480p", "720p"]:
        raise HTTPException(status_code=400, detail="Resolution must be either 480p or 720p")
    
    if not 1 <= sample_steps <= 100:
        raise HTTPException(status_code=400, detail="Sample steps must be between 1 and 100")
    
    if not 1.0 <= audio_cfg_scale <= 10.0:
        raise HTTPException(status_code=400, detail="Audio CFG scale must be between 1.0 and 10.0")
    
    if not 1 <= max_duration <= 300:  # Max 5 minutes
        raise HTTPException(status_code=400, detail="Max duration must be between 1 and 300 seconds")
    
    # Generate unique ID for this request
    request_id = uuid.uuid4().hex
    
    try:
        # Save uploaded files
        image_path = UPLOAD_DIR / f"image_{request_id}.{image.filename.split('.')[-1]}"
        audio_path = UPLOAD_DIR / f"audio_{request_id}.{audio.filename.split('.')[-1]}"
        output_path = OUTPUT_DIR / f"output_{request_id}.mp4"
        
        # Write uploaded files to disk
        with open(image_path, "wb") as f:
            shutil.copyfileobj(image.file, f)
        
        with open(audio_path, "wb") as f:
            shutil.copyfileobj(audio.file, f)
        
        # Validate image dimensions and format
        try:
            img = Image.open(image_path)
            img_width, img_height = img.size
            
            # Ensure reasonable image dimensions
            if img_width < 256 or img_height < 256:
                raise HTTPException(status_code=400, detail="Image must be at least 256x256 pixels")
            
            if img_width > 2048 or img_height > 2048:
                raise HTTPException(status_code=400, detail="Image must be no larger than 2048x2048 pixels")
                
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")
        
        # Validate audio duration
        try:
            audio_info = torchaudio.info(str(audio_path))
            duration = audio_info.num_frames / audio_info.sample_rate
            
            if duration > max_duration:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Audio duration ({duration:.1f}s) exceeds maximum ({max_duration}s)"
                )
                
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid audio file: {str(e)}")
        
        # Update model config based on parameters
        MODEL_CONFIG["size"] = f"infinitetalk-{resolution.replace('p', '')}"
        MODEL_CONFIG["sample_steps"] = sample_steps
        # Note: audio_cfg_scale parameter is provided for backward compatibility but 
        # guide scales are now automatically set based on LoRA usage
        MODEL_CONFIG["max_frame_num"] = int(max_duration * 25)  # 25 fps
        
        # Create job using JobManager
        request_params = JobRequest(
            resolution=resolution,
            sample_steps=sample_steps,
            audio_cfg_scale=audio_cfg_scale,
            max_duration=max_duration
        )
        
        job_id = JobManager.create_job(
            request_params=request_params,
            image_path=str(image_path),
            audio_path=str(audio_path),
            output_path=str(output_path)
        )
        
        # Also create legacy job info for backward compatibility
        job_info = JobInfo(
            job_id=job_id,
            status=JobStatus.PENDING,
            image_path=str(image_path),
            audio_path=str(audio_path),
            output_path=str(output_path),
            request_params=request_params,
            created_at=datetime.now(timezone.utc)
        )
        
        # Store job info
        active_jobs[job_id] = job_info
        save_jobs_to_storage()
        
        logger.info(f"Created video generation job {job_id}")
        
        # Start background processing immediately
        background_tasks.add_task(process_video_generation, job_id)
        
        # Return job ID immediately
        return JSONResponse(
            content={
                "job_id": job_id,
                "status": "queued",
                "message": "Video generation job created and processing started. Use /job/{job_id} to check progress and /download/{job_id} to download when complete.",
                "created_at": job_info.created_at.isoformat()
            },
            headers={"X-Request-ID": job_id}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video generation failed: {str(e)}")
    
    finally:
        # Note: Files are now cleaned up in the background task
        pass


@app.get("/job/{job_id}")
async def get_job_status(job_id: str):
    """
    Get the status of a video generation job
    
    - **job_id**: The job ID returned from /generate-video
    """
    # Try new job manager first
    job = JobManager.get_job(job_id)
    if job:
        response_data = {
            "job_id": job["id"],
            "status": job["status"],
            "type": job["type"],
            "created_at": job["created_at"],
            "completed_at": job.get("completed_at"),
            "error": job.get("error"),
            "progress": job.get("progress", 0.0)
        }
        
        if job["status"] == "completed":
            response_data["message"] = "Video generation completed successfully"
            response_data["download_url"] = f"/download/{job_id}"
        elif job["status"] == "failed":
            response_data["message"] = "Video generation failed"
            response_data["error_message"] = job.get("error")
        elif job["status"] == "processing":
            response_data["message"] = "Video generation in progress"
        else:
            response_data["message"] = "Job is pending"
        
        return JSONResponse(content=response_data)
    
    # Fallback to legacy job storage
    job_info = active_jobs.get(job_id)
    if not job_info:
        raise HTTPException(status_code=404, detail="Job not found")
    
    response_data = {
        "job_id": job_info.job_id,
        "status": job_info.status,
        "created_at": job_info.created_at.isoformat(),
        "progress": job_info.progress
    }
    
    if job_info.status == JobStatus.COMPLETED:
        response_data["message"] = "Video generation completed successfully"
        response_data["completed_at"] = job_info.completed_at.isoformat() if job_info.completed_at else None
        response_data["download_url"] = f"/download/{job_id}"
    elif job_info.status == JobStatus.FAILED:
        response_data["message"] = "Video generation failed"
        response_data["error_message"] = job_info.error_message
        response_data["completed_at"] = job_info.completed_at.isoformat() if job_info.completed_at else None
    elif job_info.status == JobStatus.PROCESSING:
        response_data["message"] = "Video generation in progress"
    else:
        response_data["message"] = "Job is pending"
    
    return JSONResponse(content=response_data)


@app.get("/download/{job_id}")
async def download_video(job_id: str):
    """
    Download the generated video for a completed job
    
    - **job_id**: The job ID returned from /generate-video
    """
    # Try new job manager first
    job = JobManager.get_job(job_id)
    if job:
        if job["status"] != "completed":
            raise HTTPException(
                status_code=400, 
                detail=f"Video not ready. Current status: {job['status']}"
            )
        
        if not os.path.exists(job["output_path"]):
            raise HTTPException(status_code=404, detail="Video file not found")
        
        return FileResponse(
            job["output_path"],
            media_type="video/mp4",
            filename=f"lipsync_video_{job_id}.mp4",
            headers={"X-Job-ID": job_id}
        )
    
    # Fallback to legacy job storage
    job_info = active_jobs.get(job_id)
    if not job_info:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job_info.status != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=400, 
            detail=f"Video not ready. Current status: {job_info.status}"
        )
    
    if not os.path.exists(job_info.output_path):
        raise HTTPException(status_code=404, detail="Video file not found")
    
    return FileResponse(
        job_info.output_path,
        media_type="video/mp4",
        filename=f"lipsync_video_{job_id}.mp4",
        headers={"X-Job-ID": job_id}
    )


@app.get("/jobs")
async def list_jobs(status: Optional[str] = None, limit: int = 50):
    """
    List all jobs, optionally filtered by status
    
    - **status**: Filter by job status (pending, processing, completed, failed)
    - **limit**: Maximum number of jobs to return (default: 50)
    """
    all_jobs = []
    
    # Add jobs from new JobManager
    for job_id, job_data in jobs.items():
        all_jobs.append({
            "job_id": job_id,
            "type": job_data["type"],
            "status": job_data["status"],
            "created_at": job_data["created_at"],
            "completed_at": job_data.get("completed_at"),
            "progress": job_data.get("progress", 0.0),
            "error": job_data.get("error")
        })
    
    # Add legacy jobs
    for job_info in active_jobs.values():
        all_jobs.append({
            "job_id": job_info.job_id,
            "type": "video",
            "status": job_info.status.value,
            "created_at": job_info.created_at.isoformat(),
            "completed_at": job_info.completed_at.isoformat() if job_info.completed_at else None,
            "progress": job_info.progress,
            "error": job_info.error_message
        })
    
    # Filter by status if provided
    if status:
        all_jobs = [job for job in all_jobs if job["status"] == status]
    
    # Sort by creation time (newest first)
    all_jobs.sort(key=lambda x: x["created_at"], reverse=True)
    
    # Limit results
    all_jobs = all_jobs[:limit]
    
    # Convert to response format
    response_jobs = []
    for job in all_jobs:
        job_data = {
            "job_id": job["job_id"],
            "type": job["type"],
            "status": job["status"],
            "created_at": job["created_at"],
            "progress": job["progress"]
        }
        
        if job["completed_at"]:
            job_data["completed_at"] = job["completed_at"]
        
        if job["status"] == "completed":
            job_data["download_url"] = f"/download/{job['job_id']}"
        elif job["status"] == "failed":
            job_data["error_message"] = job["error"]
            
        response_jobs.append(job_data)
    
    return JSONResponse(content={
        "jobs": response_jobs,
        "total": len(response_jobs),
        "filtered_by_status": status
    })


@app.delete("/job/{job_id}")
async def delete_job(job_id: str):
    """
    Delete a job and its associated files
    
    - **job_id**: The job ID to delete
    """
    # Try new job manager first
    job = JobManager.get_job(job_id)
    if job:
        # Don't allow deletion of processing jobs
        if job["status"] == "processing":
            raise HTTPException(status_code=400, detail="Cannot delete job that is currently processing")
        
        # Clean up files
        files_to_clean = [job["image_path"], job["audio_path"], job["output_path"]]
        for file_path in files_to_clean:
            try:
                if os.path.exists(file_path):
                    os.unlink(file_path)
            except:
                pass  # Don't fail if cleanup fails
        
        # Remove from jobs and delete job file
        del jobs[job_id]
        job_file = JOBS_DIR / f"{job_id}.json"
        if job_file.exists():
            job_file.unlink()
        
        return JSONResponse(content={
            "message": f"Job {job_id} deleted successfully"
        })
    
    # Fallback to legacy job storage
    job_info = active_jobs.get(job_id)
    if not job_info:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Don't allow deletion of processing jobs
    if job_info.status == JobStatus.PROCESSING:
        raise HTTPException(status_code=400, detail="Cannot delete job that is currently processing")
    
    # Clean up files
    files_to_clean = [job_info.image_path, job_info.audio_path, job_info.output_path]
    for file_path in files_to_clean:
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
        except:
            pass  # Don't fail if cleanup fails
    
    # Remove from active jobs
    del active_jobs[job_id]
    save_jobs_to_storage()
    
    return JSONResponse(content={
        "message": f"Job {job_id} deleted successfully"
    })


@app.get("/")
async def root():
    """Root endpoint with API information"""
    with download_lock:
        download_info = download_status.copy()
    
    return {
        "message": "InfiniteTalk Lip-Sync Video Generation API",
        "version": "1.0.0",
        "models_available": check_models_exist(),
        "download_status": download_info,
        "endpoints": {
            "generate_video": "/generate-video",
            "job_status": "/job/{job_id}",
            "download_video": "/download/{job_id}",
            "list_jobs": "/jobs",
            "delete_job": "/job/{job_id} (DELETE)",
            "model_status": "/model-status",
            "health": "/health",
            "docs": "/docs"
        },
        "supported_formats": {
            "image": ["JPG", "PNG", "WebP"],
            "audio": ["WAV", "MP3", "FLAC"]
        }
    }


if __name__ == "__main__":
    # Configuration for RunPod deployment
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        workers=1,  # Single worker for GPU usage
        access_log=True
    )
