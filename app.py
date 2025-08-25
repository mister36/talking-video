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
    sample_steps: int = Field(default=40, description="Number of sampling steps")
    audio_cfg_scale: float = Field(default=4.0, description="Audio CFG scale")
    max_duration: int = Field(default=40, description="Maximum video duration in seconds")

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

# Global variables for model paths and configurations
MODEL_CONFIG = {
    "ckpt_dir": "weights/Wan2.1-I2V-14B-480P",
    "wav2vec_dir": "weights/chinese-wav2vec2-base", 
    "infinitetalk_dir": "weights/InfiniteTalk/single/infinitetalk.safetensors",
    "size": "infinitetalk-480",  # or infinitetalk-720
    "sample_steps": 40,
    "mode": "streaming",
    "motion_frame": 9,
    "sample_text_guide_scale": 5.0,
    "sample_audio_guide_scale": 4.0,
    "max_frame_num": 1000,  # 40 seconds at 25fps
    "num_persistent_param_in_dit": 0,  # For low VRAM usage
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
    """Wrapper class for InfiniteTalk model inference"""
    
    def __init__(self):
        self.model_loaded = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def load_model(self):
        """Load the InfiniteTalk model"""
        if self.model_loaded:
            return
            
        # Check if model weights exist
        if not check_models_exist():
            # Check if models are currently downloading
            with download_lock:
                status = download_status["status"]
            
            if status == "downloading":
                raise FileNotFoundError("Models are currently downloading. Please wait for download to complete. Check /model-status for progress.")
            elif status == "failed":
                with download_lock:
                    error = download_status["error"]
                raise FileNotFoundError(f"Model download failed: {error}. Please run 'python setup.py' manually.")
            else:
                # Start download if not already started
                logger.info("Models not found during load_model, starting background download...")
                start_background_download()
                raise FileNotFoundError("Model download started in background. Please wait for download to complete. Check /model-status for progress.")
        
        # Double-check that models now exist
        required_paths = [
            MODEL_CONFIG["ckpt_dir"],
            MODEL_CONFIG["wav2vec_dir"],
            MODEL_CONFIG["infinitetalk_dir"]
        ]
        
        for path in required_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Model weights not found at: {path}")
        
        logger.info(f"Loading InfiniteTalk model on {self.device}...")
        logger.info("InfiniteTalk models will be loaded on first inference call")
        
        self.model_loaded = True
        
    def generate_video(self, image_path: str, audio_path: str, output_path: str) -> str:
        """Generate lip-sync video from image and audio"""
        if not self.model_loaded:
            self.load_model()
            
        # Create input JSON for InfiniteTalk
        input_data = {
            "image_path": image_path,
            "audio_path": audio_path,
            "output_path": output_path
        }
        
        # Create temporary JSON file
        temp_json = TEMP_DIR / f"input_{uuid.uuid4().hex}.json"
        with open(temp_json, 'w') as f:
            json.dump(input_data, f)
        
        try:
            # Build command for InfiniteTalk generation
            cmd = [
                "python", "generate_infinitetalk.py",
                "--ckpt_dir", MODEL_CONFIG["ckpt_dir"],
                "--wav2vec_dir", MODEL_CONFIG["wav2vec_dir"],
                "--infinitetalk_dir", MODEL_CONFIG["infinitetalk_dir"],
                "--input_json", str(temp_json),
                "--size", MODEL_CONFIG["size"],
                "--sample_steps", str(MODEL_CONFIG["sample_steps"]),
                "--mode", MODEL_CONFIG["mode"],
                "--motion_frame", str(MODEL_CONFIG["motion_frame"]),
                "--num_persistent_param_in_dit", str(MODEL_CONFIG["num_persistent_param_in_dit"]),
                "--save_file", output_path.replace('.mp4', '')
            ]
            
            # Run the generation process
            print(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            if result.returncode != 0:
                raise RuntimeError(f"Generation failed: {result.stderr}")
                
            return output_path
            
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Video generation failed: {e.stderr}")
        finally:
            # Clean up temporary files
            if temp_json.exists():
                temp_json.unlink()


def check_models_exist():
    """Check if all required models exist"""
    required_paths = [
        MODEL_CONFIG["ckpt_dir"],
        MODEL_CONFIG["wav2vec_dir"],
        MODEL_CONFIG["infinitetalk_dir"]
    ]
    
    for path in required_paths:
        if not os.path.exists(path):
            return False
    return True


def download_models_automatically():
    """Automatically download models using setup.py"""
    logger.info("Models not found. Starting automatic download...")
    
    try:
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
        download_status["message"] = "Starting model download..."
        download_status["error"] = None
    
    logger.info("Background model download started")
    
    try:
        # Run setup.py to download models
        setup_script = Path(__file__).parent / "setup.py"
        if not setup_script.exists():
            with download_lock:
                download_status["status"] = "failed"
                download_status["error"] = "setup.py not found!"
            logger.error("setup.py not found!")
            return
        
        with download_lock:
            download_status["progress"] = 10.0
            download_status["message"] = "Running setup.py to download models..."
        
        logger.info("Running setup.py to download models...")
        result = subprocess.run(
            [sys.executable, str(setup_script)],
            capture_output=True,
            text=True,
            check=True
        )
        
        with download_lock:
            download_status["progress"] = 90.0
            download_status["message"] = "Verifying downloaded models..."
        
        logger.info("Model download completed successfully")
        logger.info(f"Setup output: {result.stdout}")
        
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
            
    except subprocess.CalledProcessError as e:
        with download_lock:
            download_status["status"] = "failed"
            download_status["error"] = f"Model download failed: {e.stderr}"
        logger.error(f"Model download failed: {e}")
        logger.error(f"Setup stderr: {e.stderr}")
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
    
    # Load existing jobs from disk
    JobManager.load_jobs_from_disk()
    logger.info(f"Loaded {len(jobs)} existing jobs")
    
    # Check if models exist, start background download if not
    if not check_models_exist():
        logger.info("Models not found, starting background download...")
        with download_lock:
            download_status["status"] = "not_started"
        start_background_download()
        logger.info("Server will be available immediately. Models are downloading in background.")
        logger.info("Check /model-status for download progress.")
    else:
        logger.info("All required models found")
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
    sample_steps: int = Form(40, description="Number of sampling steps (default: 40)"),
    audio_cfg_scale: float = Form(4.0, description="Audio CFG scale (3-5 recommended)"),
    max_duration: int = Form(40, description="Maximum video duration in seconds")
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
        MODEL_CONFIG["sample_audio_guide_scale"] = audio_cfg_scale
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
