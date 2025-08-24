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
from datetime import datetime, timezone
from enum import Enum

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

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

# In-memory job storage (in production, use Redis or database)
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


async def process_video_generation(job_id: str):
    """Background task to process video generation"""
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
        required_paths = [
            MODEL_CONFIG["ckpt_dir"],
            MODEL_CONFIG["wav2vec_dir"],
            MODEL_CONFIG["infinitetalk_dir"]
        ]
        
        for path in required_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Model weights not found at: {path}")
        
        print(f"Loading InfiniteTalk model on {self.device}...")
        print("InfiniteTalk models will be loaded on first inference call")
        
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


# Global model instance
generator = InfiniteTalkGenerator()


@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup"""
    try:
        generator.load_model()
        print("InfiniteTalk model loaded successfully")
    except Exception as e:
        print(f"Warning: Model loading failed: {e}")
        print("Model will be loaded on first request")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": generator.model_loaded,
        "device": generator.device if hasattr(generator, 'device') else "unknown"
    }


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
        
        # Create job info
        job_info = JobInfo(
            job_id=request_id,
            status=JobStatus.PENDING,
            image_path=str(image_path),
            audio_path=str(audio_path),
            output_path=str(output_path),
            request_params=JobRequest(
                resolution=resolution,
                sample_steps=sample_steps,
                audio_cfg_scale=audio_cfg_scale,
                max_duration=max_duration
            ),
            created_at=datetime.now(timezone.utc)
        )
        
        # Store job info
        active_jobs[request_id] = job_info
        save_jobs_to_storage()
        
        # Start background processing
        background_tasks.add_task(process_video_generation, request_id)
        
        # Return job ID immediately
        return JSONResponse(
            content={
                "job_id": request_id,
                "status": JobStatus.PENDING,
                "message": "Video generation job started",
                "created_at": job_info.created_at.isoformat()
            },
            headers={"X-Request-ID": request_id}
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
    jobs = list(active_jobs.values())
    
    # Filter by status if provided
    if status:
        try:
            status_enum = JobStatus(status)
            jobs = [job for job in jobs if job.status == status_enum]
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid status: {status}")
    
    # Sort by creation time (newest first)
    jobs.sort(key=lambda x: x.created_at, reverse=True)
    
    # Limit results
    jobs = jobs[:limit]
    
    # Convert to response format
    response_jobs = []
    for job in jobs:
        job_data = {
            "job_id": job.job_id,
            "status": job.status,
            "created_at": job.created_at.isoformat(),
            "progress": job.progress
        }
        
        if job.completed_at:
            job_data["completed_at"] = job.completed_at.isoformat()
        
        if job.status == JobStatus.COMPLETED:
            job_data["download_url"] = f"/download/{job.job_id}"
        elif job.status == JobStatus.FAILED:
            job_data["error_message"] = job.error_message
            
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
    return {
        "message": "InfiniteTalk Lip-Sync Video Generation API",
        "version": "1.0.0",
        "endpoints": {
            "generate_video": "/generate-video",
            "job_status": "/job/{job_id}",
            "download_video": "/download/{job_id}",
            "list_jobs": "/jobs",
            "delete_job": "/job/{job_id} (DELETE)",
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
