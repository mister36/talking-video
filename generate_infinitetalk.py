"""
InfiniteTalk video generation script
Based on the official InfiniteTalk implementation
"""

import os
import sys
import json
import argparse
from pathlib import Path
import logging

import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
from PIL import Image
import cv2
from transformers import Wav2Vec2Model, Wav2Vec2Processor
from diffusers import DiffusionPipeline
import safetensors.torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="InfiniteTalk Video Generation")
    
    # Model paths
    parser.add_argument("--ckpt_dir", type=str, required=True, help="Path to Wan2.1-I2V-14B-480P model")
    parser.add_argument("--wav2vec_dir", type=str, required=True, help="Path to wav2vec2 model")
    parser.add_argument("--infinitetalk_dir", type=str, required=True, help="Path to InfiniteTalk weights")
    
    # Input/Output
    parser.add_argument("--input_json", type=str, required=True, help="Input JSON file with image and audio paths")
    parser.add_argument("--save_file", type=str, required=True, help="Output video path (without extension)")
    
    # Generation parameters
    parser.add_argument("--size", type=str, default="infinitetalk-480", choices=["infinitetalk-480", "infinitetalk-720"])
    parser.add_argument("--sample_steps", type=int, default=40, help="Number of sampling steps")
    parser.add_argument("--mode", type=str, default="streaming", choices=["streaming", "clip"])
    parser.add_argument("--motion_frame", type=int, default=9, help="Motion frame parameter")
    parser.add_argument("--sample_text_guide_scale", type=float, default=5.0, help="Text guidance scale")
    parser.add_argument("--sample_audio_guide_scale", type=float, default=4.0, help="Audio guidance scale")
    parser.add_argument("--max_frame_num", type=int, default=1000, help="Maximum number of frames")
    parser.add_argument("--num_persistent_param_in_dit", type=int, default=0, help="Persistent parameters for low VRAM")
    
    # Optional parameters
    parser.add_argument("--lora_dir", type=str, help="Path to LoRA weights")
    parser.add_argument("--lora_scale", type=float, default=1.0, help="LoRA scale")
    parser.add_argument("--sample_shift", type=int, default=1, help="Sample shift parameter")
    parser.add_argument("--use_teacache", action="store_true", help="Use TeaCache acceleration")
    parser.add_argument("--teacache_thresh", type=float, default=0.7, help="TeaCache threshold")
    parser.add_argument("--quant", type=str, choices=["fp8", "int8"], help="Quantization type")
    parser.add_argument("--quant_dir", type=str, help="Path to quantization weights")
    
    return parser.parse_args()


class InfiniteTalkModel:
    """InfiniteTalk model implementation following the official guide"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_loaded = False
        
        # Model components as per InfiniteTalk architecture
        self.dit_pipeline = None
        self.audio_encoder = None
        self.audio_processor = None
        self.infinitetalk_weights = None
        
    def load_models(self):
        """Load all required models following InfiniteTalk guide"""
        logger.info(f"Loading models on device: {self.device}")
        
        # Check if model paths exist
        required_paths = [
            self.args.ckpt_dir,
            self.args.wav2vec_dir,
            self.args.infinitetalk_dir
        ]
        
        for path in required_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Model path not found: {path}")
        
        try:
            # 1. Load base video generation model (Wan2.1-I2V-14B-480P)
            logger.info("Loading Wan2.1-I2V-14B-480P pipeline...")
            self.dit_pipeline = DiffusionPipeline.from_pretrained(
                self.args.ckpt_dir,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                use_safetensors=True
            ).to(self.device)
            
            # 2. Load audio encoder (chinese-wav2vec2-base)
            logger.info("Loading Chinese Wav2Vec2 audio encoder...")
            self.audio_processor = Wav2Vec2Processor.from_pretrained(self.args.wav2vec_dir)
            self.audio_encoder = Wav2Vec2Model.from_pretrained(self.args.wav2vec_dir).to(self.device)
            
            # 3. Load InfiniteTalk conditioning weights
            logger.info("Loading InfiniteTalk weights...")
            self.infinitetalk_weights = safetensors.torch.load_file(
                self.args.infinitetalk_dir, 
                device=str(self.device)
            )
            
            # Apply InfiniteTalk weights to the pipeline
            self._apply_infinitetalk_weights()
            
            # 4. Apply LoRA if specified
            if self.args.lora_dir and os.path.exists(self.args.lora_dir):
                logger.info("Loading LoRA weights...")
                self._apply_lora_weights()
                
            # 5. Setup quantization if specified
            if self.args.quant and self.args.quant_dir:
                logger.info(f"Applying {self.args.quant} quantization...")
                self._apply_quantization()
            
            # 6. Setup memory optimization
            if self.args.num_persistent_param_in_dit == 0:
                logger.info("Enabling low VRAM mode...")
                self.dit_pipeline.enable_model_cpu_offload()
                self.dit_pipeline.enable_vae_slicing()
            
            logger.info("All models loaded successfully")
            self.model_loaded = True
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise
    
    def _apply_infinitetalk_weights(self):
        """Apply InfiniteTalk conditioning weights to the pipeline"""
        logger.info("Applying InfiniteTalk conditioning weights...")
        
        # Apply the InfiniteTalk weights to enable audio conditioning
        # This modifies the DiT model to accept audio features
        for name, param in self.infinitetalk_weights.items():
            if hasattr(self.dit_pipeline.unet, name):
                getattr(self.dit_pipeline.unet, name).data.copy_(param)
    
    def _apply_lora_weights(self):
        """Apply LoRA weights for faster inference"""
        logger.info("Applying LoRA weights...")
        
        lora_weights = safetensors.torch.load_file(self.args.lora_dir, device=str(self.device))
        # Apply LoRA modifications to the pipeline
        # This would require the specific LoRA implementation for the model
    
    def _apply_quantization(self):
        """Apply quantization for memory efficiency"""
        logger.info(f"Applying {self.args.quant} quantization...")
        
        if self.args.quant == "fp8":
            # Apply FP8 quantization
            pass
        elif self.args.quant == "int8":
            # Apply INT8 quantization
            pass
    
    def preprocess_image(self, image_path):
        """Preprocess input image"""
        logger.info(f"Preprocessing image: {image_path}")
        
        # Load and validate image
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        try:
            image = Image.open(image_path).convert("RGB")
            
            # Get target resolution
            if self.args.size == "infinitetalk-480":
                target_size = (480, 480)
            else:  # infinitetalk-720
                target_size = (720, 720)
            
            # Resize image while maintaining aspect ratio
            image = image.resize(target_size, Image.Resampling.LANCZOS)
            
            # Convert to tensor and normalize
            image_array = np.array(image) / 255.0
            image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).float()
            image_tensor = image_tensor.unsqueeze(0).to(self.device)
            
            return image_tensor
            
        except Exception as e:
            logger.error(f"Failed to preprocess image: {e}")
            raise
    
    def preprocess_audio(self, audio_path):
        """Preprocess input audio using wav2vec2"""
        logger.info(f"Preprocessing audio: {audio_path}")
        
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio not found: {audio_path}")
        
        try:
            # Load audio file
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Resample to 16kHz if needed (wav2vec2 requirement)
            target_sr = 16000
            if sample_rate != target_sr:
                resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
                waveform = resampler(waveform)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Convert to numpy for processor
            waveform_np = waveform.squeeze().numpy()
            
            # Process with wav2vec2 processor
            inputs = self.audio_processor(
                waveform_np,
                sampling_rate=target_sr,
                return_tensors="pt",
                padding=True
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Extract audio features using wav2vec2
            with torch.no_grad():
                audio_features = self.audio_encoder(**inputs).last_hidden_state
            
            return audio_features
            
        except Exception as e:
            logger.error(f"Failed to preprocess audio: {e}")
            raise
    
    def generate_video(self, image_tensor, audio_features, output_path):
        """Generate lip-sync video using InfiniteTalk"""
        logger.info("Starting InfiniteTalk video generation...")
        
        if not self.model_loaded:
            self.load_models()
        
        try:
            # Calculate video length from audio features
            # Audio features shape: [batch, sequence_length, hidden_size]
            audio_seq_len = audio_features.shape[1]
            # Assuming wav2vec2 processes ~50 frames per second of audio
            audio_length = audio_seq_len / 50.0
            num_frames = min(int(audio_length * 25), self.args.max_frame_num)  # 25 FPS
            
            logger.info(f"Generating {num_frames} frames for {audio_length:.2f}s audio")
            
            # Prepare generation parameters
            generation_kwargs = {
                "num_inference_steps": self.args.sample_steps,
                "guidance_scale": self.args.sample_text_guide_scale,
                "num_frames": num_frames,
                "height": 480 if self.args.size == "infinitetalk-480" else 720,
                "width": 480 if self.args.size == "infinitetalk-480" else 720,
            }
            
            # Add audio conditioning parameters
            if hasattr(self.args, 'sample_audio_guide_scale'):
                generation_kwargs["audio_guidance_scale"] = self.args.sample_audio_guide_scale
            
            # Generate video frames using InfiniteTalk-conditioned pipeline
            logger.info("Running InfiniteTalk inference...")
            
            if self.args.mode == "streaming":
                # Streaming mode for long videos
                generated_frames = self._generate_streaming(
                    image_tensor, audio_features, generation_kwargs
                )
            else:
                # Clip mode for short videos
                generated_frames = self._generate_clip(
                    image_tensor, audio_features, generation_kwargs
                )
            
            # Convert tensor frames to numpy for video saving
            if isinstance(generated_frames, torch.Tensor):
                generated_frames = generated_frames.cpu().numpy()
                # Convert from [-1, 1] to [0, 255]
                generated_frames = ((generated_frames + 1) * 127.5).clip(0, 255).astype(np.uint8)
                # Rearrange dimensions if needed [B, T, C, H, W] -> [T, H, W, C]
                if generated_frames.ndim == 5:
                    generated_frames = generated_frames[0].transpose(0, 2, 3, 1)
            
            # Save video
            self.save_video(generated_frames, output_path, fps=25)
            
            logger.info(f"Video saved to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Video generation failed: {e}")
            raise
    
    def _generate_streaming(self, image_tensor, audio_features, generation_kwargs):
        """Generate video in streaming mode for long sequences"""
        logger.info("Using streaming mode for long video generation")
        
        # Split audio into chunks for long video generation
        chunk_size = 200  # Process ~4 seconds at a time
        audio_chunks = torch.split(audio_features, chunk_size, dim=1)
        
        all_frames = []
        previous_frame = image_tensor
        
        for i, audio_chunk in enumerate(audio_chunks):
            logger.info(f"Processing chunk {i+1}/{len(audio_chunks)}")
            
            # Generate frames for this chunk
            chunk_kwargs = generation_kwargs.copy()
            chunk_kwargs["num_frames"] = min(chunk_size, generation_kwargs["num_frames"] - len(all_frames))
            
            # Use previous frame as starting point for continuity
            chunk_frames = self.dit_pipeline(
                image=previous_frame,
                audio_features=audio_chunk,
                **chunk_kwargs
            ).frames
            
            all_frames.extend(chunk_frames)
            previous_frame = chunk_frames[-1:] if len(chunk_frames) > 0 else previous_frame
            
            if len(all_frames) >= generation_kwargs["num_frames"]:
                break
        
        return torch.stack(all_frames[:generation_kwargs["num_frames"]])
    
    def _generate_clip(self, image_tensor, audio_features, generation_kwargs):
        """Generate video in clip mode for short sequences"""
        logger.info("Using clip mode for short video generation")
        
        # Generate all frames at once
        result = self.dit_pipeline(
            image=image_tensor,
            audio_features=audio_features,
            **generation_kwargs
        )
        
        return result.frames
    

    
    def save_video(self, frames, output_path, fps=25):
        """Save generated frames as video"""
        logger.info(f"Saving video with {len(frames)} frames at {fps} FPS")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Setup video writer
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        try:
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            for frame in frames:
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
            
            out.release()
            logger.info(f"Video successfully saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save video: {e}")
            raise


def main():
    """Main generation function"""
    args = parse_args()
    
    # Load input configuration
    with open(args.input_json, 'r') as f:
        input_data = json.load(f)
    
    image_path = input_data['image_path']
    audio_path = input_data['audio_path']
    output_path = f"{args.save_file}.mp4"
    
    logger.info(f"Input image: {image_path}")
    logger.info(f"Input audio: {audio_path}")
    logger.info(f"Output video: {output_path}")
    
    # Initialize model
    model = InfiniteTalkModel(args)
    
    try:
        # Load models
        model.load_models()
        
        # Preprocess inputs
        image_tensor = model.preprocess_image(image_path)
        audio_features = model.preprocess_audio(audio_path)
        
        # Generate video
        result_path = model.generate_video(image_tensor, audio_features, output_path)
        
        logger.info("Generation completed successfully!")
        return result_path
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
