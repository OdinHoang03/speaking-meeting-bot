"""Service for handling image generation using Replicate."""

from typing import Optional
from loguru import logger
from PIL import Image
import requests
from io import BytesIO
import os
import replicate
from pathlib import Path
from dotenv import load_dotenv
from config.image_uploader import UTFSUploader
from config.prompts import IMAGE_NEGATIVE_PROMPT
import asyncio


# Load environment variables
load_dotenv()

class ImageService:
    """Service for handling image generation and processing."""
    
    def __init__(self):
        """Initialize the image service."""
        self.uploader = UTFSUploader(
            api_key=os.getenv("UTFS_KEY"),
            app_id=os.getenv("APP_ID")
        )
        # Set Replicate API token
        # Replicate API tokens shouldn't include the "sk_live_" prefix
        self.replicate_key = os.getenv("REPLICATE_KEY", "")
        if self.replicate_key.startswith("sk_live_"):
            self.replicate_key = self.replicate_key.replace("sk_live_", "")
        os.environ["REPLICATE_API_TOKEN"] = self.replicate_key
        logger.info("Initialized Replicate client and UTFSUploader for image generation")
    
    async def generate_persona_image(
        self,
        name: str,
        prompt: str,
        style: str = "realistic",
        size: tuple[int, int] = (512, 512)
        ) -> str:

        try:
            # Add style to prompt
            full_prompt = f"{style} style, {prompt}"

            logger.info(f"Generating image with prompt: {full_prompt}")

            # Generate image using Replicate's SDXL
            output = await asyncio.to_thread(replicate.run,
                "stability-ai/sdxl:7762fd07cf82c948538e41f63f77d685e02b063e37e496e96eefd46c929f9bdc",
                input={
                    "prompt": full_prompt,
                    "width": size[0],
                    "height": size[1],
                    "refine": "expert_ensemble_refiner",
                    "apply_watermark": False,
                    "num_inference_steps": 25,
                    "negative_prompt": IMAGE_NEGATIVE_PROMPT,
                    "scheduler": "DPMSolverMultistep",
                    "guidance_scale": 7.5,
                },
            )

            if not output or len(output) == 0:
                raise ValueError("No output received from Replicate")

            # Get the image URL from Replicate
            if isinstance(output, list) and len(output) > 0:
                image_url = output[0]
            else:
                raise ValueError(f"Unexpected output format from Replicate: {output}")

            # Download the image
            response = requests.get(image_url)
            if response.status_code != 200:
                raise ValueError(f"Failed to download image. Status code: {response.status_code}")
            
            # Save to temporary file
            temp_path = f"{name}.png"
            with open(temp_path, "wb") as f:
                f.write(response.content)

            # Upload to UTFS
            file_url = await asyncio.to_thread(self.uploader.upload_file, Path(temp_path))

            # Clean up temporary file
            try:
                os.remove(temp_path)
            except FileNotFoundError:
                logger.warning(f"Temporary image file not found for cleanup: {temp_path}")

            if not file_url:
                raise ValueError("Failed to upload image to UTFS")

            return file_url
        
        except Exception as e:       
            logger.error(f"Failed to generate image: {str(e)}")
            raise ValueError(f"Failed to generate image: {str(e)}") from e


# Create global instance
image_service = ImageService() 