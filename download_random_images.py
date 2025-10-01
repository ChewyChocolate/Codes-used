import requests
import os
import random
import string
import asyncio
import aiohttp
from PIL import Image
import logging
import aiohttp.client_exceptions
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def download_image(session, url, image_path, resize_to, semaphore):
    """
    Asynchronously download and resize an image with semaphore to limit concurrency.
    """
    async with semaphore:  # Limit concurrent requests
        try:
            async with session.get(url, timeout=10) as response:
                if response.status != 200:
                    raise aiohttp.ClientResponseError(
                        response.request_info,
                        response.history,
                        status=response.status,
                        message=f"HTTP {response.status}"
                    )
                
                # Save the image temporarily
                with open(image_path, 'wb') as f:
                    f.write(await response.read())
                
                # Resize the image
                img = Image.open(image_path)
                img = img.resize(resize_to, Image.LANCZOS)
                img.save(image_path, quality=95)
                logging.info(f"Successfully downloaded and resized {image_path} to {resize_to}")
                return True
        except (aiohttp.ClientResponseError, aiohttp.ClientConnectionError, asyncio.TimeoutError) as e:
            logging.error(f"Network error for {url}: {type(e).__name__}: {str(e)}")
            return False
        except Exception as e:
            logging.error(f"Unexpected error for {url}: {type(e).__name__}: {str(e)}")
            return False

async def download_random_images(
    num_images,
    output_dir,
    resize_to=(448, 448),
    resolution_ranges=None,
    max_ratio=1.75,
    max_retries=3,
    batch_size=100,
    max_concurrent=50
):
    """
    Asynchronously downloads random images from picsum.photos with varied resolutions.

    Args:
        num_images (int): Number of images to download.
        output_dir (str): Directory to save images.
        resize_to (tuple): Target size (width, height) for resizing.
        resolution_ranges (list): List of tuples [(min_dim, max_dim), ...] for resolutions.
        max_ratio (float): Maximum aspect ratio difference.
        max_retries (int): Maximum retries for failed downloads.
        batch_size (int): Number of images per batch to manage rate limiting.
        max_concurrent (int): Maximum concurrent downloads.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Default resolution ranges
    if resolution_ranges is None:
        resolution_ranges = [
            (500, 1000),
            (1000, 2000),
            (2000, 3000),
            (3000, 4000)
        ]

    semaphore = asyncio.Semaphore(max_concurrent)  # Limit concurrent requests
    total_successful = 0

    # Process images in batches
    for batch_start in range(0, num_images, batch_size):
        batch_end = min(batch_start + batch_size, num_images)
        logging.info(f"Processing batch {batch_start + 1} to {batch_end}")
        
        tasks = []
        async with aiohttp.ClientSession() as session:
            for i in range(batch_start, batch_end):
                for attempt in range(max_retries):
                    try:
                        # Select a random resolution range
                        min_dim, max_dim = random.choice(resolution_ranges)
                        
                        # Generate width and height
                        width = random.randint(min_dim, max_dim)
                        min_height = max(min_dim, int(width / max_ratio))
                        max_height = min(max_dim, int(width * max_ratio))
                        height = random.randint(min_height, max_height)

                        # Randomly swap width and height
                        if random.choice([True, False]):
                            width, height = height, width

                        # Ensure dimensions are within Picsum's limits (1 to 5000)
                        width = min(max(width, 1), 5000)
                        height = min(max(height, 1), 5000)

                        url = f"https://picsum.photos/{width}/{height}"
                        image_path = os.path.join(output_dir, f"random_image_{i+1}.jpg")
                        
                        tasks.append(download_image(session, url, image_path, resize_to, semaphore))
                        break  # Exit retry loop on success
                    except Exception as e:
                        logging.warning(f"Attempt {attempt + 1} failed for image {i + 1}: {e}")
                        if attempt == max_retries - 1:
                            logging.error(f"Failed to download image {i + 1} after {max_retries} attempts")
            
            # Run tasks in the current batch
            results = await asyncio.gather(*tasks, return_exceptions=True)
            successful = sum(1 for result in results if result is True)
            total_successful += successful
            logging.info(f"Batch completed: {successful}/{batch_end - batch_start} images successful")
            
        # Add delay between batches to avoid rate limiting
        if batch_end < num_images:
            logging.info("Pausing for 5 seconds to avoid rate limiting")
            await asyncio.sleep(5)

    logging.info(f"Completed: {total_successful}/{num_images} images downloaded successfully")

if __name__ == "__main__":
    # --- Configuration ---
    NUMBER_OF_IMAGES = 500  # Your target
    OUTPUT_DIRECTORY = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
    IMG_SIZE = 448
    RESIZE_DIMENSIONS = (IMG_SIZE, IMG_SIZE)
    
    RESOLUTION_RANGES = [
        (500, 1000),
        (1000, 2000),
        (2000, 3000),
        (3000, 4000)
    ]
    MAX_ASPECT_RATIO_DIFF = 1.75
    MAX_RETRIES = 3
    BATCH_SIZE = 100  # Process 100 images at a time
    MAX_CONCURRENT = 50  # Limit to 50 concurrent downloads

    logging.info(f"Saving images to directory: {OUTPUT_DIRECTORY}")
    
    # Run the async function
    asyncio.run(download_random_images(
        NUMBER_OF_IMAGES,
        OUTPUT_DIRECTORY,
        RESIZE_DIMENSIONS,
        RESOLUTION_RANGES,
        MAX_ASPECT_RATIO_DIFF,
        MAX_RETRIES,
        BATCH_SIZE,
        MAX_CONCURRENT
    ))
