import os
import sys
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Import our WetroCloud service
from services.wetrocloud_youtube import WetroCloudYouTubeService

def test_wetrocloud_transcript():
    """
    Test the WetroCloud YouTube transcript extraction functionality
    """
    # Initialize the service
    wetrocloud_service = WetroCloudYouTubeService()
    
    # Test URLs
    test_urls = [
        "https://www.youtube.com/watch?v=qZvfC4TOAD0",  # Example from your request
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",  # A well-known video
    ]
    
    for url in test_urls:
        logger.info(f"\n\nTesting URL: {url}")
        
        # Extract video ID
        video_id = wetrocloud_service.extract_video_id(url)
        logger.info(f"Extracted video ID: {video_id}")
        
        # Get transcript
        result = wetrocloud_service.get_transcript(url)
        
        if result.get('success'):
            # Success! Print first 200 characters of transcript
            transcript_preview = result['text'][:200] + "..." if len(result['text']) > 200 else result['text']
            logger.info(f"Transcript preview: {transcript_preview}")
            logger.info(f"Total tokens: {result.get('tokens', 'N/A')}")
        else:
            # Error
            logger.error(f"Failed to get transcript: {result.get('message')}")

if __name__ == "__main__":
    test_wetrocloud_transcript() 