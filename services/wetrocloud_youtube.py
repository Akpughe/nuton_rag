import os
import logging
import requests
from typing import Dict, Any, Optional, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WetroCloudYouTubeService:
    """
    Handles YouTube transcript extraction using the WetroCloud API.
    """
    
    def __init__(self):
        """Initialize the WetroCloud YouTube service with API credentials."""
        self.api_key = os.getenv('WETROCLOUD_API_KEY', 'wtc-sk-e169406c8358895cdfae650adc21452f97aba4d4')
        self.api_url = "https://api.wetrocloud.com/v1/youtube-transcript/"
        logger.info("WetroCloudYouTubeService initialized")
    
    def extract_video_id(self, video_url: str) -> Optional[str]:
        """
        Extract video ID from various YouTube URL formats.
        
        Args:
            video_url: YouTube video URL
            
        Returns:
            Video ID if found, None otherwise
        """
        if not video_url:
            return None
            
        try:
            if "youtu.be/" in video_url:
                # Handle youtu.be format
                return video_url.split("youtu.be/")[1].split("?")[0].split("&")[0]
            elif "youtube.com/watch?v=" in video_url:
                # Handle youtube.com format
                return video_url.split("watch?v=")[1].split("&")[0]
            elif "youtube.com/v/" in video_url:
                # Handle youtube.com/v/ format
                return video_url.split("/v/")[1].split("?")[0].split("&")[0]
            elif "youtube.com/embed/" in video_url:
                # Handle embed format
                return video_url.split("/embed/")[1].split("?")[0].split("&")[0]
            else:
                logger.warning(f"Unrecognized YouTube URL format: {video_url}")
                return None
        except Exception as e:
            logger.error(f"Error extracting video ID from {video_url}: {e}")
            return None
    
    def get_transcript(self, video_url: str) -> Dict[str, Any]:
        """
        Get transcript for a YouTube video using the WetroCloud API.
        
        Args:
            video_url: YouTube video URL
            
        Returns:
            Dictionary with transcript data and status
        """
        logger.info(f"Getting transcript for video: {video_url}")
        
        try:
            # Prepare API request
            headers = {
                'Authorization': f'Token {self.api_key}'
            }
            
            payload = {
                'youtube_link': video_url
            }
            
            # Make API request
            response = requests.post(
                self.api_url, 
                headers=headers, 
                data=payload
            )
            
            # Check if request was successful
            if response.status_code == 200:
                data = response.json()
                
                if data.get('success', False):
                    # Extract transcript entries
                    transcript_entries = data.get('response', [])
                    
                    # Convert to text
                    full_text = self.transcript_to_text(transcript_entries)
                    
                    # Get video ID for thumbnail
                    video_id = self.extract_video_id(video_url)
                    thumbnail = f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg" if video_id else None
                    
                    return {
                        'success': True,
                        'text': full_text,
                        'video_id': video_id,
                        'thumbnail': thumbnail,
                        'transcript_entries': transcript_entries,
                        'tokens': data.get('tokens', 0)
                    }
                else:
                    error_msg = f"WetroCloud API returned success=false for {video_url}"
                    logger.error(error_msg)
                    return {
                        'success': False,
                        'message': error_msg
                    }
            else:
                error_msg = f"WetroCloud API request failed with status code {response.status_code}: {response.text}"
                logger.error(error_msg)
                return {
                    'success': False,
                    'message': error_msg
                }
                
        except Exception as e:
            error_msg = f"Error getting transcript from WetroCloud API: {e}"
            logger.error(error_msg)
            return {
                'success': False,
                'message': error_msg
            }
    
    def transcript_to_text(self, transcript_entries: List[Dict[str, Any]]) -> str:
        """
        Convert transcript entries to formatted text with timestamps.
        
        Args:
            transcript_entries: List of transcript entries from WetroCloud API
            
        Returns:
            Formatted transcript text
        """
        if not transcript_entries:
            return ""
        
        try:
            # Sort transcript by start time to ensure proper ordering
            sorted_entries = sorted(transcript_entries, key=lambda x: x.get('start', 0))
            
            # Format each entry with timestamp and text
            formatted_entries = []
            for entry in sorted_entries:
                if entry.get('text'):
                    # Convert timestamp to [MM:SS] format
                    seconds = int(entry.get('start', 0))
                    minutes = seconds // 60
                    remaining_seconds = seconds % 60
                    timestamp = f"[{minutes:02d}:{remaining_seconds:02d}]"
                    
                    # Combine timestamp with text
                    formatted_entries.append(f"{timestamp} {entry.get('text').strip()}")
            
            # Join all segments with newlines for better readability
            return "\n".join(formatted_entries)
        except Exception as e:
            logger.error(f"Error converting transcript to text: {e}")
            # Fallback to simple concatenation
            return " ".join([entry.get('text', '') for entry in transcript_entries if entry.get('text')])
    
    def get_video_title(self, video_url: str, yt_api_url: str = None) -> str:
        """
        Get the title of a YouTube video.
        
        Args:
            video_url: YouTube video URL
            yt_api_url: Optional URL for title API fallback
            
        Returns:
            Video title or default title with video ID
        """
        video_id = self.extract_video_id(video_url)
        if not video_id:
            return "Unknown YouTube Video"
        
        try:
            # Try using existing title API if URL provided
            if yt_api_url:
                response = requests.post(
                    f"{yt_api_url}/yt-video-title",
                    json={'url': video_url}
                )
                response.raise_for_status()
                data = response.json()
                
                if data and 'title' in data:
                    return data['title']
            
            # Default title using video ID
            return f"YouTube Video: {video_id}"
              
        except Exception as e:
            logger.warning(f"Error getting video title: {e}")
            return f"YouTube Video: {video_id}" 