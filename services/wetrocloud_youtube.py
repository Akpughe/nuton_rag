import os
import logging
import requests
from typing import Dict, Any, Optional, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import yt-dlp service as backup
try:
    from services.ytdlp_transcript_service import YTDLPTranscriptService
    YTDLP_AVAILABLE = True
except ImportError:
    YTDLP_AVAILABLE = False
    logger.warning("yt-dlp service not available for fallback")

class WetroCloudYouTubeService:
    """
    Handles YouTube transcript extraction using the WetroCloud API.
    Falls back to yt-dlp if WetroCloud API fails.
    """

    def __init__(self, enable_ytdlp_fallback: bool = True):
        """
        Initialize the WetroCloud YouTube service with API credentials.

        Args:
            enable_ytdlp_fallback: Enable yt-dlp as backup when WetroCloud fails (default: True)
        """
        self.api_key = os.getenv('WETROCLOUD_API_KEY', 'wtc-sk-e169406c8358895cdfae650adc21452f97aba4d4')
        self.api_url = "https://api.wetrocloud.com/v2/transcript/"
        self.enable_ytdlp_fallback = enable_ytdlp_fallback and YTDLP_AVAILABLE

        if self.enable_ytdlp_fallback:
            self.ytdlp_service = YTDLPTranscriptService()
            logger.info("WetroCloudYouTubeService initialized with yt-dlp fallback enabled")
        else:
            self.ytdlp_service = None
            logger.info("WetroCloudYouTubeService initialized (no fallback)")
    
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
    
    def get_transcript(self, video_url: str, languages: List[str] = ['en']) -> Dict[str, Any]:
        """
        Get transcript for a YouTube video using the WetroCloud API.
        Falls back to yt-dlp if WetroCloud fails.

        Args:
            video_url: YouTube video URL
            languages: Preferred languages for fallback yt-dlp (default: ['en'])

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
                'link': video_url,
                'resource_type': 'youtube'
            }

            # Make API request
            response = requests.post(
                self.api_url,
                headers=headers,
                data=payload,
                timeout=30  # Add timeout to prevent hanging
            )

            # Check if request was successful
            if response.status_code == 200:
                data = response.json()

                if data.get('success', False):
                    # Extract transcript entries
                    transcript_entries = data.get('response', {}).get('data', [])

                    # Convert to text
                    full_text = self.transcript_to_text(transcript_entries)

                    # Get video ID for thumbnail
                    video_id = self.extract_video_id(video_url)
                    thumbnail = f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg" if video_id else None

                    logger.info(f"✅ WetroCloud API succeeded for {video_url}")
                    return {
                        'success': True,
                        'text': full_text,
                        'video_id': video_id,
                        'thumbnail': thumbnail,
                        'transcript_entries': transcript_entries,
                        'tokens': data.get('tokens', 0),
                        'method': 'wetrocloud'
                    }
                else:
                    error_msg = f"WetroCloud API returned success=false for {video_url}"
                    logger.warning(error_msg)

                    # Try fallback
                    if self.enable_ytdlp_fallback:
                        logger.info("⚠️ WetroCloud failed, trying yt-dlp fallback...")
                        return self._fallback_to_ytdlp(video_url, languages, error_msg)

                    return {
                        'success': False,
                        'message': error_msg
                    }
            else:
                error_msg = f"WetroCloud API request failed with status code {response.status_code}: {response.text}"
                logger.warning(error_msg)

                # Try fallback
                if self.enable_ytdlp_fallback:
                    logger.info("⚠️ WetroCloud failed, trying yt-dlp fallback...")
                    return self._fallback_to_ytdlp(video_url, languages, error_msg)

                return {
                    'success': False,
                    'message': error_msg
                }

        except requests.exceptions.Timeout:
            error_msg = f"WetroCloud API timeout for {video_url}"
            logger.warning(error_msg)

            # Try fallback
            if self.enable_ytdlp_fallback:
                logger.info("⚠️ WetroCloud timeout, trying yt-dlp fallback...")
                return self._fallback_to_ytdlp(video_url, languages, error_msg)

            return {
                'success': False,
                'message': error_msg
            }

        except Exception as e:
            error_msg = f"Error getting transcript from WetroCloud API: {e}"
            logger.warning(error_msg)

            # Try fallback
            if self.enable_ytdlp_fallback:
                logger.info("⚠️ WetroCloud error, trying yt-dlp fallback...")
                return self._fallback_to_ytdlp(video_url, languages, error_msg)

            return {
                'success': False,
                'message': error_msg
            }

    def _fallback_to_ytdlp(self, video_url: str, languages: List[str], wetrocloud_error: str) -> Dict[str, Any]:
        """
        Fallback to yt-dlp when WetroCloud fails.

        Args:
            video_url: YouTube video URL
            languages: Preferred languages
            wetrocloud_error: Error message from WetroCloud

        Returns:
            Dictionary with transcript data and status
        """
        if not self.ytdlp_service:
            return {
                'success': False,
                'message': f"WetroCloud failed and yt-dlp fallback not available. WetroCloud error: {wetrocloud_error}"
            }

        try:
            logger.info(f"🔄 Attempting yt-dlp fallback for {video_url}")
            result = self.ytdlp_service.get_transcript(video_url, languages)

            if result['success']:
                # Add fallback indicator to result
                result['method'] = 'ytdlp-fallback'
                result['wetrocloud_error'] = wetrocloud_error
                logger.info(f"✅ yt-dlp fallback succeeded for {video_url}")
                return result
            else:
                logger.error(f"❌ Both WetroCloud and yt-dlp failed for {video_url}")
                return {
                    'success': False,
                    'message': f"Both methods failed. WetroCloud: {wetrocloud_error}. yt-dlp: {result.get('message', 'Unknown error')}"
                }

        except Exception as e:
            logger.error(f"❌ yt-dlp fallback exception for {video_url}: {e}")
            return {
                'success': False,
                'message': f"Both methods failed. WetroCloud: {wetrocloud_error}. yt-dlp: {str(e)}"
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
                    # Convert timestamp to [MM:SS] format - handle floating point values
                    seconds = int(float(entry.get('start', 0)))
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