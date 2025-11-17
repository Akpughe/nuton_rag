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

    def __init__(self, enable_vcyon_fallback: bool = True, enable_ytdlp_fallback: bool = True):
        """
        Initialize the WetroCloud YouTube service with API credentials.

        Args:
            enable_vcyon_fallback: Enable Vcyon as backup when WetroCloud fails (default: True)
            enable_ytdlp_fallback: Enable yt-dlp as backup when WetroCloud and Vcyon fail (default: True)
        """
        self.api_key = os.getenv('WETROCLOUD_API_KEY', 'wtc-sk-e169406c8358895cdfae650adc21452f97aba4d4')
        self.api_url = "https://api.wetrocloud.com/v2/transcript/"

        # Vcyon API configuration
        self.vcyon_api_key = os.getenv('VCYON_API_KEY', '43aa67a8-3f7e-4308-97a8-91eefff24575')
        self.vcyon_base_url = "https://api.vcyon.com/v1/youtube"
        self.enable_vcyon_fallback = enable_vcyon_fallback

        # yt-dlp fallback configuration
        self.enable_ytdlp_fallback = enable_ytdlp_fallback and YTDLP_AVAILABLE

        if self.enable_ytdlp_fallback:
            self.ytdlp_service = YTDLPTranscriptService()
            logger.info("WetroCloudYouTubeService initialized with Vcyon and yt-dlp fallback enabled")
        elif self.enable_vcyon_fallback:
            self.ytdlp_service = None
            logger.info("WetroCloudYouTubeService initialized with Vcyon fallback enabled")
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

    def _get_video_info_from_vcyon(self, video_url: str) -> Optional[Dict[str, Any]]:
        """
        Get video information from Vcyon API.

        Args:
            video_url: YouTube video URL

        Returns:
            Dictionary with video information or None if failed
        """
        video_id = self.extract_video_id(video_url)
        if not video_id:
            logger.warning(f"Could not extract video ID from {video_url}")
            return None

        try:
            logger.info(f"🔄 Getting video info from Vcyon for video ID: {video_id}")

            headers = {
                'Authorization': f'Bearer {self.vcyon_api_key}'
            }

            params = {
                'videoId': video_id
            }

            response = requests.get(
                f"{self.vcyon_base_url}/video",
                headers=headers,
                params=params,
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()

                if data.get('success'):
                    video_data = data.get('data', {})
                    logger.info(f"✅ Vcyon video info retrieved for {video_url}")
                    return video_data
                else:
                    logger.warning(f"Vcyon video info API returned success=false for {video_url}")
                    return None
            else:
                logger.warning(f"Vcyon video info API returned status {response.status_code}: {response.text}")
                return None

        except Exception as e:
            logger.warning(f"Error getting video info from Vcyon: {e}")
            return None

    def _get_transcript_from_vcyon(self, video_url: str, languages: List[str] = ['en']) -> Dict[str, Any]:
        """
        Get transcript from Vcyon API.

        Args:
            video_url: YouTube video URL
            languages: Preferred languages (default: ['en'])

        Returns:
            Dictionary with transcript data and status
        """
        video_id = self.extract_video_id(video_url)
        if not video_id:
            return {
                'success': False,
                'message': f"Could not extract video ID from {video_url}"
            }

        try:
            logger.info(f"🔄 Attempting Vcyon API for video ID: {video_id}")

            headers = {
                'Authorization': f'Bearer {self.vcyon_api_key}'
            }

            # Use the first language from the list, default to "English"
            language_param = "English" if not languages or languages[0] == 'en' else languages[0]

            params = {
                'videoId': video_id,
                'language': language_param
            }

            response = requests.get(
                f"{self.vcyon_base_url}/transcript",
                headers=headers,
                params=params,
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()

                if data.get('success') and data.get('data', {}).get('hasTranscript'):
                    transcript_data = data['data']
                    segments = transcript_data.get('segments', [])

                    if not segments:
                        return {
                            'success': False,
                            'message': f"Vcyon API reports no transcript available for {video_url}"
                        }

                    # Convert segments to text with timestamps (matching WetroCloud format)
                    formatted_entries = []
                    for segment in segments:
                        # Convert milliseconds to seconds for consistency
                        start_seconds = int(segment.get('start', 0) / 1000)
                        minutes = start_seconds // 60
                        remaining_seconds = start_seconds % 60
                        timestamp = f"[{minutes:02d}:{remaining_seconds:02d}]"

                        text = segment.get('text', '').strip()
                        if text:
                            formatted_entries.append(f"{timestamp} {text}")

                    full_text = "\n".join(formatted_entries)
                    thumbnail = f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg"

                    # Convert segments to match expected format
                    transcript_entries = []
                    for segment in segments:
                        transcript_entries.append({
                            'start': segment.get('start', 0) / 1000,  # Convert to seconds
                            'end': segment.get('end', 0) / 1000,      # Convert to seconds
                            'text': segment.get('text', '')
                        })

                    logger.info(f"✅ Vcyon transcript retrieved for {video_url}")
                    return {
                        'success': True,
                        'text': full_text,
                        'video_id': video_id,
                        'thumbnail': thumbnail,
                        'transcript_entries': transcript_entries,
                        'language': transcript_data.get('language', language_param),
                        'method': 'vcyon'
                    }
                else:
                    return {
                        'success': False,
                        'message': f"Vcyon API reports no transcript available for {video_url}"
                    }

            else:
                error_msg = f"Vcyon API returned status {response.status_code}: {response.text}"
                logger.warning(error_msg)
                return {
                    'success': False,
                    'message': error_msg
                }

        except Exception as e:
            error_msg = f"Vcyon API error: {e}"
            logger.warning(error_msg)
            return {
                'success': False,
                'message': error_msg
            }

    def get_transcript(self, video_url: str, languages: List[str] = ['en']) -> Dict[str, Any]:
        """
        Get transcript for a YouTube video using the WetroCloud API.
        Falls back to Vcyon, then yt-dlp if WetroCloud fails.

        Args:
            video_url: YouTube video URL
            languages: Preferred languages for fallback (default: ['en'])

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
                    return self._try_fallbacks(video_url, languages, error_msg)
            else:
                error_msg = f"WetroCloud API request failed with status code {response.status_code}: {response.text}"
                logger.warning(error_msg)
                return self._try_fallbacks(video_url, languages, error_msg)

        except requests.exceptions.Timeout:
            error_msg = f"WetroCloud API timeout for {video_url}"
            logger.warning(error_msg)
            return self._try_fallbacks(video_url, languages, error_msg)

        except Exception as e:
            error_msg = f"Error getting transcript from WetroCloud API: {e}"
            logger.warning(error_msg)
            return self._try_fallbacks(video_url, languages, error_msg)

    def _try_fallbacks(self, video_url: str, languages: List[str], wetrocloud_error: str) -> Dict[str, Any]:
        """
        Try fallback methods in order: Vcyon -> yt-dlp

        Args:
            video_url: YouTube video URL
            languages: Preferred languages
            wetrocloud_error: Error message from WetroCloud

        Returns:
            Dictionary with transcript data and status
        """
        # Try Vcyon fallback first if enabled
        if self.enable_vcyon_fallback:
            logger.info("⚠️ WetroCloud failed, trying Vcyon fallback...")
            vcyon_result = self._get_transcript_from_vcyon(video_url, languages)

            if vcyon_result['success']:
                # Add WetroCloud error for reference
                vcyon_result['wetrocloud_error'] = wetrocloud_error
                logger.info(f"✅ Vcyon fallback succeeded for {video_url}")
                return vcyon_result
            else:
                vcyon_error = vcyon_result.get('message', 'Unknown error')
                logger.warning(f"Vcyon also failed: {vcyon_error}")

                # Try yt-dlp as last resort
                if self.enable_ytdlp_fallback:
                    logger.info("⚠️ WetroCloud and Vcyon failed, trying yt-dlp fallback...")
                    return self._fallback_to_ytdlp(video_url, languages, f"WetroCloud: {wetrocloud_error}. Vcyon: {vcyon_error}")

                return {
                    'success': False,
                    'message': f"All methods failed. WetroCloud: {wetrocloud_error}. Vcyon: {vcyon_error}"
                }

        # If Vcyon is not enabled, try yt-dlp directly
        if self.enable_ytdlp_fallback:
            logger.info("⚠️ WetroCloud failed, trying yt-dlp fallback...")
            return self._fallback_to_ytdlp(video_url, languages, wetrocloud_error)

        return {
            'success': False,
            'message': wetrocloud_error
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