import os
import logging
from typing import Dict, Any, Optional, List
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.proxies import WebshareProxyConfig
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
    RequestBlocked,
    IpBlocked,
    CouldNotRetrieveTranscript
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class YouTubeTranscriptService:
    """
    Handles YouTube transcript extraction using youtube-transcript-api library.
    Supports WebShare proxy configuration to bypass cloud provider IP blocking.
    """

    def __init__(self, use_proxy: bool = False):
        """
        Initialize the YouTube Transcript Service.

        Args:
            use_proxy: Whether to use WebShare proxy (recommended for cloud deployments)
        """
        self.use_proxy = use_proxy
        self.api_instance = None
        self.proxy_status = "disabled"

        if self.use_proxy:
            # Get WebShare credentials from environment
            proxy_username = os.getenv('WEBSHARE_PROXY_USERNAME')
            proxy_password = os.getenv('WEBSHARE_PROXY_PASSWORD')

            if proxy_username and proxy_password:
                self.api_instance = YouTubeTranscriptApi(
                    proxy_config=WebshareProxyConfig(
                        proxy_username=proxy_username,
                        proxy_password=proxy_password,
                    )
                )
                self.proxy_status = "enabled"
                logger.info("✅ YouTubeTranscriptService initialized with WebShare proxy")
            else:
                missing = []
                if not proxy_username:
                    missing.append("WEBSHARE_PROXY_USERNAME")
                if not proxy_password:
                    missing.append("WEBSHARE_PROXY_PASSWORD")

                self.proxy_status = f"failed - missing: {', '.join(missing)}"
                logger.error(f"❌ WebShare proxy requested but credentials not found in environment.")
                logger.error(f"   Missing environment variables: {', '.join(missing)}")
                logger.error(f"   Set these in your .env file or environment")
                self.use_proxy = False

        if not self.use_proxy:
            logger.info("YouTubeTranscriptService initialized without proxy")

    def get_proxy_status(self) -> Dict[str, Any]:
        """
        Get the current proxy configuration status.

        Returns:
            Dictionary with proxy status information
        """
        return {
            'proxy_enabled': self.use_proxy,
            'proxy_status': self.proxy_status,
            'has_credentials': bool(os.getenv('WEBSHARE_PROXY_USERNAME') and os.getenv('WEBSHARE_PROXY_PASSWORD'))
        }

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
            # Handle different YouTube URL formats
            if "youtu.be/" in video_url:
                return video_url.split("youtu.be/")[1].split("?")[0].split("&")[0]
            elif "youtube.com/watch?v=" in video_url:
                return video_url.split("watch?v=")[1].split("&")[0]
            elif "youtube.com/v/" in video_url:
                return video_url.split("/v/")[1].split("?")[0].split("&")[0]
            elif "youtube.com/embed/" in video_url:
                return video_url.split("/embed/")[1].split("?")[0].split("&")[0]
            elif len(video_url) == 11:
                # Assume it's already a video ID
                return video_url
            else:
                logger.warning(f"Unrecognized YouTube URL format: {video_url}")
                return None
        except Exception as e:
            logger.error(f"Error extracting video ID from {video_url}: {e}")
            return None

    def get_transcript(
        self,
        video_url: str,
        languages: List[str] = ['en']
    ) -> Dict[str, Any]:
        """
        Get transcript for a YouTube video.

        Args:
            video_url: YouTube video URL or video ID
            languages: List of preferred language codes (default: ['en'])

        Returns:
            Dictionary with transcript data and status
        """
        video_id = self.extract_video_id(video_url)

        if not video_id:
            error_msg = f"Invalid YouTube URL or video ID: {video_url}"
            logger.error(error_msg)
            return {
                'success': False,
                'message': error_msg
            }

        logger.info(f"Getting transcript for video ID: {video_id}")

        try:
            # Fetch transcript using the appropriate API instance
            if self.use_proxy and self.api_instance:
                transcript_list = self.api_instance.get_transcript(video_id, languages=languages)
            else:
                transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=languages)

            # Format transcript to text
            full_text = self.transcript_to_text(transcript_list)

            # Generate thumbnail URL
            thumbnail = f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg"

            return {
                'success': True,
                'text': full_text,
                'video_id': video_id,
                'thumbnail': thumbnail,
                'transcript_entries': transcript_list,
                'language': languages[0] if languages else 'en'
            }

        except TranscriptsDisabled:
            error_msg = f"Transcripts are disabled for video: {video_id}"
            logger.error(error_msg)
            return {'success': False, 'message': error_msg}

        except NoTranscriptFound:
            error_msg = f"No transcript found for video: {video_id} in languages: {languages}"
            logger.error(error_msg)
            return {'success': False, 'message': error_msg}

        except VideoUnavailable:
            error_msg = f"Video is unavailable: {video_id}"
            logger.error(error_msg)
            return {'success': False, 'message': error_msg}

        except IpBlocked:
            error_msg = "IP blocked by YouTube. This usually happens with cloud provider IPs. Enable WebShare proxy to fix this."
            logger.error(error_msg)
            return {'success': False, 'message': error_msg}

        except RequestBlocked:
            error_msg = "Request blocked by YouTube. Too many requests from this IP. Enable WebShare proxy to fix this."
            logger.error(error_msg)
            return {'success': False, 'message': error_msg}

        except CouldNotRetrieveTranscript as e:
            error_msg = f"Could not retrieve transcript: {str(e)}"
            logger.error(error_msg)
            return {'success': False, 'message': error_msg}

        except Exception as e:
            error_msg = f"Unexpected error getting transcript: {str(e)}"
            logger.error(error_msg)

            # Provide helpful suggestions based on error type
            suggestions = []
            error_str = str(e).lower()

            if "no element found" in error_str or "line 1, column 0" in error_str:
                suggestions.append("YouTube returned an empty response. This usually means:")
                suggestions.append("- Your IP is blocked (try enabling WebShare proxy with use_proxy=true)")
                suggestions.append("- The video is age-restricted or requires login")
                suggestions.append("- The video doesn't have transcripts available")

            if "http" in error_str or "connection" in error_str:
                suggestions.append("Network or connection issue. Check your internet connection.")

            result = {'success': False, 'message': error_msg}
            if suggestions:
                result['suggestions'] = suggestions

            return result

    def get_available_transcripts(self, video_url: str) -> Dict[str, Any]:
        """
        Get list of available transcript languages for a video.

        Args:
            video_url: YouTube video URL or video ID

        Returns:
            Dictionary with available transcripts info
        """
        video_id = self.extract_video_id(video_url)

        if not video_id:
            return {
                'success': False,
                'message': f"Invalid YouTube URL or video ID: {video_url}"
            }

        try:
            if self.use_proxy and self.api_instance:
                transcript_list = self.api_instance.list_transcripts(video_id)
            else:
                transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

            available_languages = []

            # Manual transcripts
            for transcript in transcript_list:
                available_languages.append({
                    'language': transcript.language,
                    'language_code': transcript.language_code,
                    'is_generated': transcript.is_generated,
                    'is_translatable': transcript.is_translatable
                })

            return {
                'success': True,
                'video_id': video_id,
                'available_transcripts': available_languages
            }

        except Exception as e:
            error_msg = f"Error listing available transcripts: {str(e)}"
            logger.error(error_msg)
            return {'success': False, 'message': error_msg}

    def transcript_to_text(self, transcript_entries: List[Dict[str, Any]]) -> str:
        """
        Convert transcript entries to formatted text with timestamps.

        Args:
            transcript_entries: List of transcript entries with 'start', 'duration', and 'text'

        Returns:
            Formatted transcript text
        """
        if not transcript_entries:
            return ""

        try:
            formatted_entries = []

            for entry in transcript_entries:
                text = entry.get('text', '').strip()
                if text:
                    # Convert timestamp to [MM:SS] format
                    seconds = int(float(entry.get('start', 0)))
                    minutes = seconds // 60
                    remaining_seconds = seconds % 60
                    timestamp = f"[{minutes:02d}:{remaining_seconds:02d}]"

                    formatted_entries.append(f"{timestamp} {text}")

            return "\n".join(formatted_entries)

        except Exception as e:
            logger.error(f"Error converting transcript to text: {e}")
            # Fallback to simple concatenation
            return " ".join([entry.get('text', '').strip() for entry in transcript_entries if entry.get('text')])

    def get_video_info(self, video_url: str) -> Dict[str, Any]:
        """
        Get basic video information.

        Args:
            video_url: YouTube video URL or video ID

        Returns:
            Dictionary with video information
        """
        video_id = self.extract_video_id(video_url)

        if not video_id:
            return {
                'success': False,
                'message': f"Invalid YouTube URL or video ID: {video_url}"
            }

        return {
            'success': True,
            'video_id': video_id,
            'video_url': f"https://www.youtube.com/watch?v={video_id}",
            'thumbnail': f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg",
            'thumbnail_maxres': f"https://i.ytimg.com/vi/{video_id}/maxresdefault.jpg"
        }
