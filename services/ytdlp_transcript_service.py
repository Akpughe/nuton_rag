import os
import logging
import tempfile
from typing import Dict, Any, Optional, List
import yt_dlp
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class YTDLPTranscriptService:
    """
    YouTube transcript extraction using yt-dlp.

    yt-dlp is more robust than youtube-transcript-api because:
    - Uses YouTube's internal APIs directly
    - Actively maintained and updated for YouTube changes
    - Doesn't get blocked by IP restrictions in most cases
    - Can extract both manual and auto-generated captions
    - No proxy required in 99% of cases
    """

    def __init__(self, browser: Optional[str] = None):
        """
        Initialize the yt-dlp transcript service.

        Args:
            browser: Browser to extract cookies from (chrome, firefox, safari, edge, etc.)
                    If None, will try common browsers in order.
        """
        self.browser = browser
        logger.info(f"YTDLPTranscriptService initialized with browser: {browser or 'auto-detect'}")

    def _get_base_ydl_opts(self) -> Dict[str, Any]:
        """
        Get base yt-dlp options with cookie handling.

        Returns:
            Dictionary of yt-dlp options
        """
        opts = {
            'quiet': True,
            'no_warnings': True,
        }

        # Add cookie extraction from browser
        if self.browser:
            # Use specific browser if provided
            opts['cookiesfrombrowser'] = (self.browser,)
            logger.info(f"Using cookies from browser: {self.browser}")
        else:
            # Try common browsers in order (works on macOS, Linux, Windows)
            # yt-dlp will try each until it finds one with valid cookies
            try:
                # Try chrome first (most common)
                opts['cookiesfrombrowser'] = ('chrome',)
                logger.info("Attempting to use cookies from Chrome")
            except Exception:
                logger.warning("Could not use Chrome cookies, will try default")

        return opts

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
        Get transcript for a YouTube video using yt-dlp.

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

        logger.info(f"Extracting transcript for video ID: {video_id} with yt-dlp")

        # Create temporary directory for subtitle files
        with tempfile.TemporaryDirectory() as temp_dir:
            output_template = os.path.join(temp_dir, '%(id)s.%(ext)s')

            # yt-dlp options for subtitle extraction only
            ydl_opts = self._get_base_ydl_opts()
            ydl_opts.update({
                'skip_download': True,  # Don't download video
                'writesubtitles': True,  # Download manual subtitles
                'writeautomaticsub': True,  # Download auto-generated subtitles
                'subtitleslangs': languages,  # Preferred languages
                'subtitlesformat': 'vtt',  # VTT format (easier to parse)
                'outtmpl': output_template,
                'extract_flat': False,
            })

            try:
                # Extract subtitles
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(video_url, download=False)

                    # Check if subtitles are available
                    subtitles = info.get('subtitles', {})
                    automatic_captions = info.get('automatic_captions', {})

                    # Prioritize manual subtitles over automatic
                    available_subs = subtitles if subtitles else automatic_captions

                    if not available_subs:
                        return {
                            'success': False,
                            'message': f'No subtitles available for video {video_id}'
                        }

                    # Find the best matching language
                    selected_lang = None
                    for lang in languages:
                        if lang in available_subs:
                            selected_lang = lang
                            break

                    # If no exact match, try the first available language
                    if not selected_lang and available_subs:
                        selected_lang = list(available_subs.keys())[0]
                        logger.info(f"Requested languages {languages} not available, using: {selected_lang}")

                    if not selected_lang:
                        return {
                            'success': False,
                            'message': 'No matching subtitles found'
                        }

                    # Download the subtitle file
                    ydl_opts['subtitleslangs'] = [selected_lang]
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl2:
                        ydl2.download([video_url])

                    # Read the downloaded subtitle file
                    subtitle_file = os.path.join(temp_dir, f"{video_id}.{selected_lang}.vtt")

                    if not os.path.exists(subtitle_file):
                        # Try without language code
                        subtitle_file = os.path.join(temp_dir, f"{video_id}.vtt")

                    if not os.path.exists(subtitle_file):
                        return {
                            'success': False,
                            'message': f'Subtitle file not found for language: {selected_lang}'
                        }

                    # Parse VTT file
                    with open(subtitle_file, 'r', encoding='utf-8') as f:
                        vtt_content = f.read()

                    # Convert VTT to structured transcript
                    transcript_entries = self._parse_vtt(vtt_content)
                    full_text = self.transcript_to_text(transcript_entries)

                    # Get video metadata
                    title = info.get('title', f'YouTube Video: {video_id}')
                    thumbnail = info.get('thumbnail', f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg")

                    return {
                        'success': True,
                        'text': full_text,
                        'video_id': video_id,
                        'video_title': title,
                        'thumbnail': thumbnail,
                        'transcript_entries': transcript_entries,
                        'language': selected_lang,
                        'is_automatic': selected_lang in automatic_captions,
                        'method': 'yt-dlp'
                    }

            except yt_dlp.utils.DownloadError as e:
                error_msg = f"yt-dlp download error: {str(e)}"
                logger.error(error_msg)
                return {
                    'success': False,
                    'message': error_msg
                }
            except Exception as e:
                error_msg = f"Unexpected error: {str(e)}"
                logger.error(error_msg)
                return {
                    'success': False,
                    'message': error_msg
                }

    def _parse_vtt(self, vtt_content: str) -> List[Dict[str, Any]]:
        """
        Parse WebVTT subtitle format to extract transcript entries.

        Args:
            vtt_content: Raw VTT file content

        Returns:
            List of transcript entries with start time and text
        """
        entries = []

        # Remove VTT header
        content = re.sub(r'^WEBVTT\n\n', '', vtt_content, flags=re.MULTILINE)

        # Split into cue blocks
        blocks = content.split('\n\n')

        for block in blocks:
            if not block.strip():
                continue

            lines = block.strip().split('\n')

            # Look for timestamp line (format: 00:00:00.000 --> 00:00:05.000)
            timestamp_pattern = r'(\d{2}:\d{2}:\d{2}\.\d{3})\s+-->\s+(\d{2}:\d{2}:\d{2}\.\d{3})'

            for i, line in enumerate(lines):
                match = re.search(timestamp_pattern, line)
                if match:
                    start_time = match.group(1)
                    # Get text (everything after timestamp line)
                    text_lines = lines[i+1:]
                    text = ' '.join(text_lines).strip()

                    # Remove VTT tags like <c> </c>
                    text = re.sub(r'<[^>]+>', '', text)

                    if text:
                        # Convert timestamp to seconds
                        h, m, s = start_time.split(':')
                        start_seconds = int(h) * 3600 + int(m) * 60 + float(s)

                        entries.append({
                            'start': start_seconds,
                            'text': text
                        })
                    break

        return entries

    def transcript_to_text(self, transcript_entries: List[Dict[str, Any]]) -> str:
        """
        Convert transcript entries to formatted text with timestamps.

        Args:
            transcript_entries: List of transcript entries

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

    def get_available_subtitles(self, video_url: str) -> Dict[str, Any]:
        """
        Get list of available subtitle languages for a video.

        Args:
            video_url: YouTube video URL or video ID

        Returns:
            Dictionary with available subtitles info
        """
        video_id = self.extract_video_id(video_url)

        if not video_id:
            return {
                'success': False,
                'message': f"Invalid YouTube URL or video ID: {video_url}"
            }

        try:
            ydl_opts = self._get_base_ydl_opts()
            ydl_opts.update({
                'skip_download': True,
            })

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(video_url, download=False)

                subtitles = info.get('subtitles', {})
                automatic_captions = info.get('automatic_captions', {})

                available_languages = []

                # Manual subtitles
                for lang in subtitles.keys():
                    available_languages.append({
                        'language_code': lang,
                        'is_generated': False,
                        'is_translatable': True
                    })

                # Automatic captions
                for lang in automatic_captions.keys():
                    if lang not in subtitles:  # Avoid duplicates
                        available_languages.append({
                            'language_code': lang,
                            'is_generated': True,
                            'is_translatable': True
                        })

                return {
                    'success': True,
                    'video_id': video_id,
                    'video_title': info.get('title', 'Unknown'),
                    'available_subtitles': available_languages,
                    'method': 'yt-dlp'
                }

        except Exception as e:
            error_msg = f"Error listing subtitles: {str(e)}"
            logger.error(error_msg)
            return {'success': False, 'message': error_msg}

    def get_video_info(self, video_url: str) -> Dict[str, Any]:
        """
        Get video metadata.

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

        try:
            ydl_opts = self._get_base_ydl_opts()
            ydl_opts.update({
                'skip_download': True,
            })

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(video_url, download=False)

                return {
                    'success': True,
                    'video_id': video_id,
                    'title': info.get('title', 'Unknown'),
                    'description': info.get('description', ''),
                    'duration': info.get('duration', 0),
                    'channel': info.get('channel', 'Unknown'),
                    'upload_date': info.get('upload_date', ''),
                    'view_count': info.get('view_count', 0),
                    'thumbnail': info.get('thumbnail', f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg"),
                    'video_url': f"https://www.youtube.com/watch?v={video_id}"
                }

        except Exception as e:
            error_msg = f"Error getting video info: {str(e)}"
            logger.error(error_msg)
            return {'success': False, 'message': error_msg}
