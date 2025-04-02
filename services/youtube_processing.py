import os
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
from youtube_transcript_api import YouTubeTranscriptApi
from supabase import create_client
import groq
from langchain.docstore.document import Document

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class YouTubeTranscriptProcessor:
    """
    Processes YouTube videos by extracting and translating transcripts,
    then storing them in Supabase with appropriate metadata.
    """
    
    def __init__(self):
        """Initialize processor with API clients."""
        # Initialize Supabase client
        supabase_url = os.getenv('SUPABASE_URL_DEV')
        supabase_key = os.getenv('SUPABASE_KEY_DEV')
        
        if not supabase_url or not supabase_key:
            logger.error("Missing Supabase credentials in environment variables")
            raise ValueError("Supabase credentials are required")
            
        self.supabase = create_client(supabase_url, supabase_key)
        
        # Initialize Groq client for translation if needed
        groq_api_key = os.getenv('GROQ_API_KEY')
        if not groq_api_key:
            logger.warning("GROQ_API_KEY not found in environment variables - translation may fail")
            
        self.groq_client = groq.Client(api_key=groq_api_key)
        logger.info("YouTubeTranscriptProcessor initialized")

    def process_videos(self, video_urls: List[str], space_id: str) -> Dict[str, Any]:
        """
        Process multiple YouTube videos concurrently.
        Returns a list of documents from all processed videos with their database IDs.
        
        Args:
            video_urls: List of YouTube video URLs to process
            space_id: The space ID to associate the videos with
            
        Returns:
            Dictionary with processing results
        """
        if not video_urls:
            logger.warning("No video URLs provided for processing")
            return {
                'documents': [],
                'document_ids': {},
                'status': 'error',
                'message': 'No video URLs provided for processing'
            }
        
        if not space_id:
            logger.error("No space_id provided for video processing")
            return {
                'documents': [],
                'document_ids': {},
                'status': 'error',
                'message': 'Space ID is required'
            }
            
        logger.info(f"Processing {len(video_urls)} videos for space {space_id}")
        
        # Process videos concurrently
        all_documents = []
        video_ids = {}
        failed_urls = []
        
        with ThreadPoolExecutor(max_workers=min(5, len(video_urls))) as executor:
            # Submit processing tasks for each video
            future_to_url = {
                executor.submit(self.process_single_video, video_url, space_id): video_url 
                for video_url in video_urls
            }
            
            # Collect results
            for future in as_completed(future_to_url):
                video_url = future_to_url[future]
                try:
                    result = future.result()
                    if result and result.get('status') == 'success':
                        all_documents.extend(result.get('documents', []))
                        if result.get('document_id'):
                            video_ids[video_url] = result['document_id']
                            
                            # If result contains a thumbnail, update the space thumbnail
                            if result.get('thumbnail'):
                                try:
                                    self.supabase.table('spaces').update({
                                        'thumbnail': result['thumbnail']
                                    }).eq('id', space_id).execute()
                                    logger.info(f"Updated space {space_id} thumbnail")
                                except Exception as e:
                                    logger.error(f"Error updating space thumbnail: {e}")
                    else:
                        failed_urls.append(video_url)
                        logger.warning(f"Failed to process {video_url}: {result.get('message')}")
                except Exception as e:
                    failed_urls.append(video_url)
                    logger.error(f"Error processing {video_url}: {e}")
        
        # Prepare result summary
        success = len(all_documents) > 0
        message = f"Processed {len(video_ids)}/{len(video_urls)} videos successfully"
        if failed_urls:
            message += f". Failed URLs: {failed_urls[:3]}"
            if len(failed_urls) > 3:
                message += f" and {len(failed_urls) - 3} more"
        
        return {
            'documents': all_documents,
            'document_ids': video_ids,
            'status': 'success' if success else 'error',
            'message': message,
            'total_processed': len(video_ids),
            'failed_count': len(failed_urls)
        }

    def process_single_video(self, video_url: str, space_id: str) -> Dict[str, Any]:
        """
        Process a single YouTube video, extract transcript, and store in Supabase.
        
        Args:
            video_url: YouTube video URL
            space_id: Space ID to associate the video with
            
        Returns:
            Dictionary with processing results
        """
        logger.info(f"Processing video: {video_url}")
        
        try:
            # Extract video ID from URL
            video_id = self._extract_video_id(video_url)
            if not video_id:
                return {
                    'documents': [],
                    'document_id': None,
                    'status': 'error',
                    'message': f'Invalid YouTube URL format: {video_url}'
                }
            
            # Get video thumbnail
            thumbnail = f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"
            
            # Try to get transcript with fallback translation
            transcript_result = self._get_transcript_with_fallback(video_id)
            if not transcript_result.get('success'):
                return {
                    'documents': [],
                    'document_id': None,
                    'status': 'error',
                    'message': transcript_result.get('message', 'Failed to get transcript')
                }
                
            full_text = transcript_result['text']
            
            # Extract basic video metadata
            video_title = self._extract_video_title(video_url, video_id)
            
            # Insert into Supabase yts table
            try:
                result = self.supabase.table('yts').insert({
                    'space_id': space_id,
                    'extracted_text': full_text,
                    'yt_url': video_url,
                    'thumbnail': thumbnail
                }).execute()
                
                if not result.data:
                    raise ValueError(f"Supabase returned empty result for video {video_id}")
                    
                video_db_id = result.data[0]['id']
                
                # Store YouTube ID in the record
                # self.supabase.table('yts').update({
                #     'youtube_id': video_id
                # }).eq('id', video_db_id).execute()
            except Exception as db_error:
                logger.error(f"Database error for video {video_id}: {db_error}")
                return {
                    'documents': [],
                    'document_id': None,
                    'status': 'error',
                    'message': f'Database error: {str(db_error)}'
                }
            
            # Create a document object with video metadata
            doc = Document(
                page_content=full_text,
                metadata={
                    'source_url': video_url,
                    'document_id': video_db_id,
                    'video_id': video_db_id,
                    'youtube_id': video_id,
                    'space_id': space_id,
                    'content_type': 'youtube_transcript',
                    'title': video_title,
                    'source': 'youtube',
                    'source_type': 'youtube_video',
                    'thumbnail': thumbnail,
                    'file_type': 'youtube'
                }
            )
            
            logger.info(f"Successfully processed video {video_id}")
            return {
                'documents': [doc],
                'document_id': video_db_id,
                'video_id': video_db_id,
                'status': 'success',
                'message': f'Successfully processed YouTube video {video_url}',
                'thumbnail': thumbnail,
                'title': video_title
            }
        
        except Exception as e:
            logger.error(f"Error processing YouTube video {video_url}: {e}")
            return {
                'documents': [],
                'document_id': None,
                'status': 'error',
                'message': f'Error processing YouTube video {video_url}: {str(e)}'
            }

    def _extract_video_id(self, video_url: str) -> Optional[str]:
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

    def _transcript_to_text(self, transcript: List[Dict[str, Any]]) -> str:
        """
        Convert YouTube transcript format to plain text while preserving timestamps.
        
        Args:
            transcript: List of transcript entries from YouTube API
            
        Returns:
            Plain text representation of the transcript with timestamps
        """
        if not transcript:
            return ""
        
        try:
            # Sort transcript by start time to ensure proper ordering
            sorted_transcript = sorted(transcript, key=lambda x: x.get('start', 0))
            
            # Format each entry with timestamp and text
            formatted_entries = []
            for entry in sorted_transcript:
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
            return " ".join([entry.get('text', '') for entry in transcript if entry.get('text')])
            
    def _get_transcript_with_fallback(self, video_id: str) -> Dict[str, Any]:
        """
        Get transcript with fallback to translation if needed.
        
        Args:
            video_id: YouTube video ID
            
        Returns:
            Dictionary with success status and transcript text
        """
        # Try to get English transcript first
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
            full_text = self._transcript_to_text(transcript)
            return {'success': True, 'text': full_text}
        except Exception as en_error:
            logger.warning(f"Error getting English transcript for {video_id}: {en_error}")
            
            # Try to get transcript in any language and translate
            try:
                transcript = YouTubeTranscriptApi.get_transcript(video_id)
                
                # Check if we got a valid transcript
                if not transcript:
                    return {
                        'success': False, 
                        'message': f'No transcript available for video {video_id}'
                    }
                
                # Extract raw text from transcript
                raw_text = " ".join([line['text'] for line in transcript])
                
                # Translate using Groq
                response = self.groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",  # Using Llama model from Groq
                    messages=[
                        {"role": "system", "content": "You are a translator. Translate the following text to English:"},
                        {"role": "user", "content": raw_text}
                    ]
                )
                
                translated_text = response.choices[0].message.content
                return {'success': True, 'text': translated_text}
                
            except Exception as translate_error:
                logger.error(f"Failed to get and translate transcript for {video_id}: {translate_error}")
                return {
                    'success': False,
                    'message': f'Transcript retrieval failed: {str(translate_error)}'
                }
    
    def _extract_video_title(self, video_url: str, video_id: str) -> str:
        """
        Extract or generate a title for the video.
        
        Args:
            video_url: YouTube video URL
            video_id: YouTube video ID
            
        Returns:
            Video title
        """
        # Default title using video ID
        video_title = f"YouTube Video: {video_id}"
        
        try:
            # Try to extract title from URL if available
            if "title=" in video_url:
                video_title = video_url.split("title=")[1].split("&")[0]
                video_title = video_title.replace("+", " ")
                
            # In a production system, you would use YouTube API here
            # This is a placeholder for the actual implementation
        except Exception as title_error:
            logger.warning(f"Error extracting video title: {title_error}")
            
        return video_title 