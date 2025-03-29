import os
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
from youtube_transcript_api import YouTubeTranscriptApi
from supabase import create_client
import groq
from langchain.docstore.document import Document

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class YouTubeTranscriptProcessor:
    def __init__(self):
        # Initialize Supabase client
        supabase_url = os.getenv('SUPABASE_URL_DEV')
        supabase_key = os.getenv('SUPABASE_KEY_DEV')
        self.supabase = create_client(supabase_url, supabase_key)
        # Initialize Groq client for translation if needed
        self.groq_client = groq.Client(api_key=os.getenv('GROQ_API_KEY'))

    def process_videos(self, video_urls: List[str], space_id: str) -> Dict[str, Any]:
        """
        Process multiple YouTube videos concurrently
        Returns a list of documents from all processed videos with their database IDs
        """
        all_documents = []
        video_ids = {}
        
        if not video_urls:
            return {
                'documents': [],
                'video_ids': {},
                'status': 'error',
                'message': 'No video URLs provided for processing'
            }
        
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
                    if result and result.get('documents'):
                        all_documents.extend(result['documents'])
                        if result.get('document_id'):
                            video_ids[video_url] = result['document_id']
                            
                            # If result contains a thumbnail, update the space thumbnail
                            if result.get('thumbnail'):
                                try:
                                    self.supabase.table('spaces').update({
                                        'thumbnail': result['thumbnail']
                                    }).eq('id', space_id).execute()
                                    logger.info(f"Updated space {space_id} with thumbnail from {video_url}")
                                except Exception as e:
                                    logger.error(f"Error updating space thumbnail: {e}")
                except Exception as e:
                    logger.error(f"Error processing {video_url}: {e}")
        
        return {
            'documents': all_documents,
            'document_ids': video_ids,
            'status': 'success' if all_documents else 'error',
            'message': f'Processed {len(all_documents)} video transcripts successfully' if all_documents else 'Failed to process any videos',
            'total_processed': len(video_ids)
        }

    def process_single_video(self, video_url: str, space_id: str) -> Dict[str, Any]:
        """
        Process a single YouTube video, extract transcript, and store in Supabase
        """
        try:
            # Extract video ID from various YouTube URL formats
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
            
            try:
                # Try to get English transcript first
                transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
                full_text = self._transcript_to_text(transcript)
            except Exception as e:
                logger.error(f"Error getting English transcript: {e}")
                try:
                    # If English not available, get transcript in any language and translate
                    transcript = YouTubeTranscriptApi.get_transcript(video_id)
                    
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
                    
                    full_text = response.choices[0].message.content
                    
                except Exception as translate_error:
                    logger.error(f"Failed to get and translate transcript: {translate_error}")
                    return {
                        'documents': [],
                        'document_id': None,
                        'status': 'error',
                        'message': f'Failed to get transcript for video {video_url}: {str(e)}, {str(translate_error)}'
                    }
            
            # Try to fetch video title from YouTube
            video_title = f"YouTube Video {video_id}"
            try:
                # This is a placeholder - in a real implementation, you might use YouTube API
                # to fetch the actual title, but that requires API keys and additional setup
                # For now, we'll extract a basic title from the URL
                if "title=" in video_url:
                    video_title = video_url.split("title=")[1].split("&")[0]
                    video_title = video_title.replace("+", " ")
                else:
                    # Try to get a cleaner video ID for display
                    video_title = f"YouTube Video: {video_id}"
            except Exception as title_error:
                logger.error(f"Error getting video title: {title_error}")
                # Use default title - no need to fail the whole process for this
            
            # Insert into Supabase yts table
            result = self.supabase.table('yts').insert({
                'space_id': space_id,
                'extracted_text': full_text,
                'yt_url': video_url,
                'thumbnail': thumbnail,
                # 'title': video_title
            }).execute()
            
            video_db_id = result.data[0]['id']
            
            # Store YouTube ID in the record for future reference
            try:
                self.supabase.table('yts').update({
                    'youtube_id': video_id
                }).eq('id', video_db_id).execute()
            except Exception as e:
                logger.warning(f"Could not update YouTube ID in record: {e}")
            
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
        Extract video ID from various YouTube URL formats
        """
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
        return None

    def _transcript_to_text(self, transcript: List[Dict[str, Any]]) -> str:
        """
        Convert YouTube transcript format to plain text
        """
        if not transcript:
            return ""
        
        # Sort transcript by start time to ensure proper ordering
        sorted_transcript = sorted(transcript, key=lambda x: x.get('start', 0))
        
        # Join all text segments with appropriate spacing
        full_text = " ".join([entry.get('text', '') for entry in sorted_transcript])
        
        return full_text 