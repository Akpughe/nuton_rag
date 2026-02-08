"""
Pydantic models for Course Generation POC.
Following KISS principle - simple, clear models with validation.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal
from datetime import datetime
from enum import Enum


# Enums for type safety and validation
class FormatPreference(str, Enum):
    READING = "reading"
    LISTENING = "listening"
    TESTING = "testing"
    MIXED = "mixed"


class DepthPreference(str, Enum):
    QUICK = "quick"
    DETAILED = "detailed"
    CONVERSATIONAL = "conversational"
    ACADEMIC = "academic"


class UserRole(str, Enum):
    STUDENT = "student"
    PROFESSIONAL = "professional"
    GRADUATE_STUDENT = "graduate_student"


class LearningGoal(str, Enum):
    EXAMS = "exams"
    CAREER = "career"
    CURIOSITY = "curiosity"
    SUPPLEMENT = "supplement"


class ExamplePreference(str, Enum):
    REAL_WORLD = "real_world"
    TECHNICAL = "technical"
    STORIES = "stories"
    ANALOGIES = "analogies"


class ExpertiseLevel(str, Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class CourseStatus(str, Enum):
    GENERATING = "generating"
    READY = "ready"
    ERROR = "error"


class SourceType(str, Enum):
    TOPIC = "topic"
    FILES = "files"
    YOUTUBE = "youtube"
    WEB = "web"
    MIXED = "mixed"


class OrganizationType(str, Enum):
    THEMATIC_BRIDGE = "thematic_bridge"
    SEQUENTIAL_SECTIONS = "sequential_sections"
    SEPARATE_COURSES = "separate_courses"


class ModelProvider(str, Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GROQ = "groq"


# Learning Profile Models
class LearningProfile(BaseModel):
    """User learning preferences - 5 questions from PRD"""
    user_id: str
    format_pref: FormatPreference
    depth_pref: DepthPreference
    role: UserRole
    learning_goal: LearningGoal
    example_pref: ExamplePreference
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class LearningProfileRequest(BaseModel):
    """Request to create/update learning profile"""
    user_id: str
    format_pref: FormatPreference
    depth_pref: DepthPreference
    role: UserRole
    learning_goal: LearningGoal
    example_pref: ExamplePreference


# Course Outline Models
class ChapterOutline(BaseModel):
    """Lightweight chapter info for outline"""
    order: int = Field(..., ge=1, le=10)
    title: str
    objectives: List[str] = Field(..., min_length=1, max_length=4)
    key_concepts: List[str] = Field(..., min_length=1, max_length=5)
    estimated_time: int = Field(..., ge=5, le=60)
    prerequisites: List[str] = []


class CourseOutline(BaseModel):
    """Generated course outline"""
    title: str
    description: str
    learning_objectives: List[str]
    chapters: List[ChapterOutline]
    total_estimated_time: int


# Full Chapter Models
class QuizQuestion(BaseModel):
    """Individual quiz question"""
    id: str
    type: Literal["multiple_choice", "true_false"]
    question: str
    options: List[str]
    correct_answer: int  # Index of correct option
    explanation: str


class Source(BaseModel):
    """Citation source"""
    number: int
    title: str
    url: str
    date: Optional[str] = None
    source_type: Literal["academic", "news", "documentation", "book", "other"]
    relevance: str


class Chapter(BaseModel):
    """Full chapter with content"""
    id: str
    course_id: str
    order: int
    title: str
    learning_objectives: List[str]
    content: str  # Markdown
    content_format: Literal["markdown"] = "markdown"
    estimated_time: int
    key_concepts: List[str]
    sources: List[Source]
    quiz: Dict[str, Any]  # Contains questions list
    status: Literal["pending", "ready", "error"] = "pending"
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    word_count: int = 0


class SourceFile(BaseModel):
    """Reference to uploaded source file"""
    file_id: str
    filename: str
    extracted_topic: str


class PersonalizationParams(BaseModel):
    """Stored preferences used for generation"""
    format_pref: FormatPreference
    depth_pref: DepthPreference
    role: UserRole
    learning_goal: LearningGoal
    example_pref: ExamplePreference


class Course(BaseModel):
    """Full course model"""
    id: str
    user_id: str
    title: str
    description: str
    topic: str
    source_type: SourceType
    source_files: List[SourceFile] = []
    multi_file_organization: Optional[OrganizationType] = None
    total_chapters: int
    estimated_time: int
    status: CourseStatus
    personalization_params: PersonalizationParams
    outline: CourseOutline
    model_used: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None


# Progress Models
class QuizAttempt(BaseModel):
    """Single quiz attempt"""
    attempt_id: int
    score: int = Field(..., ge=0, le=100)
    answers: List[str]
    completed_at: datetime


class ChapterProgress(BaseModel):
    """Progress for individual chapter"""
    chapter_id: str
    completed: bool = False
    completed_at: Optional[datetime] = None
    quiz_attempts: List[QuizAttempt] = []
    time_spent_minutes: int = 0


class CourseProgress(BaseModel):
    """Overall course progress"""
    user_id: str
    course_id: str
    chapter_progress: List[ChapterProgress]
    overall_progress: Dict[str, Any]


# Request Models
class CourseFromTopicRequest(BaseModel):
    """Generate course from topic string"""
    user_id: str
    topic: str = Field(..., min_length=10, max_length=500)
    context: Dict[str, Any] = Field(default_factory=dict)
    model: Optional[str] = None  # claude-sonnet-4, gpt-4o, llama-4-scout


class CourseFromFilesRequest(BaseModel):
    """Generate course from uploaded files"""
    user_id: str
    organization: Literal["auto", "thematic_bridge", "sequential_sections", "separate_courses"] = "auto"
    model: Optional[str] = None


class ProgressUpdateRequest(BaseModel):
    """Update chapter progress"""
    chapter_id: str
    completed: bool
    quiz_score: Optional[int] = None
    time_spent_minutes: Optional[int] = None


# Response Models
class CourseGenerationResponse(BaseModel):
    """Response after course generation"""
    course_id: str
    status: CourseStatus
    course: Course
    storage_path: Optional[str] = None
    generation_time_seconds: float


class SingleCourseInfo(BaseModel):
    """Summary info for one course in a multi-course response"""
    id: str
    title: str
    topic: str
    status: CourseStatus
    total_chapters: int
    estimated_time: int
    storage_path: Optional[str] = None


class MultiFileAnalysis(BaseModel):
    """Analysis of multiple uploaded files"""
    topics: List[Dict[str, str]]
    organization: OrganizationType
    reason: str
    similarity: Optional[float] = None
    options: Optional[List[OrganizationType]] = None


# Model Configuration
class ModelConfig(BaseModel):
    """Configuration for AI model"""
    provider: ModelProvider
    model: str
    max_tokens: int
    cost_per_1k_input: float
    cost_per_1k_output: float
    supports_search: bool
