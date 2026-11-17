"""HTTP service clients for the W2V Orchestrator."""

from src.clients.ai_content_generation_client import AIContentGenerationClient
from src.clients.app_inspector_client import AppInspectorClient
from src.clients.base import BaseServiceClient
from src.clients.bdd_playwright_client import BDDPlaywrightClient
from src.clients.brand_extractor_client import BrandExtractorClient
from src.clients.critique_client import CritiqueClient
from src.clients.design_client import DesignClient
from src.clients.frontend_test_runner_client import FrontendTestRunnerClient
from src.clients.media_discovery_client import MediaDiscoveryClient
from src.clients.slide_service_client import SlideServiceClient
from src.clients.transcription_worker_client import TranscriptionWorkerClient
from src.clients.tts_client import TTSClient
from src.clients.video_analyzer_client import VideoAnalyzerClient
from src.clients.video_compile import VideoCompileClient
from src.clients.video_production_client import VideoProductionClient
from src.clients.youtube_upload_client import YouTubeUploadClient

__all__ = [
    "AIContentGenerationClient",
    "AppInspectorClient",
    "BDDPlaywrightClient",
    "BaseServiceClient",
    "BrandExtractorClient",
    "CritiqueClient",
    "DesignClient",
    "FrontendTestRunnerClient",
    "MediaDiscoveryClient",
    "SlideServiceClient",
    "TTSClient",
    "TranscriptionWorkerClient",
    "VideoAnalyzerClient",
    "VideoCompileClient",
    "VideoProductionClient",
    "YouTubeUploadClient",
]
