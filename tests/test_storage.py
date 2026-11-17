"""Tests for storage service."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.services.storage_service import StorageService


@pytest.mark.asyncio
async def test_local_storage_lifecycle():
    """Full lifecycle: initialise, store, retrieve, list, delete artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("src.services.storage_service.get_settings") as mock_settings:
            mock_settings.return_value.storage.backend = "local"
            mock_settings.return_value.storage.local_path = tmpdir
            mock_settings.return_value.storage.minio_endpoint = ""
            mock_settings.return_value.storage.minio_access_key = ""
            mock_settings.return_value.storage.minio_secret_key = ""
            mock_settings.return_value.storage.minio_bucket = ""
            mock_settings.return_value.storage.minio_secure = False
            mock_settings.return_value.storage.presign_expiry_seconds = 3600

            service = StorageService()

            await service.initialize()

            # Store artifact
            path = await service.store_stage_artifact(
                "job-1",
                "analyze",
                b'{"test": true}',
            )
            assert path is not None

            # Retrieve
            data = await service.retrieve_stage_artifact("job-1", "analyze")
            assert data == b'{"test": true}'

            # List
            artifacts = await service.list_job_artifacts("job-1")
            assert len(artifacts) > 0

            # Health
            assert await service.check_health() is True

            # Delete
            assert await service.delete_job_artifacts("job-1") is True

            # Verify deletion
            data_after = await service.retrieve_stage_artifact("job-1", "analyze")
            assert data_after is None


@pytest.mark.asyncio
async def test_local_storage_multiple_stages():
    """Storing artifacts for different stages of the same job."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("src.services.storage_service.get_settings") as mock_settings:
            mock_settings.return_value.storage.backend = "local"
            mock_settings.return_value.storage.local_path = tmpdir
            mock_settings.return_value.storage.minio_endpoint = ""
            mock_settings.return_value.storage.minio_access_key = ""
            mock_settings.return_value.storage.minio_secret_key = ""
            mock_settings.return_value.storage.minio_bucket = ""
            mock_settings.return_value.storage.minio_secure = False
            mock_settings.return_value.storage.presign_expiry_seconds = 3600

            service = StorageService()
            await service.initialize()

            await service.store_stage_artifact("job-2", "crawl", b'{"pages": 5}')
            await service.store_stage_artifact("job-2", "analyze", b'{"score": 8}')

            artifacts = await service.list_job_artifacts("job-2")
            assert len(artifacts) == 2

            crawl_data = await service.retrieve_stage_artifact("job-2", "crawl")
            assert crawl_data == b'{"pages": 5}'

            analyze_data = await service.retrieve_stage_artifact("job-2", "analyze")
            assert analyze_data == b'{"score": 8}'


@pytest.mark.asyncio
async def test_local_storage_retrieve_nonexistent():
    """Retrieving a non-existent artifact returns None."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("src.services.storage_service.get_settings") as mock_settings:
            mock_settings.return_value.storage.backend = "local"
            mock_settings.return_value.storage.local_path = tmpdir
            mock_settings.return_value.storage.minio_endpoint = ""
            mock_settings.return_value.storage.minio_access_key = ""
            mock_settings.return_value.storage.minio_secret_key = ""
            mock_settings.return_value.storage.minio_bucket = ""
            mock_settings.return_value.storage.minio_secure = False
            mock_settings.return_value.storage.presign_expiry_seconds = 3600

            service = StorageService()
            await service.initialize()

            data = await service.retrieve_stage_artifact("ghost-job", "crawl")
            assert data is None


@pytest.mark.asyncio
async def test_local_storage_delete_nonexistent():
    """Deleting artifacts for a non-existent job returns False."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("src.services.storage_service.get_settings") as mock_settings:
            mock_settings.return_value.storage.backend = "local"
            mock_settings.return_value.storage.local_path = tmpdir
            mock_settings.return_value.storage.minio_endpoint = ""
            mock_settings.return_value.storage.minio_access_key = ""
            mock_settings.return_value.storage.minio_secret_key = ""
            mock_settings.return_value.storage.minio_bucket = ""
            mock_settings.return_value.storage.minio_secure = False
            mock_settings.return_value.storage.presign_expiry_seconds = 3600

            service = StorageService()
            await service.initialize()

            result = await service.delete_job_artifacts("nonexistent-job")
            assert result is False


@pytest.mark.asyncio
async def test_local_storage_list_empty():
    """Listing artifacts for a job with none returns empty list."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("src.services.storage_service.get_settings") as mock_settings:
            mock_settings.return_value.storage.backend = "local"
            mock_settings.return_value.storage.local_path = tmpdir
            mock_settings.return_value.storage.minio_endpoint = ""
            mock_settings.return_value.storage.minio_access_key = ""
            mock_settings.return_value.storage.minio_secret_key = ""
            mock_settings.return_value.storage.minio_bucket = ""
            mock_settings.return_value.storage.minio_secure = False
            mock_settings.return_value.storage.presign_expiry_seconds = 3600

            service = StorageService()
            await service.initialize()

            artifacts = await service.list_job_artifacts("empty-job")
            assert artifacts == []


@pytest.mark.asyncio
async def test_local_storage_store_final_video():
    """store_final_video creates a video file in the local backend."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("src.services.storage_service.get_settings") as mock_settings:
            mock_settings.return_value.storage.backend = "local"
            mock_settings.return_value.storage.local_path = tmpdir
            mock_settings.return_value.storage.minio_endpoint = ""
            mock_settings.return_value.storage.minio_access_key = ""
            mock_settings.return_value.storage.minio_secret_key = ""
            mock_settings.return_value.storage.minio_bucket = ""
            mock_settings.return_value.storage.minio_secure = False
            mock_settings.return_value.storage.presign_expiry_seconds = 3600

            service = StorageService()
            await service.initialize()

            path = await service.store_final_video("job-1", b"\x00\x00video-data")
            assert path is not None
            assert Path(path).exists()


@pytest.mark.asyncio
async def test_local_storage_custom_filename():
    """store with custom filename and retrieve by same name."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("src.services.storage_service.get_settings") as mock_settings:
            mock_settings.return_value.storage.backend = "local"
            mock_settings.return_value.storage.local_path = tmpdir
            mock_settings.return_value.storage.minio_endpoint = ""
            mock_settings.return_value.storage.minio_access_key = ""
            mock_settings.return_value.storage.minio_secret_key = ""
            mock_settings.return_value.storage.minio_bucket = ""
            mock_settings.return_value.storage.minio_secure = False
            mock_settings.return_value.storage.presign_expiry_seconds = 3600

            service = StorageService()
            await service.initialize()

            # Store with custom filename
            path = await service.store_stage_artifact(
                "job-custom",
                "analyze",
                b'{"custom": true}',
                filename="report.json",
            )
            assert "report.json" in path

            # Retrieve by the same custom filename
            data = await service.retrieve_stage_artifact(
                "job-custom",
                "analyze",
                filename="report.json",
            )
            assert data == b'{"custom": true}'

            # Default filename should NOT find it
            default_data = await service.retrieve_stage_artifact(
                "job-custom",
                "analyze",
            )
            assert default_data is None


@pytest.mark.asyncio
async def test_local_storage_presigned_url_returns_none():
    """get_presigned_url on local backend returns None."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("src.services.storage_service.get_settings") as mock_settings:
            mock_settings.return_value.storage.backend = "local"
            mock_settings.return_value.storage.local_path = tmpdir
            mock_settings.return_value.storage.minio_endpoint = ""
            mock_settings.return_value.storage.minio_access_key = ""
            mock_settings.return_value.storage.minio_secret_key = ""
            mock_settings.return_value.storage.minio_bucket = ""
            mock_settings.return_value.storage.minio_secure = False
            mock_settings.return_value.storage.presign_expiry_seconds = 3600

            service = StorageService()
            await service.initialize()

            url = await service.get_presigned_url("some/path")
            assert url is None
