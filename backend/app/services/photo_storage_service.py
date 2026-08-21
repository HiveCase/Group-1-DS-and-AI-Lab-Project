from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile

from app.core.config import get_settings


class PhotoStorageService:
    def __init__(self):
        self.upload_dir = get_settings().upload_dir
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    async def save(self, claim_id: str, photo: UploadFile) -> tuple[str, str, str]:
        claim_dir = self.upload_dir / claim_id
        claim_dir.mkdir(parents=True, exist_ok=True)
        suffix = Path(photo.filename or "photo.jpg").suffix or ".jpg"
        filename = f"{uuid4().hex}{suffix}"
        target = claim_dir / filename
        content = await photo.read()
        target.write_bytes(content)
        return str(target), photo.filename or filename, photo.content_type or "application/octet-stream"
