from app.core.config import get_settings
from app.services.damage_detection_service import DamageDetectionService


def test_resolve_default_model_path_uses_backend_models_dir():
    service = DamageDetectionService()
    resolved = service._resolve_default_model_path()
    assert resolved == get_settings().model_dir or resolved == get_settings().model_dir / "model.pt"
