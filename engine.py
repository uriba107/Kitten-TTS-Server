# File: engine.py
# Core TTS model loading and speech generation logic for KittenTTS.

import logging
import numpy as np
from typing import Optional, Tuple

from kittentts import KittenTTS

from config import config_manager

logger = logging.getLogger(__name__)

kittentts_model: Optional[KittenTTS] = None
MODEL_LOADED: bool = False

KITTEN_TTS_VOICES = [
    "Bella",
    "Jasper",
    "Luna",
    "Bruno",
    "Rosie",
    "Hugo",
    "Kiki",
    "Leo",
]


def load_model() -> bool:
    global kittentts_model, MODEL_LOADED

    if MODEL_LOADED:
        logger.info("KittenTTS model is already loaded.")
        return True

    try:
        model_repo_id = config_manager.get_string(
            "model.repo_id", "KittenML/kitten-tts-micro-0.8"
        )

        logger.info(f"Loading KittenTTS model from: {model_repo_id}")

        device = config_manager.get_string("tts_engine.device", "cpu")
        logger.info(f"Using device: {device}")

        kittentts_model = KittenTTS(model_repo_id, device=device)

        MODEL_LOADED = True
        logger.info("KittenTTS model loaded successfully.")
        return True

    except Exception as e:
        logger.error(f"Error loading KittenTTS model: {e}", exc_info=True)
        kittentts_model = None
        MODEL_LOADED = False
        return False


def synthesize(
    text: str, voice: str, speed: float = 1.0
) -> Tuple[Optional[np.ndarray], Optional[int]]:
    global kittentts_model

    if not MODEL_LOADED or kittentts_model is None:
        logger.error("KittenTTS model is not loaded. Cannot synthesize audio.")
        return None, None

    if voice not in KITTEN_TTS_VOICES:
        logger.error(
            f"Voice '{voice}' not available. Available voices: {KITTEN_TTS_VOICES}"
        )
        return None, None

    try:
        logger.debug(f"Synthesizing with voice='{voice}', speed={speed}")
        logger.debug(f"Input text (first 100 chars): '{text[:100]}...'")

        audio = kittentts_model.generate(text=text, voice=voice, speed=speed)

        sample_rate = 24000

        logger.info(
            f"Successfully generated {len(audio)} audio samples at {sample_rate}Hz"
        )
        return audio, sample_rate

    except Exception as e:
        logger.error(f"Error during KittenTTS synthesis: {e}", exc_info=True)
        return None, None
