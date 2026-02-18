import importlib
import os
import subprocess
import tempfile

from app.core import config


def word_bytes_to_pdf_bytes(word_bytes: bytes) -> bytes:
    """
    Converts Word document bytes (DOC/DOCX) into PDF bytes using LibreOffice in headless mode.
    :param word_bytes: Raw bytes of the Word document.
    :return: Converted PDF file as bytes.
    """
    with tempfile.TemporaryDirectory() as tmp:
        word_path = os.path.join(tmp, "input.doc")
        pdf_path = os.path.join(tmp, "input.pdf")

        with open(word_path, "wb") as f:
            f.write(word_bytes)

        subprocess.run(
            ["soffice", "--headless", "--convert-to", "pdf", word_path, "--outdir", tmp],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        with open(pdf_path, "rb") as f:
            return f.read()

def load_model_class(model_type: str):
    """
    Dynamically import and return a model class from MODEL_REGISTRY.
    This allows adding new models without changing the training logic —
    just register the class path in MODEL_REGISTRY and it will be picked up automatically.

    :param model_type: Model type key as defined in MODEL_REGISTRY (e.g. "xlm_roberta")
    :return: The model class itself, not an instance
    """
    module_path, class_name = config.MODEL_REGISTRY[model_type].rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)