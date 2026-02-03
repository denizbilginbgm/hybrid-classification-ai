import os
import subprocess
import tempfile


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