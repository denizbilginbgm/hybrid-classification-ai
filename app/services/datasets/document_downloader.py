from typing import List, Optional

import requests
from PIL.Image import Image
from pdf2image import convert_from_bytes

from app.core import config
from app.services.datasets.image_preprocessor import ImagePreprocessor
from app.utils.utils import word_bytes_to_pdf_bytes


class DocumentDownloader:
    def __init__(self):
        self.__doc_download_endpoint = "https://panel.birlesikgumrukleme.com.tr/api/AntrepoStok/git/evrak/"
        self.__login_endpoint = "https://panel.birlesikgumrukleme.com.tr/api/account/login"

        self.__image_preprocessor = ImagePreprocessor()

        self.__jwt = None
        self.__request_header = None

    def update_jwt(self):
        payload = {
            "email": "dbilgin@birlesikgumrukleme.com.tr",
            "password": "3TAICc"
        }

        headers = {
            "Content-Type": "application/json"
        }

        response = requests.post(self.__login_endpoint, json=payload, headers=headers)
        if not response.ok:
            print("Login failed:", response.status_code, response.text)
            return False

        data = response.json()
        self.__jwt = data["token"]

        self.__request_header = {
            "Authorization": f"Bearer {self.__jwt}"
        }
        print("JWT updated")

        return True

    def get_document_image(self, document_id: str) -> Optional[List[Image]]:
        """
        Downloads a document and converts its pages into processed images.
        Supports PDF files directly and Word documents (DOC/DOCX) via PDF conversion.

        :param document_id: ID of the document to download.
        :return: List of processed page images, or None if conversion fails.
        """
        if not self.__jwt:
            if not self.update_jwt():
                return None

        url = self.__doc_download_endpoint + document_id
        response = requests.get(url, headers=self.__request_header)

        if response.status_code == 401:
            if self.update_jwt():
                response = requests.get(url, headers=self.__request_header)
            else:
                return None

        if not response.ok:
            print(response.status_code, response.text)
            return None


        content_type = response.headers.get("Content-Type", "").lower()
        document_bytes = response.content
        try:
            if "pdf" in content_type:
                page_images = convert_from_bytes(
                    document_bytes,
                    dpi=config.READING_DPI,
                    poppler_path=config.POPPLER_PATH
                )
            elif "msword" in content_type or "officedocument" in content_type:
                pdf_bytes = word_bytes_to_pdf_bytes(document_bytes)

                page_images = convert_from_bytes(
                    pdf_bytes,
                    dpi=config.READING_DPI,
                    poppler_path=config.POPPLER_PATH
                )
            else:
                print("Unsupported content type:", content_type)
                return None
            return self.__image_preprocessor.process_images(page_images)
        except Exception as e:
            print(f"Conversion failed for document {document_id}: {e}")
            return None