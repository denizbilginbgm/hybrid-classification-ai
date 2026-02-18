from typing import Tuple

import pandas as pd
import torch
from PIL.Image import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from app.core import config
from app.services.database.database_service import DatabaseService
from app.services.datasets.document_downloader import DocumentDownloader


class DatasetCreator:
    def __init__(self):
        self.__prompt = "Free OCR."
        self.__model, self.__processor = self.__initialize_vlm()

        self.__database_service = DatabaseService()
        self.__document_downloader = DocumentDownloader()

    def __initialize_vlm(self) -> Tuple[Qwen3VLForConditionalGeneration, AutoProcessor]:
        """
        Initialize Qwen3 VL model
        :return: Qwen3VL Model, Processor
        """
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            config.VLM_MODEL_NAME,
            dtype=torch.float16,
            device_map="auto",
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            max_memory={0: config.VLM_MODEL_ALLOWED_MAX_GPU, "cpu": config.VLM_MODEL_ALLOWED_MAX_CPU},
        ).eval()

        processor = AutoProcessor.from_pretrained(config.VLM_MODEL_NAME)

        return model, processor

    def __extract_text(self, image: Image) -> str:
        """
        Extracts text from an image using a Vision Language Model (VLM).

        This function performs Optical Character Recognition (OCR) on the provided image
        using a pre-trained VLM model. It applies a chat template, processes the image
        with the given prompt, and generates text output using beam search decoding.
        :param image: PIL Image object to extract text from (e.g., from pdf2image's convert_from_bytes).
        :return: The extracted text from the image
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": self.__prompt}
                ]
            }
        ]

        text = self.__processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.__processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True
        )

        device = self.__model.device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = self.__model.generate(
                **inputs,
                max_new_tokens=config.VLM_MODEL_MAX_NEW_TOKENS,
                num_beams=config.VLM_MODEL_NUM_BEAMS,
                repetition_penalty=config.VLM_MODEL_REPETITION_PENALTY,
                no_repeat_ngram_size=config.VLM_MODEL_NO_REPEAT_NGRAM_SIZE,
                early_stopping=config.VLM_MODEL_EARLY_STOPPING
            )

        # Cleaning input prompt
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]

        extracted_text = self.__processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0].strip()

        return extracted_text

    def create_n_document_text_dataset(self) -> pd.DataFrame:
        """
        Runs OCR on a balanced set of documents and returns extracted page texts.

        Fetches document metadata, downloads each document, converts pages to images,
        applies OCR per page, and aggregates results into a DataFrame.
        :return: Extracted text with document, company, type, and page info.
        """
        print("OCR process is started.")
        docs_df = self.__database_service.get_company_balanced_n_document()
        self.__analyze_percentage_of_db_result(docs_df)

        rows = []

        for _, row in tqdm(docs_df.iterrows(), total=len(docs_df)):
            document_id = str(row["belgeID"])
            document_type = row["belgeTipi"]
            company_id = row["firmaID"]

            # PDF to Image List
            page_images = self.__document_downloader.get_document_image(document_id)

            if not page_images:
                continue

            for page_number, page_image in enumerate(page_images, start=1):
                try:
                    # Image to Text
                    page_text = self.__extract_text(page_image)

                    rows.append({
                        "document_id": document_id,
                        "company_id": company_id,
                        "document_type": document_type,
                        "page_number": page_number,
                        "text": page_text
                    })

                    print(f"Document {document_id}, page {page_number} processed successfully.")
                except Exception as e:
                    print(f"OCR failed. Document {document_id} page {page_number}: {e}")

        final_df = pd.DataFrame(rows)
        return final_df

    def __analyze_percentage_of_db_result(self, result_df: pd.DataFrame):
        """
        Calculates and prints the percentage distribution of documents by type.
        :param result_df: Query result containing document types.
        """
        type_totals = (
            result_df.groupby("belgeTipi")
            .size()
            .reset_index(name="count")
        )

        total = type_totals["count"].sum()
        type_totals["%"] = (type_totals["count"] / total * 100).round(2)
        type_totals = type_totals.sort_values("%", ascending=False)
        print(type_totals)