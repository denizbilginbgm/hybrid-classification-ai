import os
from datetime import datetime

import pandas as pd

from app.core import config
from app.services.datasets.dataset_creator import DatasetCreator


def create_dataset_for_hybrid_classification():
    start_time = datetime.now()

    dataset_creator = DatasetCreator()
    final_df = dataset_creator.create_n_document_text_dataset()

    os.makedirs(config.OUTPUTS_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M")
    row_count = len(final_df)

    file_name = f"{timestamp}_{row_count}_documents.parquet"
    output_path = os.path.join(config.OUTPUTS_DIR, file_name)

    final_df.to_parquet(output_path, engine="pyarrow")

    print(f"Saved to {output_path}")

    duration = datetime.now() - start_time
    days = duration.days
    hours, remainder = divmod(duration.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    print(
        f"Total runtime: {days} days {hours} hours {minutes} minutes {seconds} seconds."
    )

if __name__ == '__main__':
    #create_dataset_for_hybrid_classification()
    df = pd.read_parquet(os.path.join(config.OUTPUTS_DIR,"02-02-2026_17-38_5027_documents.parquet"))
    doc = df.loc[df["document_id"] == "EFCA02A1-506F-4C16-BFE3-08DE3D6A6987"]
    print(type(df["text"].iloc[0]))
    print(doc.iloc[0]["text"])
