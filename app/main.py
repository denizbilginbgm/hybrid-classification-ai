import os
from datetime import datetime

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
    create_dataset_for_hybrid_classification()