import os

import pandas as pd
import pyodbc

from app.core import config


class DatabaseService:
    def __init__(self):
        self.__connection = pyodbc.connect(config.CONNECTION_STRING)

    def get_company_balanced_n_document(self) -> pd.DataFrame:
        """
        Retrieves a balanced set of documents per company and document type.

        The query limits the number of documents per company to distribute samples
        evenly across firms, and adds a separate fixed sample for penalty documents.
        :return: A result dataframe to train an AI system.
        """
        allowed_ids = ",".join(str(v) for v in config.ALLOWED_DOCUMENT_TYPES.values())

        query_template = self.__get_query_template("get_company_balanced_n_documents.sql")
        query = query_template.format(allowed_ids=allowed_ids, n_documents=config.N_DOCUMENTS)

        result = pd.read_sql(query, self.__connection)
        self.__connection.close()

        return result

    # -------------------------
    # Helper Functions
    # -------------------------
    def __get_query_template(self, query_filename: str) -> str:
        """
        Load and return the SQL query template from a file.

        :param query_filename: The name of the SQL query file to load.
        :return: The content of the SQL query file as a string.
        """
        query_template_path = os.path.join(config.BASE_SQL_QUERIES_PATH, query_filename)
        with open(query_template_path) as f:
            return f.read()
