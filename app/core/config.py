# IMAGE PREPROCESSING
READING_DPI = 200
POPPLER_PATH = "app/utils/poppler-24.08.0/Library/bin"
ENABLE_ROTATION = True
ENABLE_DESKEW = True
ENABLE_BINARIZATION = True
BINARIZATION_METHOD = "adaptive"    # 'adaptive' or 'otsu'


# VLM
VLM_MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct"
VLM_MODEL_ALLOWED_MAX_GPU = "8GB"
VLM_MODEL_ALLOWED_MAX_CPU = "32GB"
VLM_MODEL_MAX_NEW_TOKENS = 1024
VLM_MODEL_NUM_BEAMS = 1
VLM_MODEL_REPETITION_PENALTY = 1.1
VLM_MODEL_NO_REPEAT_NGRAM_SIZE = 0
VLM_MODEL_EARLY_STOPPING = False


# DATABASE
DB_USERNAME = "BGM_Bigu"
DB_IP = "192.168.1.9"
DB_NAME = "BGM.PORTAL"
DB_PASSWORD = "Bigu+123"
CONNECTION_STRING = (
    "DRIVER={ODBC Driver 18 for SQL Server};"
    f"SERVER={DB_IP};"
    f"DATABASE={DB_NAME};"
    f"UID={DB_USERNAME};"
    f"PWD={DB_PASSWORD};"
    "TrustServerCertificate=yes;"
)
BASE_SQL_QUERIES_PATH = "app/services/database/sql/"

ALLOWED_DOCUMENT_TYPES = {
    "Menşe Şehadetnamesi": 20,
    #"Fatura": 18,
    #"A.TR Dolaşım Belgesi": 16,
    #"Çeki Listesi": 42,
    #"Konşimento": 23,
    #"EUR.1 Dolaşım Belgesi": 17,
    #"EUR-MED Dolaşım Belgesi": 197,
    #"Ceza Kararı": 13,
}


# TEXT EXTRACTOR
N_DOCUMENTS = 1


# SYSTEM
OUTPUTS_DIR = "outputs/"


# TRAINING
CHECKPOINTS_DIR = "checkpoints/"
MODEL_REGISTRY = {      # Model registry — add new models here as you implement them
    "xlm_roberta": "app.services.models.xlm_roberta.XlmRobertaTextClassifier",
}
