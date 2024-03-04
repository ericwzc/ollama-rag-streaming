from chromadb.config import Settings

# Define the folder for storing database
PERSIST_DIRECTORY = 'db'
SOURCE_DIRECTORY = 'source_documents'
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
RETRIEVED_DOC_NUMBER = 4

LAPTOP = True
LAPTOP_EMBEDDING_MODEL_NAME = 'C:\\Users\\ERICWAN5\\.cache\\torch\\sentence_transformers\\hkunlp_instructor-large'

# Define the Chroma settings
CHROMA_SETTINGS = Settings(
        persist_directory=PERSIST_DIRECTORY,
        anonymized_telemetry=False
)

LLM_MODEL = 'gemma:2b'
