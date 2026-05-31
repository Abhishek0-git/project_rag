import ingestion
import retrival

def ingestion_pipeline(file):
    documents = ingestion.load_documents(file)
    chunks = ingestion.split_documents(documents)
    ingestion.create_vector_store(chunks)
        
def retrival_pipeline(query):
    return retrival.retrival_pipeline(query)

