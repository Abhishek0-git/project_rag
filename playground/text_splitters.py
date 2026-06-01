from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter, MarkdownTextSplitter, PythonCodeTextSplitter, Language
# from langchain_experimental.text_splitter import SemanticChunker
from langchain_ollama import OllamaEmbeddings, ChatOllama

with open("docs/Harry_Potter.txt", "r", encoding='utf-8') as file:
    text = file.read()

def print_chunks(chunks):
    for i in chunks:
        print("-"*100)
        print(f"\n {i.page_content}")
    print("-"*100)
    

#---CharacterTextSplitter---
def characterTextSplitter():
    text_splitter = CharacterTextSplitter(
        chunk_size=450, chunk_overlap=5, separator='', strip_whitespace=False
    )
    chunks = text_splitter.create_documents([text])
    print_chunks(chunks[:10])

#---RecursiveCharacterTextSplitter---
def recursiveCharacterTextSplitter():
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=450, chunk_overlap=10
    )
    chunks = text_splitter.create_documents([text])
    print_chunks(chunks[:10])
    
#---MarkdownTextSplitter---
def markdownTextSplitter():
    text_splitter = MarkdownTextSplitter(
        chunk_size=450, chunk_overlap=10
    )
    chunks = text_splitter.create_documents([text])
    print_chunks(chunks[:10])

#---PythonCodeTextSplitter---
def pythonCodeTextSplitter():
    with open("docs/demo_python_project.txt", "r") as file:
        text = file.read()
    text_splitter = PythonCodeTextSplitter(
        chunk_size=450, chunk_overlap=10
    )
    chunks = text_splitter.create_documents([text])
    print_chunks(chunks[:10])

#---CodeTextSplitter---
#---RecursiveCharacterTextSplitter---
def language_splitter():
    with open("docs/demo_java_project.txt", "r") as file:
        text = file.read()
    text_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.JAVA, chunk_size=450, chunk_overlap=10
    )
    chunks = text_splitter.create_documents([text])
    print_chunks(chunks[20:30])    

# def semanticChunker():
#     text_splitter = SemanticChunker(
#         OllamaEmbeddings(model="bge-m3:latest"), breakpoint_threshold_type='percentile'
#     )
#     chunks = text_splitter.create_documents([text])
#     print_chunks(chunks[0:10])

def agentic_chunking():
    llm = ChatOllama(model="granite3.1-moe:1b")
    prompt = f"""
    You are a text chunking expert. Split this text into logical chunks.

    Rules:
    - Each chunk should be around 200 characters or less
    - Split at natural topic boundaries
    - Keep related information together
    - Put "<<<SPLIT>>>" between chunks

    Text:
    {text[:4000]}

    Return the text with <<<SPLIT>>> markers where you want to split:
    """
    response = llm.invoke(prompt)
    marked_text = response.content
    chunks = marked_text.split("<<<SPLIT>>>")
    clean_chunks = []
    for chunk in chunks:
        cleaned = chunk.strip()
        if cleaned:
            clean_chunks.append(cleaned)
    for i in clean_chunks[:10]:
        print(i)
        

# characterTextSplitter()
# recursiveCharacterTextSplitter()
# markdownTextSplitter()
# pythonCodeTextSplitter()
# language_splitter()
# semanticChunker()
# agentic_chunking()