import os
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import DirectoryLoader, TextLoader, JSONLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from langchain_openai import OpenAIEmbeddings

FILE_PATH="/Users/gurmindersingh/knowledge_graph/techqa/train.jsonl"

loader = JSONLoader(
    file_path = FILE_PATH,
    jq_schema='.document,.question,.answer',
    json_lines = True
)
docs = loader.load()

# Create a text splitter
text_splitter = CharacterTextSplitter(separator='\n\n',chunk_size=1500,chunk_overlap=200)

# Split documents into chunks
chunks = text_splitter.split_documents(docs)

# Create a Neo4j vector store
neo4j_db = Neo4jVector.from_documents(
    chunks,
    OpenAIEmbeddings(openai_api_key=os.getenv('OPENAI_API_KEY')),
    url=os.getenv('NEO4J_URI'),
    username=os.getenv('NEO4J_USERNAME'),
    password=os.getenv('NEO4J_PASSWORD'),
    database='neo4j',
    index_name='techqaVector',
    node_label='techqa',
    text_node_property='text',
    embedding_node_property='embedding',
)

