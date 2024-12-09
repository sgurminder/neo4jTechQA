import os
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

from langchain_community.document_loaders import DirectoryLoader, TextLoader, JSONLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from langchain_community.graphs import Neo4jGraph
from langchain_openai import OpenAIEmbeddings
from neo4j import GraphDatabase
import jsonlines

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
URI = os.getenv('NEO4J_URI')
USERNAME = os.getenv('NEO4J_USERNAME')
PASSWORD = os.getenv('NEO4J_PASSWORD')
AUTH = ("USERNAME","PASSWORD")

        

#driver = GraphDatabase.driver(URI,auth=AUTH)
kg = Neo4jGraph(url=URI, username=USERNAME, password=PASSWORD, database='neo4j')

FILE_PATH="/Users/gurmindersingh/knowledge_graph/techqa/train.jsonl"

llm = OpenAI()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)



def get_embedding(llm,text):
    response = llm.embeddings.create(
        input=text,
        model = "text-embedding-ada-002"
    )
    return response.data[0].embedding

def question_answer_doc_tx(question, q_embedding, answer, a_embedding, chunks,id):
    
    kg.query("""
    MERGE (q:Question {id: $id})
    ON CREATE SET q.text= $q, q.embedding = $q_embedding
    MERGE (a:Answer {id: $id})
    ON CREATE SET a.text = $a, a.embedding = $a_embedding
    MERGE (q)-[:ANSWERED_BY]->(a)
""",params = {"q_embedding":q_embedding, "a_embedding": a_embedding,"q": question, "a": answer, "id": id})

    for idx, chunk in enumerate(chunks):
        kg.query("""
        MATCH (a:Answer) WHERE a.id=$id
        MERGE (c:Chunk{id: $chunk_id})
        ON CREATE SET c.text = $chunk_content, c.embedding = $chunk_embedding
        MERGE (a)-[:HAS_CHUNK]->(c)
        """, params= { "chunk_id": f"{id}_chunk_{idx}","chunk_content": chunk['content'],"chunk_embedding": chunk['embedding'],"id":id})



def create_vectorIndex():
    kg.query("""
    CREATE VECTOR INDEX question_index IF NOT EXISTS
    FOR (q:Question) ON (q.embedding)
    OPTIONS { indexConfig: {
    `vector.dimensions`: 1536,
    `vector.similarity_function`: 'cosine'
    }}""")
    
    kg.query("""
    CREATE VECTOR INDEX answer_index IF NOT exists
    FOR (a:Answer) ON (a.embedding)
    OPTIONS { indexConfig: {
    `vector.dimensions`: 1536,
    `vector.similarity_function`: 'cosine'
    }}""")
    
    kg.query(""" 
    CREATE VECTOR INDEX chunk_index IF NOT exists
    FOR (c:Chunk) ON (c.embedding)
    OPTIONS { indexConfig: {
    `vector.dimensions`: 1536,
    `vector.similarity_function`: 'cosine'
    }}
    """)


with jsonlines.open(FILE_PATH) as reader:
    for obj in reader:
        id = obj.get("id")
        question = obj.get("question","")
        answer = obj.get("answer","")
        document_content = obj.get("document","")

        question_embedding = get_embedding(llm,question) 
        answer_embedding = get_embedding(llm,answer)

        chunks = []
        for chunk in text_splitter.split_text(document_content):
            chunks.append({
                "content": chunk,
                "embedding": get_embedding(llm,chunk)
            })
        question_answer_doc_tx(question,question_embedding,answer,answer_embedding,chunks,id)
    create_vectorIndex()


