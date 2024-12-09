import os
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
from langchain_community.graphs import Neo4jGraph
from langchain.prompts import PromptTemplate
from langchain.chains import GraphCypherQAChain, RetrievalQA
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Neo4jVector
from langchain.schema import BaseRetriever, Document
from pydantic import Field 
from typing import List


llm = ChatOpenAI(api_key=os.getenv('OPENAI_API_KEY'))

embedding_provider = OpenAIEmbeddings(api_key=os.getenv('OPENAI_API_KEY'),model = "text-embedding-ada-002")

graph = Neo4jGraph(
    url=os.getenv('NEO4J_URI'),
    username=os.getenv('NEO4J_USERNAME'),
    password=os.getenv('NEO4J_PASSWORD')
) 


def search_question_answer_chunks(query, top_k=10):
    query_vector = embedding_provider.embed_query(query)

    cypher_query="""
     MATCH (q:Question)-[:ANSWERED_BY]->(a:Answer)
    WITH q,a,
    vector.similarity.cosine(q.embedding, $query_vector) AS question_similarity
    WHERE question_similarity > 0.5
    ORDER BY question_similarity DESC
    LIMIT $top_k
    WITH collect({question: q.text, answer: a.text}) AS similar_question_and_answers

    MATCH (a:Answer)-[:ANSWERED_BY]-(q:Question)
    WITH a,q,
    vector.similarity.cosine(a.embedding, $query_vector) AS answer_similarity, similar_question_and_answers
    WHERE answer_similarity > 0.5
    ORDER BY answer_similarity DESC
    LIMIT $top_k
    WITH similar_question_and_answers, collect({question: q.text, answer: a.text}) AS simiar_answer_questions
    WITH similar_question_and_answers + simiar_answer_questions AS qa_pairs

    MATCH (c:Chunk)
    WITH c,
    vector.similarity.cosine(c.embedding, $query_vector) AS chunk_similarity, qa_pairs
    WHERE chunk_similarity > 0.5
    ORDER BY chunk_similarity DESC
    LIMIT $top_k
    RETURN qa_pairs, collect(c.text) AS similar_chunks

"""

    result = graph.query(cypher_query, {"query_vector": query_vector, "top_k": top_k})
    qa_pairs = result[0].get("qa_pairs",[])
    similar_chunks = result[0].get("similar_chunks",[])
    
    context = "\n\n".join(
        [
            f"Q: {qa['question']}\nA: {qa['answer']}"
            for qa in qa_pairs
        ]
    )
    chunk_context = "\n\n".join(similar_chunks)
    full_context = f"{context}\n\n{chunk_context}"

    return full_context

class Neo4jQARetriever(BaseRetriever):
    top_k: int = Field(default=5, description="Number of top documents to retrieve")

    def __init__(self, top_k=10):
        super().__init__()
        self.top_k = top_k

    def get_relevant_documents(self,query):
        full_context = search_question_answer_chunks(query,self.top_k)

        documents = [
            Document(
                page_content = full_context,
                metadata={}
            )
        ]
        return documents

    async def aget_relevant_documents(self, query:str) -> List[Document]:
        return self.get_relevant_documents(query)



techqa_retriever = Neo4jQARetriever(top_k=10)


class RetrievalWithContext:
    def __init__(self, techqa_retriever, llm, prompt_template):
        self.retriever = techqa_retriever
        self.llm = llm
        self.prompt_template = prompt_template

    def run(self, query: str):
        # Retrieve context
        retrieved_docs = self.retriever.get_relevant_documents(query)
        context = retrieved_docs[0].page_content if retrieved_docs else "No relevant context found."

        # Build the prompt
        prompt = self.prompt_template.format(context=context, query=query)

        # Query the LLM
        return self.llm(prompt)


prompt_template = PromptTemplate(
    input_variables=["context", "query"],
    template=(
        "You are expert in providing solutions to IBM customers for various technical products. Use the context provided from IBM's tech forums and documents  below to answer the user's query.\n\n"
        "Context:\n{context}\n\n"
        "Query:\n{query}"
    )
)



qa_chain = RetrievalWithContext(techqa_retriever,llm,prompt_template)

query = "How to proceed when jextract utility is throwing an OutOfMemory error ? \n\nI was getting a crash and was told to provide a \”jextracted core dump\”. I ran the jextract command as instructed:\n\n/java/jre/bin/jextract [CORE_PATH]\n\nbut I am getting now an OutOfMemory error when jextract itself is running so I cannot proceed with the original crash investigation"


response = qa_chain.run(query)

print(response)

