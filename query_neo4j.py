import os
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
from langchain_community.graphs import Neo4jGraph
from langchain.prompts import PromptTemplate
from langchain.chains import  RetrievalQA
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Neo4jVector

 
llm = ChatOpenAI(api_key=os.getenv('OPENAI_API_KEY'))

embedding_provider = OpenAIEmbeddings(api_key=os.getenv('OPENAI_API_KEY'),model = "text-embedding-ada-002")

# Connect to Neo4j
graph = Neo4jGraph(
    url=os.getenv('NEO4J_URI'),
    username=os.getenv('NEO4J_USERNAME'),
    password=os.getenv('NEO4J_PASSWORD')
) 

#Sample question from techqa validation.jsonl file
template="I receive the following during UDX compilation Can't exec \"/nz/kit/bin/adm/nzudxcompile\": Argument list too long "

prompt = PromptTemplate(input_variables=["question"],template=template)

techqa_vector = Neo4jVector.from_existing_index(
    embedding_provider,
    graph=graph,
    index_name='techqaVector',
    embedding_node_property="embedding",
    text_node_property="text"
)

techqa_retriever = RetrievalQA.from_llm(
    llm=llm,
    retriever=techqa_vector.as_retriever(),
    verbose=True,        #Make it false if no need details
    return_source_documents=True   #Make it false if don't need source docs
)

response = techqa_retriever.invoke(
{"query":prompt.format(question=question) }
)

print(response)
