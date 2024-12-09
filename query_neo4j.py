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
template="I need to move my perpetual license of SPSS v24 to my new machine. I have my authorization code first given to me, but since we did not renew our support contract we can't seem to download the software. I was originally told that I could download the trial version and it would have the authorization wizard where I could input the code and move forward. But the only downloadable version is subscription version with no authorization wizard. How do I move forward? "

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
