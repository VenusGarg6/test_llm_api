#SEMANTIC SEARCH WITH PINECONE VECTOR DATABASE

# Step 1: Create the Document(Store it in files folder) 
# Step 2: Create the API Key(here, pinecone API key, store it in Keys folder) 
# Step 3: Create an index

import os
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv('PINECONE_API_KEY')

from pinecone import Pinecone
api_key = os.environ.get("PINECONE_API_KEY")

# configure the pinecone client
pc = Pinecone(api_key=api_key)

# Define the cloud provider and region where we want to deploy our index.
from pinecone import ServerlessSpec
cloud = os.environ.get('PINECONE_CLOUD') or 'aws'
region = os.environ.get('PINECONE_REGION') or 'us-east-1'
spec = ServerlessSpec(cloud=cloud, region=region)

# Now, we create a new index called semantic-search-fast. 
# It's important that we align the index dimention and metrics parameters with those required by the MiniLM-L6 model.

index_name = 'semantic-search-obama-text-dec2025'

import time

existing_indexes = [
    index_info["name"] for index_info in pc.list_indexes()
]

# check if index already exists (it shouldn't if this is first time)
if index_name not in existing_indexes:
    # if does not exist, create index
    pc.create_index(
        index_name,
        dimension=1536,  # dimensionality of minilm(dimension of embedding model)
        metric='cosine',
        spec=spec
    )
    # wait for index to be initialized
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)

# connect to index
index = pc.Index(index_name)
time.sleep(1)
# view index stats
index.describe_index_stats()

#4. Data Preparation

from langchain_community.document_loaders import TextLoader,PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter

#5. Chunking
# Load the document, split it into chunks, embed each chunk and load it into the vector store.
raw_documents = TextLoader('content/sotu_address_obama.txt', encoding='utf-8').load()    #   Doc ai Models advanced
text_splitter = CharacterTextSplitter(chunk_size=600,separator='\n',chunk_overlap=100)
chunks = text_splitter.split_documents(raw_documents)

#Create the LLM API Key(here, using OpenAI LLM) and selecting the model, here it is 'text-embedding-ada-002'.
openai_api_key = os.getenv('OPENAI_API_KEY')
embeddings = OpenAIEmbeddings(model='text-embedding-ada-002')

#7.Storing the chunks and their respective embeddings in a new JSON Document.
from importlib import metadata
docs=[]
docs_json={}
id=0
for chunk in chunks:
  id+=1
  values=embeddings.embed_query(chunk.page_content) #got the embeddings using the embedding model
  docs_json={
      'id':str(id),
      'values':values,
      'metadata':{
      'text':chunk.page_content,
      'source':chunk.metadata['source'],
      'author':'Venus',
      'createdON':'09-10-2025'
      }
  }
  docs.append(docs_json)

  #8. INSERT DATA INTO INDEX
  # prompt: give me a code to process python list of batch size of 10
batch_size = 10
for i in range(0, len(docs), batch_size):
    batch = docs[i:i + batch_size]
    print(f"upserting batch {i} to {i + batch_size}")
    print(batch)
    index.upsert(batch)

    #9. Making Queries - Semantic Search.
    query = "can you summarize what obama said about schools?"

# create the query vector
#xq = model.encode(query).tolist()
xq = embeddings.embed_query(query)
# now query
xc = index.query(vector=xq, top_k=5, include_metadata=True)
#Reformat the data to be easier to read:
for result in xc['matches']:
    print(f"{round(result['score'], 2)}: {result['metadata']['text']}")

    query = "what does it mean by metropolies"
# create the query vector
#xq = model.encode(query).tolist()
xq = embeddings.embed_query(query)

# now query
xc = index.query(vector=xq, top_k=3, include_metadata=True)
for result in xc['matches']:
    print(f"{round(result['score'], 2)}: {result['metadata']['text']}")

from langchain_groq import ChatGroq
groq_api_key = os.getenv('GROQ_API_KEY')
llm=ChatGroq(model='llama-3.3-70b-versatile',temperature=0.5)
context=''
for result in xc['matches']:
    context=context + ' '+ str({result['metadata']['text']})
    prompt='''
can you summarize the context in 10 bullet points based on the context:{context}
'''.format(context=context,question=query)
    
response=llm.invoke(prompt)
print(response.content)