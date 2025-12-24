#from google.colab import userdata
import os
# api_key=userdata.get('GROQ_API_KEY')
#os.environ["GROQ_API_KEY"] = userdata.get('GROQ_KEY')

from dotenv import load_dotenv
load_dotenv()
api_key=(os.getenv("GROQ_API_KEY"))

from langchain.chat_models import init_chat_model
llm=init_chat_model(model="llama-3.3-70b-versatile",model_provider='groq', temperature=0,max_tokens=300)#gemma-2-9b-it

prompt="Hello, world!"
response=llm.invoke(prompt)
print(response.content)