from flask import request, render_template
from flask import Flask
import os
import sys

import openai
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma

import constants

os.environ["OPENAI_API_KEY"] = constants.APIKEY

# Enable to save to disk & reuse the model (for repeated queries on the same data)
PERSIST = False

query = None
if len(sys.argv) > 1:
    query = sys.argv[1]

if PERSIST and os.path.exists("persist"):
    print("Reusing index...\n")
    vectorstore = Chroma(persist_directory="persist",
                         embedding_function=OpenAIEmbeddings())
    index = VectorStoreIndexWrapper(vectorstore=vectorstore)
else:
    # Use this line if you only need data.txt
    loader = TextLoader("data/data.txt")
    # loader = DirectoryLoader("data/")
    if PERSIST:
        index = VectorstoreIndexCreator(
            vectorstore_kwargs={"persist_directory": "persist"}).from_loaders([loader])
    else:
        index = VectorstoreIndexCreator().from_loaders([loader])

chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(model="gpt-3.5-turbo"),
    retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}),
)

chat_history = []
while True:
    if not query:
        query = input("Prompt: Hn bhai How can I help you today ")
    if query in ['quit', 'q', 'exit']:
        sys.exit()
    result = chain({"question": query, "chat_history": chat_history})
    print(result['answer'])

    chat_history.append((query, result['answer']))
    query = None

app = Flask(__name__)
app.config["SECRET_KEY"] = "some-secret-key"
app.config["DEBUG"] = True


@app.route("/", methods=["GET", "POST"])
def index():
    # Initialize an empty query and chat history
    query = None
    chat_history = []
    # Check if the request method is POST
    if request.method == "POST":
        # Get the user input from the form
        query = request.form.get("query")
        # Pass the query and the chat history to the chain object, and get the result as a dictionary
        result = chain({"question": query, "chat_history": chat_history})
        # Get the answer from the result dictionary
        answer = result["answer"]
        # Append the query and the answer to the chat history list for context
        chat_history.append((query, answer))
    # Render the index.html template with the query and chat history variables
    return render_template("index.html", query=query, chat_history=chat_history)
