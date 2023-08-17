from flask import Flask, render_template, request

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

app = Flask(__name__)

chat_history = []  # Store chat history globally
PERSIST = False

if PERSIST and os.path.exists("persist"):
    print("Reusing index...\n")
    vectorstore = Chroma(persist_directory="persist",
                         embedding_function=OpenAIEmbeddings())
    index = VectorStoreIndexWrapper(vectorstore=vectorstore)
else:
    # Use this line if you only need data.txt
    loader = DirectoryLoader("data/")
    if PERSIST:
        index = VectorstoreIndexCreator(
            vectorstore_kwargs={"persist_directory": "persist"}).from_loaders([loader])
    else:
        index = VectorstoreIndexCreator().from_loaders([loader])

chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(model="gpt-3.5-turbo"),
    retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}),
)


@app.route("/", methods=["GET", "POST"])
def index():
    query = None

    if request.method == "POST":
        query = request.form.get("query")
        result = chain({"question": query, "chat_history": chat_history})
        answer = result["answer"]
        chat_history.append((query, answer))  # Append to global chat history

    return render_template("index.html", query=query, chat_history=chat_history)


if __name__ == "__main__":
    chat_history = []  # Empty chat history when re-running the application
    app.run()
