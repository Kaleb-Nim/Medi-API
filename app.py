from time import time
from fastapi import FastAPI, __version__
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import openai 
from dotenv import load_dotenv
import pinecone
import openai
import os
import json
from time import time 
import tqdm
# Typing
from typing import List, Dict, Any, Optional, Union, Tuple

from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone.index import Index
from pinecone.index import UpsertResponse
from langchain.chains import LLMChain
from llm.chains import output_chain
from llm.Orchestrator import Orchestrator
# Data processing stuff
import pandas as pd
from PineconeUtils.Queryer import PineconeQuery
from PineconeUtils.Indexer import Indexer,DataEmbedding
load_dotenv()
PINECONE_API_KEY,PINECONE_ENVIRONMENT,INDEX_NAME = os.getenv("PINECONE_API_KEY"),os.getenv("PINECONE_ENVIRONMENT"),os.getenv("PINECONE_INDEX_NAME")
openai.api_key = os.getenv("OPENAI_API_KEY")

pineconeQuery = PineconeQuery(PINECONE_API_KEY,PINECONE_ENVIRONMENT,INDEX_NAME)
orchestrator = Orchestrator(pineconeQuery,chain=output_chain)


from flask import Flask, render_template, request, jsonify
import time

app = Flask(__name__)

html = """
<!DOCTYPE html>
<html>
    <head>
        <title>FastAPI on Vercel</title>
        <link rel="icon" href="/static/favicon.ico" type="image/x-icon" />
    </head>
    <body>
        <div class="bg-gray-200 p-4 rounded-lg shadow-lg">
            <h1>Hello from FastAPI@{__version__}</h1>
            <ul>
                <li><a href="/docs">/docs</a></li>
                <li><a href="/redoc">/redoc</a></li>
            </ul>
            <p>Powered by <a href="https://vercel.com" target="_blank">Vercel</a></p>
        </div>
    </body>
</html>
"""

@app.route("/")
def root():
    return render_template(html, version=__version__)

@app.route("/ping", methods=["GET"])
def hello():
    return jsonify({'res': 'pong', 'version': __version__, "time": time.time()})

@app.route("/test", methods=["POST"])
def query2():

    return jsonify({'res': 'post testing', 'version': __version__, "time": time.time()})


@app.route("/ping", methods=["POST"])
def query():
    # Get user input from the query parameter
    user_question = request.args.get("user_question")

    if user_question is None:
        return "Invalid input", 400

    question_json = orchestrator.findRelevantQuestion(user_question)
    return jsonify({'res': 'post testing', 'version': __version__, "time": time.time(), 'output': question_json})

if __name__ == "__main__":
    app.run(debug=True)
