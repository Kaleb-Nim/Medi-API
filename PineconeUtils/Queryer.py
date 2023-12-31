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
# Data processing stuff
import pandas as pd

from PineconeUtils.Indexer import Indexer,DataEmbedding

class PineconeQuery:
    """Main class to query both text/sentence/images with pinecone"""
    embedding_model:OpenAIEmbeddings = OpenAIEmbeddings(model='text-embedding-ada-002')
    NAMESPACES = ["personal","experience","projects","thoughts"]

    def _initPinecone(self,PINECONE_API_KEY:str,PINECONE_ENVIRONMENT:str,INDEX_NAME:str) -> pinecone.index:
        """Init Pinecone stuff"""

        pinecone.init(api_key=PINECONE_API_KEY,environment=PINECONE_ENVIRONMENT)
        # connect to index
        index:Index = pinecone.Index(INDEX_NAME)
        return index

    def __init__(self,PINECONE_API_KEY:str,PINECONE_ENVIRONMENT:str,INDEX_NAME:str):
        """ Initialize the Pinecone Index from .env file pinecone variables"""
        start_time = time()
        
        # connect to index
        self.index:Index = self._initPinecone(PINECONE_API_KEY,PINECONE_ENVIRONMENT,INDEX_NAME)
        self.embedding_model:OpenAIEmbeddings = self.embedding_model
        self.namespaces:list[str]  = ["personal","experience","projects","thoughts"]
        # Init the namespaces docsearch
        # self.docsearch = Pinecone.from_existing_index(INDEX_NAME, embedding_model) # default namespace
        # self.docsearch_personal = Pinecone.from_existing_index(INDEX_NAME, embedding_model, namespace="personal")
        # self.docsearch_experience = Pinecone.from_existing_index(INDEX_NAME, embedding_model, namespace="experience")
        # self.docsearch_projects = Pinecone.from_existing_index(INDEX_NAME, embedding_model, namespace="projects")
        # self.docsearch_thoughts = Pinecone.from_existing_index(INDEX_NAME, embedding_model, namespace="thoughts")
        print(f'Successfully connected to Pinecone Index:\n{self.index.describe_index_stats()},took {time() - start_time} seconds')


    def _checkValidNamespace(self,namespace:str) -> bool:
        # Check if namespace is valid
        if namespace not in self.namespaces:
            raise Warning(f"Namespace not found, must be one of the following {self.namespaces}. using default namespace= None")
        
        return True
    


    def query(self,query:str,namespace:str=None,top_k:int = 3) -> list[dict]:
        """Select a query and fetch the results

        Raises ValueError if namespace is not one of the following:
            ValueError: Namespace must be one of the following ['personal', 'experience', 'projects', 'thoughts']

        Args:
            query (str): Query to search
            namespace (str): Namespace to search can be one of the following
                >personal experience, projects, thoughts

        Returns:
            list[dict]: List of matched documents, Top 3 relevant documents
            Example matched_docs[
                {'id': '4',
  'score': 0.737784088,
  'values': [],
  'metadata': {'categories': 'personal',
   'isImage': False,
   'text': 'WTH ( Finalist ) Activities: NUS LifeHack 2021 ( Participant ) SUTD What The Hack: Environment 2021 ( Participant ) Appetizer Hackathon 2021 ( Participant ) SPAI Beginner Machine Learning Bootcamp 2021 SPAI Advance Machine Learning Workshop 2021 SEED Code League 2021 ( Participant ) NUS LifeHack 2022 ( Participant ) NTUtion 2022 Hackathon ( Participant ) NUS LifeHack 2023( Participant )'}},
 {'id': '2',
  'score': 0.728767276,
  'values': [],
  'metadata': {'categories': 'personal',
   'isImage': False,
   'text': 'Activities:\nNUS LifeHack 2021 ( Participant )\nSUTD What The Hack: Environment 2021 ( Participant )\nAppetizer Hackathon 2021 ( Participant )\nSPAI Beginner Machine Learning Bootcamp 2021\nSPAI Advance Machine Learning Workshop 2021\nSEED Code League 2021 ( Participant )\nNUS LifeHack 2022 ( Participant )\nNTUtion 2022 Hackathon ( Participant )'}},
 {'id': '3',
  'score': 0.725435615,
  'values': [],
  'metadata': {'categories': 'personal',
   'isImage': False,
   'text': 'NUS LifeHack 2023( Participant )Achievement: Polyfintech100 API Hackthon 2023 ( Champion ) Batey Hackathon 2022 ( Champion Gold ) Polyfintech100 API Hackthon 2023 ( 1st runner up ) Polyfintech 2022 ( 1st Runner-Up) DSAC AI Makerspace Holiday Challenge 2021 ( Champion ) FPG FIT Hack 2021 ( Finalist ) SUTD WTH ( Finalist ) Activities:'}}
   ]
        """

        query_embedding = self.embedding_model.embed_query(query)
        # Get the top 3 results
        results = self.index.query(query_embedding,top_k=top_k,include_metadata=True)
        print('results: ', results)
        
        results_dict = results.to_dict()
        matched_docs = results_dict["matches"]
        return matched_docs

