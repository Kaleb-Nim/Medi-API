from langchain.chains.openai_functions import create_structured_output_chain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI


prompt = PromptTemplate(
    input_variables=['questions','user_question'],
    template = """Role: You are a question checker. Given valid questions, match the user question to one of the questions based on contextual meaning. 
    If none of the given questions match, return False

    Example 1:
    Questions:
    ---
    1.1) Is your loved one with dementia restless? 
    1.2) Does your loved one with dementia seem sad or upset? 
    1.3) Does your loved one with dementia seem to be looking for something?
    ---
    2.1) What are the psychological impacts of tube feeding? 
    2.2) tube feeding impacts
    2.3) psychological impacts of tube feeding

    User Question: 
    "potential impacts of tube feeding for my grandma"

    Your Answer:
    {
        "isValidQuestion": true,
        "matched_question": "tube feeding impacts",
        "question_list_index": 2
    }

    ------------------------------------------------------------------------------------------------------
    Questions: 
    {questions}

    User Question: 
    {user_question}

    Task:
    1. Check if the user question is valid. Valid questions are questions the user_question that has the same contextual meaning as at least one of the questions in the list of questions
    2. Given the document questions, match the user question to one of the questions based on contextual meaning.
    """
)
llm = ChatOpenAI(model_name='gpt-3.5-turbo',temperature=0)

output_schema = {
    "name": "Question Checker",
    "description": "Given valid questions, match the user question to one of the questions based on contextual meaning. If none of the given questions match, return False else return the matched question without re-phrasing. ",
    "type": "object",
    "properties": {
      "matched_question":{
        "type": "string",
        "description": "Which exact question is the user question? DO NOT RE-PHRASE THE QUESTION. Return the question as is. If the user_question doesn't match any of the questions, return 'no matched'"
      },
      "question_list_index":{
        "type": "integer",
        "description": "Which question list is the user question? If the user_question doesn't match any of the questions, return -1"
      }
    },
    "required": ["matched_question","question_list_index"]
}

output_chain = create_structured_output_chain(llm=llm,prompt = prompt,output_schema=output_schema)