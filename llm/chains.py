from langchain.chains.openai_functions import create_structured_output_chain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI


prompt = PromptTemplate(
    input_variables=['questions','user_question'],
    template = """Role: You are a question checker. Given valid questions, match the user question to one of the questions based on contextual meaning. 
    If none of the given questions match, return False

    Questions: 
    {questions}

    User Question: 
    {user_question}

    Is the user question valid? Valid questions are questions the user_question that has the same contextual meaning as one of the questions in the list of questions:
    DO NOT RE-PHRASE THE QUESTION. Return the question as is. If the user_question doesn't match any of the questions, return 'no matched':
    """
)
llm = ChatOpenAI(model_name='gpt-3.5-turbo',temperature=0)

output_schema = {
    "name": "Question Checker",
    "description": "Given valid questions, match the user question to one of the questions based on contextual meaning. If none of the given questions match, return False else return the matched question without re-phrasing. ",
    "type": "object",
    "properties": {
      "isValidQuestion": {
        "type": "boolean",
        "description": "Is the user question valid?, True or False. Valid questions are questions the user_question that has the same contextual meaning as one of the questions in the list of questions"
      },
      "matched_question":{
        "type": "string",
        "description": "Which exact question is the user question? DO NOT RE-PHRASE THE QUESTION. Return the question as is. If the user_question doesn't match any of the questions, return 'no matched'"
      },
    },
    "required": ["isValidQuestion","matched_question"]
}

output_chain = create_structured_output_chain(llm=llm,prompt = prompt,output_schema=output_schema)