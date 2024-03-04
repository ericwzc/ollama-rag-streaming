from typing import AsyncIterable, List, Union

import uvicorn
from fastapi import FastAPI

from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from constants import LLM_MODEL, PERSIST_DIRECTORY, LAPTOP, LAPTOP_EMBEDDING_MODEL_NAME, RETRIEVED_DOC_NUMBER

llm = Ollama(model=LLM_MODEL)

retrieve_prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    ("user",
     "Given the above conversation, generate a search query to look up in order to get information relevant to the "
     "conversation")
])

answer_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the user's questions based on the below context:\n\n{context}"),
    # MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
])

embeddings = HuggingFaceInstructEmbeddings()

if LAPTOP:
    embeddings = HuggingFaceInstructEmbeddings(
        model_name=LAPTOP_EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"}
    )

db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
retriever = db.as_retriever(search_kwargs={"k": RETRIEVED_DOC_NUMBER})

retriever_chain = create_history_aware_retriever(llm, retriever, retrieve_prompt)
document_chain = create_stuff_documents_chain(llm, answer_prompt)
retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)

app = FastAPI()


class StreamRequest(BaseModel):
    message: str
    chat_history: List[str]


async def map_it(ait: AsyncIterable):
    async for text in ait:
        yield f'data: {text}\n\n'


def history_transformer(chat_history: List[str]) -> List[Union[HumanMessage, AIMessage]]:
    return [AIMessage(content=v) if i % 2 == 0 else HumanMessage(content=v) for i, v in
            enumerate(chat_history, start=1)]


@app.post("/stream")
def stream(body: StreamRequest):
    return StreamingResponse(
        map_it(retrieval_chain.astream({
            "input": f"{body.message}",
            "chat_history": history_transformer(body.chat_history)})),
        media_type="text/event-stream")


if __name__ == '__main__':
    uvicorn.run(host='0.0.0.0', port=8888, app=app)
