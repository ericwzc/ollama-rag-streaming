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

llm = Ollama(model="gemma:2b")

# prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:
#
# <context>
# {context}
# </context>
#
# Question: {input}""")

retrieve_prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    ("user",
     "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
])

answer_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the user's questions based on the below context:\n\n{context}"),
    # MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
])

embeddings = HuggingFaceInstructEmbeddings(
    model_name="C:\\Users\\ERICWAN5\\.cache\\torch\\sentence_transformers\\hkunlp_instructor-large",
    model_kwargs={"device": "cpu"}
)

db = Chroma(persist_directory="db", embedding_function=embeddings)
retriever = db.as_retriever(search_kwargs={"k": 4})

# chain = prompt | llm | output_parser
retriever_chain = create_history_aware_retriever(llm, retriever, retrieve_prompt)
document_chain = create_stuff_documents_chain(llm, answer_prompt)
retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)

# setup_and_retrieval = RunnableParallel(
#     {"context": retriever, "input": RunnablePassthrough()}
# )
# chain = setup_and_retrieval | prompt | llm | output_parser


# if __name__ == '__main__':
#     # while True:
#     #     question = input("Enter your question:")
#     #     response = retrieval_chain.invoke({"input": f"{question}"})
#     # # print(response["answer"])
#     #     print(response)
#     question = 'who is Sultan al-Jaber?'
#     for text in retrieval_chain.stream(
#         {"input": f"{question}"}
#     ):
#         print(text, flush=True)


# async def main():
#     question = 'who is Sultan al-Jaber?'
#     async for text in retrieval_chain.astream(
#             {"input": f"{question}"}
#     ):
#         print(text, flush=True)
#
# asyncio.run(main())

app = FastAPI()


class StreamRequest(BaseModel):
    message: str
    chat_history: List[str]


async def map_it(ait: AsyncIterable):
    async for text in ait:
        yield f'data: {text}\n\n'


def history_transformer(chat_history: List[str]) -> List[Union[HumanMessage, AIMessage]]:
    return [AIMessage(content=v) if i % 2 == 0 else HumanMessage(content=v) for i, v in enumerate(chat_history, start=1)]


@app.post("/stream")
def stream(body: StreamRequest):
    return StreamingResponse(
        map_it(retrieval_chain.astream({
            "input": f"{body.message}",
            "chat_history": history_transformer(body.chat_history)})),
        media_type="text/event-stream")


if __name__ == '__main__':
    uvicorn.run(host='0.0.0.0', port=8888, app=app)
