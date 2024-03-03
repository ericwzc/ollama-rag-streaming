import asyncio
from typing import AsyncIterable

import uvicorn
from fastapi import FastAPI

from langchain.chains import create_retrieval_chain
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

llm = Ollama(model="gemma:2b")
# prompt = ChatPromptTemplate.from_messages([
#     # ("system", "You are world class technical documentation writer."),
#     ("user", "{input}")
# ])

prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}""")

output_parser = StrOutputParser()

embeddings = HuggingFaceInstructEmbeddings(
    model_name="C:\\Users\\ERICWAN5\\.cache\\torch\\sentence_transformers\\hkunlp_instructor-large",
    model_kwargs={"device": "cpu"}
)

db = Chroma(persist_directory="db", embedding_function=embeddings)
retriever = db.as_retriever(search_kwargs={"k": 4})

# chain = prompt | llm | output_parser
document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)

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


async def map_it(ait: AsyncIterable):
    async for text in ait:
        yield f'data: {text}\n\n'


@app.post("/stream")
def stream(body: StreamRequest):
    return StreamingResponse(map_it(retrieval_chain.astream({"input": f"{body.message}"})),
                             media_type="text/event-stream")


if __name__ == '__main__':
    uvicorn.run(host='0.0.0.0', port=8888, app=app)
