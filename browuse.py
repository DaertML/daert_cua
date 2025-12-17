from browser_use import Agent, ChatOllama
from dotenv import load_dotenv
import asyncio

load_dotenv()

async def main():
    # good
    #llm = ChatOllama(model="llama3.2")

    #llm = ChatOllama(model="Qwen3:14b")

    # bad
    #llm = ChatOllama(model="gpt-oss:20b")

    # bad/slow
    #llm = ChatOllama(model="hopephoto/Qwen3-4B-Thinking-2507_q8")

    # hallucinates goals
    #llm = ChatOllama(model="hf.co/mradermacher/GUI-Owl-7B-GGUF")

    # good
    #llm = ChatOllama(model="deepseek-r1:1.5b")

    # bad
    #llm = ChatOllama(model="mistral-nemo")
    #llm = ChatOllama(model="qwen3:32b")

    # good
    #llm = ChatOllama(model="granite3.2-vision")

    # good
    #llm = ChatOllama(model="phi4")

    # bad
    #llm = ChatOllama(model="hf.co/Mungert/rwkv7-2.9B-g1-GGUF")

    # bad
    #llm = ChatOllama(model="mistral:7b")

    # bad
    #llm = ChatOllama(model="qwen2.5:32b")

    # good
    llm = ChatOllama(model="lucasmg/deepseek-r1-8b-0528-qwen3-q4_K_M-tool-true")

    #task = "go to amazon.com, and make a list of all the products you find and their price. Get the price of the products, save the results to a .txt file"
    #task = "wait until https://infinitemac.org/1984/System%201.0 boots up, and open the Welcome! file"

    task = "go to https://infinitemac.org/1984/System%201.0, DO NOT OPEN ANY OTHER TAB, ONLY USE THAT TAB. Task: open the Welcome! file"

    #task = "go to https://lichess.org/practice/checkmates/piece-checkmates-i/BJy6fEDf, complete the exercises, DO NOT OPEN ANY OTHER PAGE"
    agent = Agent(task=task, llm=llm)
    await agent.run()

if __name__ == "__main__":
    asyncio.run(main())
