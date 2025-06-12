import os
import time

import edge_tts
from flask import jsonify
from langchain.chains.question_answering import load_qa_chain
from langchain.retrievers import ContextualCompressionRetriever
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain.retrievers.document_compressors import LLMChainFilter
import warnings
import speech_recognition as sr
from pydub import AudioSegment
from pydub.utils import which
import asyncio

#AudioSegment.converter = r"F:/ffmpeg-7.1.1-essentials_build/bin/ffmpeg.exe"

warnings.filterwarnings("ignore")


os.environ['OPENAI_API_KEY'] = "sk-bCI5yo6f9TadMj4RRQXBLSGIfqJBuIb2J1BxPHfEUEcmiv9z"
os.environ['OPENAI_API_PROXY'] = "https://sg.uiuiapi.com/v1"

api_key = os.getenv("OPENAI_API_KEY")
api_base = os.getenv("OPENAI_API_PROXY")


def identify_intent(question):
    math_keywords = ["几何", "方程", "函数", "微积分", "概率", "三角", "面积", "公式", "数学", "怎么算", "加减", "乘除"]
    if any(kw in question for kw in math_keywords):
        return "math"
    return "chinese"

# 各学科智能体的配置（名字和文本路径）
TEACHER_MAP = {
    "chinese": {
        "name": "李白",
        "text_path": "F:/面试示例/text_chinese.txt"
    },
    "math": {
        "name": "欧几里得",
        "text_path": "F:/面试示例/text_math.txt"
    }
}

assistant_cache = {}

class TTSAssistant():
    def __init__(self, ai_name):
        self.ai_name = ai_name
        self.splitText = []
        self.template = """下面是一段名字叫{ai_name}的老师与人类的对话，{ai_name}会针对人类问题，提供尽可能详细的回答，\n
                        如果{ai_name}不知道答案，会直接回复'我真的不知道'，参考以下相关文档以及历史对话信息，AI会据此组织最终回答内容：
                        {context}
                        {chat_history}
                        Human:{human_input}
                        {ai_name}:"""

        audio_dir = os.path.abspath("audio_files")
        os.makedirs(audio_dir, exist_ok=True)
        self.audio_path = os.path.join(audio_dir, "response.mp3")
        self.latest_audio_path = None

        self.prompt = PromptTemplate(
            template=self.template,
            input_variables=["context", "chat_history", "human_input", "ai_name"]
        )
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            input_key="human_input",
            return_messages=True
        )

    def get_file(self, path):
        loaders = TextLoader(path, encoding="utf-8")
        text = loaders.load()
        if text :
            return text
        else:
            return None

    def split_sentence(self, path):
        text = self.get_file(path)
        if not text:
            print("未加载到任何文本")

        text_splitter = CharacterTextSplitter(
            separator="。",
            chunk_size=200,
            chunk_overlap=20,
            is_separator_regex=False,
            add_start_index=True
        )
        texts = text_splitter.split_documents(text)
        self.splitText = texts

    def embeddingAndVectorDB(self):

        embeddings = OpenAIEmbeddings(
            openai_api_key=api_key,
            openai_api_base=api_base,
        )
        db = Chroma.from_documents(documents=self.splitText, embedding=embeddings)
        return db

    def askAndFindQuestion(self, question):
        db = self.embeddingAndVectorDB()

        retriever = db.as_retriever()
        llm = ChatOpenAI(
            temperature=0,
            openai_api_key=api_key,
            openai_api_base=api_base,
        )
        compressor = LLMChainFilter.from_llm(llm=llm)
        compressor_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=retriever,
        )
        docs = compressor_retriever.invoke(question)
        return docs

    def generate_voice(self, text):
        """生成语音并清理旧的语音文件"""
        # 清理旧的语音文件
        audio_dir = os.path.abspath("audio_files")
        if os.path.exists(audio_dir):
            # 获取所有音频文件并按修改时间排序
            files = [f for f in os.listdir(audio_dir) if f.startswith("response_") and f.endswith(".mp3")]
            files_with_time = [(f, os.path.getmtime(os.path.join(audio_dir, f))) for f in files]
            files_with_time.sort(key=lambda x: x[1], reverse=True)

            # 删除除了最新的1个文件之外的所有文件
            for i in range(1, len(files_with_time)):
                file_to_delete = os.path.join(audio_dir, files_with_time[i][0])
                try:
                    os.remove(file_to_delete)
                    print(f"已删除旧音频文件: {file_to_delete}")
                except Exception as e:
                    print(f"删除文件时出错: {e}")

        # 生成新的语音文件
        filename = f"response_{int(time.time() * 1000)}.mp3"
        self.audio_path = os.path.join("audio_files", filename)

        async def run():
            communicate = edge_tts.Communicate(text, "zh-CN-XiaoxiaoNeural")
            await communicate.save(self.audio_path)

        asyncio.run(run())
        return self.audio_path

    def selfChain(self, question):
        docs = self.askAndFindQuestion(question)
        chain = load_qa_chain(
            llm = ChatOpenAI(
                temperature=0,
                openai_api_key=api_key,
                openai_api_base=api_base,
            ),
            memory=self.memory,
            prompt=self.prompt,
            chain_type="stuff"
        )
        result = chain.invoke({"input_documents":docs, "human_input":question, "ai_name":self.ai_name})
        output = result["output_text"]
        self.generate_voice(output)

        return output




'''
ai_name = input("请你为AI取名：").strip() or "李白"
assistant = TTSAssistant(ai_name)
assistant.split_sentence()

while True:
    question = input("请输入问题：").strip()
    if question.lower() in ["exit", "quit", "退出"]:
        print(f"{ai_name}:再见，祝你好运")
        break

    result = assistant.selfChain(question)
    print(f"{ai_name} : {result['output_text']}")
    print("\n")
'''
