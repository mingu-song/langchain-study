import streamlit as st
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import ChatMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.callbacks.base import BaseCallbackHandler

load_dotenv()

def print_messages() -> None:
    if "messages" in st.session_state and len(st.session_state["messages"]) > 0:
        for message in st.session_state["messages"]:
            st.chat_message(message.role).write(message.content)


def get_session_history(session_ids: str) -> BaseChatMessageHistory:
    if session_ids not in st.session_state["store"]:
        st.session_state["store"][session_ids] = ChatMessageHistory()
    return st.session_state["store"][session_ids]

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text="") -> None:
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


## START

st.set_page_config(page_title="ChatGPT", page_icon="🐶")
st.title("🐶 ChatGPT")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "store" not in st.session_state:
    st.session_state["store"] = dict()

with st.sidebar:
    session_id = st.text_input("Session ID", value="session1")
    clear_btn = st.button("대화기록 초기화")
    if clear_btn:
        st.session_state["messages"] = []
        st.session_state["store"] = dict()
        st.rerun()

print_messages()

if user_input := st.chat_input("메시지를 입력해주세요"):
    st.chat_message("user").write(f"{user_input}")
    st.session_state["messages"].append(ChatMessage(role="user", content=user_input))

    with st.chat_message("assistant"):
        stream_handler = StreamHandler(st.empty())

        llm = ChatOpenAI(streaming=True, callbacks=[stream_handler], model="gpt-4o", temperature=0.7)
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system", "질문에 친절하게 답변해 주세요",
                ),
                # 대화기록을 변수로 사용, history 가 MessageHistory 의 key
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}"),
            ]
        )
        chain = prompt | llm
        chain_with_memory = (
            RunnableWithMessageHistory(
                chain, # 실행할 Runnable 
                get_session_history, # 세션 기록을 가져오는 함수
                input_messages_key="question", # 사용자 질문 키
                history_messages_key="history", # 기록 메시지 키
            )
        )
        response = chain_with_memory.invoke(
            {"question": user_input},
            config={"configurable": {"session_id": session_id}},
        )

        st.session_state["messages"].append(ChatMessage(role="assistant", content=response.content))
