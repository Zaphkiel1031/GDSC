import requests
import streamlit as st

# GIF 圖片的原始 URL
gif_url = "https://raw.githubusercontent.com/j7808833/test_02/main/pic/Cyberpunk_bar_05.gif"

# 使用 st.image 顯示 GIF 圖片
st.image(gif_url)


def call_coze_api(question):
    url = "https://api.coze.com/open_api/v2/chat"
    payload = {
        "conversation_id": "123",
        "bot_id": "7355203240146829328",
        "user": "123333333",
        "query": question,
        "stream": False,
    }
    headers = {
        "Host": "api.coze.com",
        "Content-Type": "application/json",
        "Authorization": "Bearer pat_39uVa6W8tvvVWxNslJ21K9dL15bQo7cGAGhMzyjAxAcQWB2TpCr0gfgvhqFo8Wfo",
        "Connection": "keep-alive",
        "Accept": "*/*",
    }
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()  # This will raise an exception for HTTP errors
        return response.json()  # Assuming the server responds with valid JSON
    except requests.exceptions.RequestException as e:  # This will catch any request-related errors
        return {"error": str(e)}

# Display messages
if "knowledge_messages" not in st.session_state:
    st.session_state.knowledge_messages = [
        {"role": "assistant", "content": "你好，歡迎詢問問題！"}
    ]

for message in st.session_state.knowledge_messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
if question := st.chat_input("請輸入問題..."):
    with st.chat_message("user"):
        st.markdown(f"{question}")
    st.session_state.knowledge_messages.append({"role": "user", "content": question})
    with st.spinner("思考中..."):
        response = call_coze_api(question)
        try:
            if "messages" in response:
                messages = response["messages"]
                # Filter out the messages with type "answer"
                answer_messages = [msg for msg in messages if msg["type"] == "answer"]
                if answer_messages:
                    response_txt = answer_messages[0]["content"]
                else:
                    response_txt = "未能獲取有效回應。"
            else:
                response_txt = response.get("error", "未能獲取有效回應。")
        except Exception as e:
            response_txt = f"處理回應時發生錯誤: {str(e)}"
        st.session_state.knowledge_messages.append(
            {"role": "assistant", "content": response_txt}
        )
    with st.chat_message("assistant"):
        st.markdown(response_txt)
