import streamlit as st
from dotenv import load_dotenv

from src.task10_generation import generate_with_citation


load_dotenv()

st.set_page_config(
    page_title="Hà Nội Travel Assistant",
    page_icon="🏮",
    layout="wide",
)

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.title("🏮 Du lịch Hà Nội")
    st.caption(
        "Trợ lý trả lời từ văn bản chính thức và cẩm nang của Sở Du lịch Hà Nội."
    )
    top_k = st.slider("Số chunks", 3, 10, 5)
    st.info("Câu trả lời chỉ được xác nhận khi có trích dẫn [S1], [S2] từ corpus.")

st.title("Trợ lý du lịch Hà Nội")
st.caption(
    "Hỏi về điểm đến, di sản, ẩm thực và định hướng phát triển du lịch Thủ đô."
)


def show_sources(sources: list[dict], retrieval_source: str) -> None:
    if not sources:
        return
    st.caption(f"Phương thức truy xuất: {retrieval_source}")
    with st.expander(f"Nguồn đã sử dụng ({len(sources)})"):
        for index, source in enumerate(sources, 1):
            metadata = source["metadata"]
            st.markdown(f"**[S{index}] {metadata['title']}**")
            url = metadata.get("url")
            if url:
                st.markdown(f"[Mở nguồn gốc]({url})")
            st.caption(
                f"{metadata['source']} · {source['retrieval_method']} · "
                f"score={source['score']:.4f}"
            )
            st.write(source["content"][:500] + ("…" if len(source["content"]) > 500 else ""))

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            show_sources(
                message.get("sources", []), message.get("retrieval_source", "none")
            )

query = st.chat_input("Nhập câu hỏi...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Đang tìm trong cẩm nang Hà Nội…"):
            result = generate_with_citation(query, top_k=top_k)
        st.markdown(result["answer"])
        show_sources(result["sources"], result["retrieval_source"])

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
            "retrieval_source": result["retrieval_source"],
        }
    )
