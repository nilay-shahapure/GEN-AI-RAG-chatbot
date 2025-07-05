# ui.py
import streamlit as st
from agents import answer

st.set_page_config(page_title="Voya Finance Chatbot")
st.title("Voya Finance Chatbot")

q = st.text_input("Ask a question:")
if st.button("Ask") and q.strip():
    with st.spinner("Thinking…"):
        resp = answer(q)
    st.write(resp)
