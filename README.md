agents.py is the setup for all the Agno agents used in the project, it contains the Domain classifier and the Main RAG agent with various custom tools baked in
build_index.py is to be run once on your machine which converts the information PDFs from the documents folder into chunks, currently numbering 603, and store them in a FAISS vector index
userinterface contains the streamlit UI code and it can be launched from terminal to directly open the UI on your machine and access the bot seamlessly
