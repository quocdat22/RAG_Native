"""Streamlit frontend for RAG Native."""
import requests
import streamlit as st
from datetime import datetime

# API Configuration
API_BASE_URL = "http://localhost:8000"

# Page config
st.set_page_config(
    page_title="RAG Native - Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .source-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .chunk-preview {
        background-color: #fafafa;
        padding: 0.8rem;
        border-left: 3px solid #1E88E5;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "search_type" not in st.session_state:
        st.session_state.search_type = "hybrid"
    if "top_k" not in st.session_state:
        st.session_state.top_k = 5


def upload_document(file):
    """Upload a document to the API."""
    files = {"file": (file.name, file.getvalue(), file.type)}
    try:
        response = requests.post(f"{API_BASE_URL}/documents/upload", files=files)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error uploading document: {e}")
        return None


def get_documents():
    """Get list of all documents."""
    try:
        response = requests.get(f"{API_BASE_URL}/documents")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching documents: {e}")
        return {"documents": [], "total": 0}


def delete_document(doc_id):
    """Delete a document."""
    try:
        response = requests.delete(f"{API_BASE_URL}/documents/{doc_id}")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error deleting document: {e}")
        return None


def search_documents(query, top_k, search_type):
    """Search documents."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/search",
            json={
                "query": query,
                "top_k": top_k,
                "search_type": search_type
            }
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error searching: {e}")
        return None


def chat(query, top_k, search_type):
    """Ask a question using RAG."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/chat",
            json={
                "query": query,
                "top_k": top_k,
                "search_type": search_type
            }
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error in chat: {e}")
        return None


def render_sidebar():
    """Render sidebar with document management."""
    with st.sidebar:
        st.markdown("## 📚 Document Library")
        
        # Upload section
        st.markdown("### Upload Document")
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=["pdf", "docx", "txt"],
            help="Upload PDF, DOCX, or TXT files"
        )
        
        if uploaded_file is not None:
            if st.button("📤 Upload", use_container_width=True):
                with st.spinner("Processing document..."):
                    result = upload_document(uploaded_file)
                    if result:
                        st.success(
                            f"✅ Uploaded: {result['filename']}\n\n"
                            f"Chunks created: {result['chunk_count']}"
                        )
                        st.rerun()
        
        st.markdown("---")
        
        # Document list
        st.markdown("### Documents")
        docs = get_documents()
        
        if docs["total"] == 0:
            st.info("No documents uploaded yet")
        else:
            st.caption(f"Total: {docs['total']} documents")
            
            for doc in docs["documents"]:
                with st.container():
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.markdown(f"**{doc['filename']}**")
                        st.caption(f"{doc['file_type'].upper()}")
                    
                    with col2:
                        if st.button("🗑️", key=f"del_{doc['document_id']}", help="Delete"):
                            result = delete_document(doc['document_id'])
                            if result:
                                st.success("Deleted!")
                                st.rerun()
                    
                    st.markdown("---")
        
        # Settings
        st.markdown("### ⚙️ Settings")
        
        st.session_state.search_type = st.selectbox(
            "Search Method",
            ["hybrid", "vector", "bm25"],
            index=["hybrid", "vector", "bm25"].index(st.session_state.search_type),
            help="Vector: semantic similarity | BM25: keyword matching | Hybrid: both combined"
        )
        
        st.session_state.top_k = st.slider(
            "Number of chunks to retrieve",
            min_value=3,
            max_value=10,
            value=st.session_state.top_k,
            help="More chunks = more context but slower"
        )


def render_main():
    """Render main chat interface."""
    st.markdown('<p class="main-title">🔬 Research Assistant</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Ask questions about your research documents</p>', unsafe_allow_html=True)
    
    # Check if documents exist
    docs = get_documents()
    if docs["total"] == 0:
        st.warning("👈 Please upload documents in the sidebar to get started")
        
        # Show example queries
        st.markdown("### Example Questions You Can Ask:")
        st.markdown("""
        - What are the main findings in the papers about machine learning?
        - Compare the methodologies used in different studies
        - What datasets were used in the research?
        - Summarize the conclusions from the uploaded documents
        """)
        return
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display sources if available
            if message.get("sources"):
                with st.expander("📖 Sources"):
                    for source in message["sources"]:
                        st.markdown(
                            f"- **{source['filename']}**, page {source['page']} "
                            f"({source['file_type'].upper()})"
                        )
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = chat(
                    query=prompt,
                    top_k=st.session_state.top_k,
                    search_type=st.session_state.search_type
                )
                
                if response:
                    st.markdown(response["answer"])
                    
                    # Show sources
                    if response.get("sources"):
                        with st.expander("📖 Sources"):
                            for source in response["sources"]:
                                st.markdown(
                                    f"- **{source['filename']}**, page {source['page']} "
                                    f"({source['file_type'].upper()})"
                                )
                    
                    # Add to message history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response["answer"],
                        "sources": response.get("sources", [])
                    })
                else:
                    st.error("Failed to generate response")


def main():
    """Main application."""
    init_session_state()
    render_sidebar()
    render_main()


if __name__ == "__main__":
    main()
