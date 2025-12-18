import streamlit as st
import requests
import re
from datetime import datetime

# API Configuration
API_BASE_URL = "http://localhost:8000"

# Page config
st.set_page_config(
    page_title="Document Library - RAG Native",
    page_icon="📚",
    layout="wide"
)

# Custom CSS for the "Zoom" effect and grid
st.markdown("""
<style>
    .chunk-card {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 15px;
        height: 200px;
        overflow: hidden;
        position: relative;
        background-color: white;
        transition: transform 0.2s, box-shadow 0.2s;
        cursor: pointer;
    }
    .chunk-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border-color: #1E88E5;
    }
    .chunk-header {
        font-weight: bold;
        font-size: 0.8rem;
        color: #666;
        margin-bottom: 8px;
        display: flex;
        justify-content: space-between;
    }
    .chunk-content {
        font-size: 0.9rem;
        line-height: 1.4;
        color: #333;
        display: -webkit-box;
        -webkit-line-clamp: 6;
        -webkit-box-orient: vertical;
        overflow: hidden;
    }
    .chunk-overlay {
        position: absolute;
        bottom: 0;
        left: 0;
        right: 0;
        height: 40px;
        background: linear-gradient(transparent, white);
        pointer-events: none;
    }
    .zoom-btn {
        position: absolute;
        top: 10px;
        right: 10px;
        background-color: #1E88E5;
        color: white;
        border-radius: 50%;
        width: 24px;
        height: 24px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

def format_latex(text: str) -> str:
    if not text:
        return text
    text = re.sub(r'\\\[(.*?)\\\]', r'$$\1$$', text, flags=re.DOTALL)
    text = re.sub(r'\\\((.*?)\\\)', r'$\1$', text, flags=re.DOTALL)
    return text

def get_documents():
    try:
        response = requests.get(f"{API_BASE_URL}/documents")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching documents: {e}")
        return {"documents": [], "total": 0}

def get_document_chunks(doc_id):
    try:
        response = requests.get(f"{API_BASE_URL}/documents/{doc_id}/chunks")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching chunks: {e}")
        return None

def delete_document(doc_id):
    try:
        response = requests.delete(f"{API_BASE_URL}/documents/{doc_id}")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error deleting document: {e}")
        return None

@st.dialog("Chunk Detail", width="large")
def show_chunk_detail(chunk):
    st.markdown(f"### Chunk ID: `{chunk['chunk_id']}`")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Page", chunk['metadata'].get('page', 'N/A'))
    with col2:
        st.metric("Tokens", chunk['metadata'].get('token_count', 'N/A'))
    with col3:
        st.metric("Source", chunk['metadata'].get('filename', 'Unknown'))
    
    st.markdown("---")
    st.markdown("#### Content")
    st.markdown(format_latex(chunk['text']))
    
    st.markdown("---")
    st.json(chunk['metadata'])

def main():
    st.title("📚 Document Library")
    st.markdown("Manage your documents and inspect their content chunks.")
    
    # Refresh button
    if st.button("🔄 Refresh Documents"):
        st.rerun()

    # Get documents
    docs_data = get_documents()
    docs = docs_data.get("documents", [])
    
    if not docs:
        st.info("No documents found. Please upload documents in the Chat page.")
        return

    # Document selection
    doc_options = {f"{doc['filename']} ({doc['document_id'][:8]})": doc for doc in docs}
    selected_doc_name = st.selectbox("Select a document to inspect:", options=list(doc_options.keys()))
    selected_doc = doc_options[selected_doc_name]

    # Document Actions
    col_info, col_del = st.columns([4, 1])
    with col_info:
        st.write(f"**Type:** {selected_doc['file_type'].upper()} | **Uploaded:** {selected_doc['upload_timestamp']}")
    with col_del:
        if st.button("🗑️ Delete Document", use_container_width=True, type="secondary"):
            if st.checkbox("Confirm Delete"):
                res = delete_document(selected_doc['document_id'])
                if res:
                    st.success("Deleted!")
                    st.rerun()

    st.markdown("---")
    
    # Load and display chunks
    with st.spinner("Loading chunks..."):
        chunks_data = get_document_chunks(selected_doc['document_id'])
    
    if chunks_data and chunks_data.get("chunks"):
        chunks = chunks_data["chunks"]
        st.subheader(f"Chunks ({len(chunks)})")
        
        # Grid layout
        cols_per_row = 3
        for i in range(0, len(chunks), cols_per_row):
            row_chunks = chunks[i:i + cols_per_row]
            cols = st.columns(cols_per_row)
            
            for j, chunk in enumerate(row_chunks):
                with cols[j]:
                    # Using a combination of HTML and a button for the "Zoom" UX
                    # Since we can't easily put a streamlit button inside a custom HTML div with full functionality
                    # we use a container and style it.
                    
                    # Create a unique key for the button
                    chunk_key = f"chunk_{chunk['chunk_id']}"
                    
                    st.markdown(f"""
                    <div class="chunk-header">
                        <span>Chunk {i+j+1}</span>
                        <span>Page {chunk['metadata'].get('page', 'N/A')}</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Truncated preview
                    preview_text = chunk['text'][:500] + "..." if len(chunk['text']) > 500 else chunk['text']
                    st.markdown(f'<div class="chunk-content">{preview_text}</div>', unsafe_allow_html=True)
                    
                    if st.button("🔍 Zoom In", key=chunk_key):
                        show_chunk_detail(chunk)
                    
                    st.markdown('<div style="height: 20px;"></div>', unsafe_allow_html=True)
    else:
        st.warning("No chunks found for this document.")

if __name__ == "__main__":
    main()
