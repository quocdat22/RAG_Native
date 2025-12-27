"""Zilliz Cloud synchronization utility for syncing with Supabase."""
import logging
from pathlib import Path
from typing import Optional

from config.settings import settings
from src.embedding.embedder import get_embedder
from src.ingestion.chunking import smart_chunk_documents, smart_chunk_markdown
from src.ingestion.loaders import DocumentLoader
from src.storage.supabase_client import get_supabase_storage
from src.storage.zilliz_store import get_zilliz_store

logger = logging.getLogger(__name__)


async def sync_zilliz_from_supabase(document_id: Optional[str] = None) -> dict:
    """
    Sync Zilliz Cloud with Supabase documents.
    
    Args:
        document_id: If provided, sync only this document. Otherwise sync all.
        
    Returns:
        dict with sync statistics
    """
    try:
        supabase_storage = get_supabase_storage()
        vector_store = get_zilliz_store()
        embedder = get_embedder()
        
        # Get documents from Supabase
        if document_id:
            doc = supabase_storage.get_document(document_id)
            documents = [doc] if doc else []
        else:
            documents = supabase_storage.list_documents(limit=1000)
        
        if not documents:
            logger.info("No documents found in Supabase")
            return {"synced": 0, "failed": 0, "skipped": 0}
        
        logger.info(f"🔄 Starting sync for {len(documents)} documents from Supabase to Zilliz")
        
        synced = 0
        failed = 0
        skipped = 0
        
        for doc in documents:
            try:
                # Check if already in Zilliz by checking chunk count
                existing_count = vector_store.count_document_chunks(doc['id'])
                expected_count = doc.get('chunk_count', 0)
                
                if existing_count == expected_count and expected_count > 0:
                    logger.info(f"⏭️  Skipping {doc['filename']} - already synced ({existing_count} chunks)")
                    skipped += 1
                    continue
                
                logger.info(f"📥 Syncing {doc['filename']} (ID: {doc['id']})")
                
                # Download file from Supabase Storage
                file_path = doc.get('file_path')
                if not file_path:
                    logger.warning(f"⚠️  No file_path for document {doc['id']}, skipping")
                    skipped += 1
                    continue
                
                file_content = supabase_storage.download_document(file_path)
                
                # Save to temp location for processing
                temp_dir = Path(settings.documents_dir) / "temp"
                temp_dir.mkdir(exist_ok=True, parents=True)
                temp_file = temp_dir / f"sync_{doc['id']}_{doc['filename']}"
                
                with open(temp_file, "wb") as f:
                    f.write(file_content)
                
                try:
                    # Load and process document
                    pages, is_markdown = DocumentLoader.load(temp_file)
                    
                    # Chunk document
                    if is_markdown:
                        chunks = smart_chunk_markdown(
                            pages,
                            chunk_size=settings.chunking.size,
                            chunk_overlap=settings.chunking.overlap
                        )
                    else:
                        chunks = smart_chunk_documents(
                            pages,
                            chunk_size=settings.chunking.size,
                            chunk_overlap=settings.chunking.overlap
                        )
                    
                    # Generate embeddings
                    texts = [chunk.text for chunk in chunks]
                    embeddings = embedder.embed_texts(texts)
                    
                    # Add to Zilliz with the original document ID
                    vector_store.add_documents(chunks, embeddings, document_id=doc['id'])
                    
                    logger.info(f"✅ Synced {doc['filename']}: {len(chunks)} chunks")
                    synced += 1
                    
                finally:
                    # Cleanup temp file
                    if temp_file.exists():
                        temp_file.unlink()
                
            except Exception as e:
                logger.error(f"❌ Failed to sync document {doc.get('filename', doc['id'])}: {e}")
                failed += 1
                continue
        
        result = {
            "synced": synced,
            "failed": failed,
            "skipped": skipped,
            "total": len(documents)
        }
        
        logger.info(f"🎉 Sync complete: {synced} synced, {skipped} skipped, {failed} failed")
        return result
        
    except Exception as e:
        logger.error(f"❌ Sync failed: {e}")
        raise


def sync_zilliz_from_supabase_sync(document_id: Optional[str] = None) -> dict:
    """Synchronous version of sync_zilliz_from_supabase for use in lifespan."""
    import asyncio
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(sync_zilliz_from_supabase(document_id))
