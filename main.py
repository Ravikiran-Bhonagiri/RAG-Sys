"""
Main CLI for Production RAG System.
Orchestrates ingestion and query workflows.
"""
import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
# Add ROOT directory to path for imports
sys.path.insert(0, str(Path(__file__).parents[1]))

from production_rag.ingestion.extractors import PDFExtractor
from production_rag.ingestion.processors import TextChunker, ImageCaptioner
from production_rag.ingestion.storage import VectorStore
from production_rag.retrieval.search import HybridRetriever
from production_rag.generation.rag import RAGGenerator
from production_rag.config import DATA_DIR


def ingest_document(pdf_path: str, reset: bool = False):
    """
    Ingest a PDF document into the vector store.
    
    Args:
        pdf_path: Path to PDF file
        reset: Whether to reset the vector store before ingesting
    """
    print(f"\n🚀 Starting Document Ingestion")
    print(f"📄 PDF: {pdf_path}")
    
    # Initialize components
    vector_store = VectorStore()
    
    if reset:
        print("🗑️ Resetting vector store...")
        vector_store.reset()
    
    text_chunker = TextChunker()
    image_captioner = ImageCaptioner()
    
    # Step 1: Extract
    print("\n📂 Step 1: Extracting content from PDF...")
    with PDFExtractor(pdf_path) as extractor:
        data = extractor.extract_all()
    
    print(f"   Found {data['metadata']['total_pages']} pages")
    print(f"   Found {len(data['images'])} images")
    
    # Step 2: Process text
    print("\n✂️ Step 2: Chunking text...")
    text_chunks = text_chunker.chunk_pages(data['pages'])
    print(f"   Created {len(text_chunks)} text chunks")
    
    # Step 3: Process images
    print("\n🖼️ Step 3: Captioning images...")
    image_captions = image_captioner.caption_images(data['images'])
    print(f"   Generated {len(image_captions)} captions")
    
    # Step 4: Store
    print("\n💾 Step 4: Storing in vector database...")
    all_chunks = text_chunks + image_captions
    vector_store.add_chunks(all_chunks)
    
    print(f"\n✅ Ingestion Complete!")
    print(f"📊 Total chunks in DB: {vector_store.count()}")


def query_system(question: str):
    """
    Query the RAG system.
    
    Args:
        question: User question
    """
    print(f"\n🔍 Query: {question}")
    
    # Initialize components
    vector_store = VectorStore()
    
    if vector_store.count() == 0:
        print("\n❌ Error: No documents in database. Please ingest documents first.")
        return
    
    retriever = HybridRetriever(vector_store)
    generator = RAGGenerator()
    
    # Retrieve
    print("\n📚 Retrieving relevant context...")
    results = retriever.retrieve_and_rerank(question)
    print(f"   Found {len(results)} relevant chunks")
    
    # Generate
    print("\n🤖 Generating answer...")
    response = generator.answer_query(question, results)
    
    # Display
    print("\n" + "="*60)
    print("📝 ANSWER:")
    print("="*60)
    print(response['answer'])
    print("\n" + "="*60)
    print(f"📌 Used {response['num_sources']} sources")
    print("="*60)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Production RAG System",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Ingest command
    ingest_parser = subparsers.add_parser('ingest', help='Ingest a PDF document')
    ingest_parser.add_argument('--file', required=True, help='Path to PDF file')
    ingest_parser.add_argument('--reset', action='store_true', help='Reset database before ingesting')
    
    # Query command
    query_parser = subparsers.add_parser('query', help='Query the RAG system')
    query_parser.add_argument('question', help='Question to ask')
    
    args = parser.parse_args()
    
    if args.command == 'ingest':
        pdf_path = Path(args.file)
        if not pdf_path.exists():
            # Try relative to DATA_DIR
            pdf_path = DATA_DIR / args.file
        
        if not pdf_path.exists():
            print(f"❌ Error: File not found: {args.file}")
            sys.exit(1)
        
        ingest_document(str(pdf_path), reset=args.reset)
    
    elif args.command == 'query':
        query_system(args.question)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
