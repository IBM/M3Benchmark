"""
Simple ChromaDB-based retriever using IBM Granite-embedding-r2 model.

This example demonstrates:
1. Setting up ChromaDB with a custom embedding function
2. Indexing sample text chunks
3. Querying the database to retrieve relevant chunks
"""

import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer


class GraniteEmbeddingFunction(chromadb.EmbeddingFunction):
    """Custom embedding function using IBM Granite-embedding-r2 model."""

    def __init__(self, model_name: str = "ibm-granite/granite-embedding-english-r2"):
        """
        Initialize the Granite embedding model.

        Uses granite-embedding-english-r2 from HuggingFace.
        See: https://huggingface.co/ibm-granite/granite-embedding-english-r2
        """
        self._model_name = model_name
        self.model = SentenceTransformer(model_name)

    def __call__(self, input: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts."""
        embeddings = self.model.encode(input, convert_to_numpy=True)
        return embeddings.tolist()

    def name(self) -> str:
        return "granite_embedding"


# Sample chunks to index - these represent document fragments
SAMPLE_CHUNKS = [
    {
        "id": "chunk_1",
        "text": "Python is a high-level, interpreted programming language known for its simplicity and readability. It was created by Guido van Rossum and first released in 1991.",
        "metadata": {"source": "programming_docs", "topic": "python"}
    },
    {
        "id": "chunk_2",
        "text": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.",
        "metadata": {"source": "ai_docs", "topic": "machine_learning"}
    },
    {
        "id": "chunk_3",
        "text": "ChromaDB is an open-source embedding database designed for AI applications. It makes it easy to store and query embeddings along with their metadata.",
        "metadata": {"source": "database_docs", "topic": "chromadb"}
    },
    {
        "id": "chunk_4",
        "text": "Natural Language Processing (NLP) is a field of AI that focuses on the interaction between computers and human language, enabling machines to understand and generate text.",
        "metadata": {"source": "ai_docs", "topic": "nlp"}
    },
    {
        "id": "chunk_5",
        "text": "Vector databases store data as high-dimensional vectors, enabling semantic search capabilities. They are essential for building modern AI applications like RAG systems.",
        "metadata": {"source": "database_docs", "topic": "vector_db"}
    },
    {
        "id": "chunk_6",
        "text": "Retrieval Augmented Generation (RAG) combines retrieval systems with generative AI models to produce more accurate and contextually relevant responses.",
        "metadata": {"source": "ai_docs", "topic": "rag"}
    },
    {
        "id": "chunk_7",
        "text": "IBM Granite is a family of foundation models developed by IBM Research. The Granite embedding models are optimized for creating dense vector representations of text.",
        "metadata": {"source": "ibm_docs", "topic": "granite"}
    },
    {
        "id": "chunk_8",
        "text": "Embeddings are numerical representations of text that capture semantic meaning. Similar texts have embeddings that are close together in vector space.",
        "metadata": {"source": "ai_docs", "topic": "embeddings"}
    }
]


class ChromaDBRetriever:
    """A simple retriever using ChromaDB and Granite embeddings."""

    def __init__(self, collection_name: str = "knowledge_base", persist_directory: str = None):
        """
        Initialize the ChromaDB retriever.

        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the database (None for in-memory)
        """
        # Initialize the Granite embedding function
        self.embedding_fn = GraniteEmbeddingFunction()

        # Create ChromaDB client (in-memory or persistent)
        if persist_directory:
            self.client = chromadb.PersistentClient(path=persist_directory)
        else:
            self.client = chromadb.Client()

        # Get or create collection with custom embedding function
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn
        )

    def add_chunks(self, chunks: list[dict], batch_size: int = 5000) -> None:
        """
        Add chunks to the collection in batches.

        Args:
            chunks: List of dicts with 'id', 'text', and optional 'metadata'
            batch_size: Max chunks per insert (ChromaDB limit is 5461)
        """
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            ids = [chunk["id"] for chunk in batch]
            documents = [chunk["text"] for chunk in batch]
            metadatas = [chunk.get("metadata", {}) for chunk in batch]

            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
            )
            print(f"Added batch {start // batch_size + 1} ({len(batch)} chunks)")

        print(f"Total: {len(chunks)} chunks added to the collection.")

    def query(self, question: str, n_results: int = 3) -> dict:
        """
        Query the collection to find relevant chunks.

        Args:
            question: The query text
            n_results: Number of results to return

        Returns:
            Dictionary containing matching documents, distances, and metadata
        """
        results = self.collection.query(
            query_texts=[question],
            n_results=n_results
        )
        return results

    def get_relevant_chunks(self, question: str, n_results: int = 3) -> list[str]:
        """
        Get the most relevant text chunks for a question.

        Args:
            question: The query text
            n_results: Number of results to return

        Returns:
            List of relevant text chunks
        """
        results = self.query(question, n_results)
        return results["documents"][0] if results["documents"] else []


def main():
    """Main function demonstrating the ChromaDB retriever."""
    print("=" * 60)
    print("ChromaDB Retriever with Granite Embeddings")
    print("=" * 60)

    # Initialize the retriever
    print("\n1. Initializing retriever with Granite-embedding-r2 model...")
    retriever = ChromaDBRetriever()

    # Index the sample chunks
    print("\n2. Indexing sample chunks...")
    retriever.add_chunks(SAMPLE_CHUNKS)

    # Example queries
    queries = [
        "What is RAG and how does it work?",
        "Tell me about Python programming",
        "How do vector databases work?",
        "What are IBM Granite models?"
    ]

    print("\n3. Running queries...")
    print("-" * 60)

    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 40)

        # Get results with metadata
        results = retriever.query(query, n_results=2)

        for i, (doc, distance, metadata) in enumerate(zip(
            results["documents"][0],
            results["distances"][0],
            results["metadatas"][0]
        )):
            print(f"\nResult {i + 1}:")
            print(f"  Text: {doc[:100]}..." if len(doc) > 100 else f"  Text: {doc}")
            print(f"  Distance: {distance:.4f}")
            print(f"  Metadata: {metadata}")

    print("\n" + "=" * 60)
    print("Done!")


if __name__ == "__main__":
    main()
