from typing import Any, Dict, List, Union

import numpy as np
from sentence_transformers import SentenceTransformer


class ToolShortlister:
    """Retrieve the most relevant tools for a query via semantic similarity.

    Parameters
    ----------
    top_k : int
        Maximum number of tools to return.  If the catalog is already at or
        below this size the shortlister is a no-op.
    model_name : str
        HuggingFace sentence-transformers model identifier.
    """

    def __init__(self, top_k: int = 128, model_name: str = "all-MiniLM-L6-v2"):
        self.top_k = top_k
        self.model = SentenceTransformer(model_name)
        # Populated by encode_tools()
        self._tool_embeddings: np.ndarray | None = None
        self._tools: list | None = None

    @staticmethod
    def _tool_text(tool: Any) -> str:
        """Build the text representation used for embedding.

        Accepts either:
        - A ``dict`` with ``name`` and ``description`` keys.
        - A LangChain ``StructuredTool`` (or any object with ``.name`` /
          ``.description`` attributes).
        """
        if isinstance(tool, dict):
            name = tool.get("name", "")
            description = tool.get("description", "")
        else:
            name = getattr(tool, "name", "")
            description = getattr(tool, "description", "")
        return f"{name}: {description}"


    def encode_tools(self, tools: list) -> None:
        """Pre-compute embeddings for a tool catalog.

        Call this **once per domain** before any :py:meth:`shortlist` calls.

        Parameters
        ----------
        tools : list
            Either ``List[dict]`` (with *name* / *description* keys) or
            ``List[StructuredTool]``.
        """
        self._tools = list(tools)
        texts = [self._tool_text(t) for t in self._tools]
        self._tool_embeddings = self.model.encode(texts, convert_to_numpy=True)

    def shortlist(self, query: str, tools: list) -> list:
        """Return the *top_k* tools most similar to *query*.

        If ``len(tools) <= top_k`` the original list is returned unchanged
        (no embedding computation).

        Parameters
        ----------
        query : str
            The user query to match against tool descriptions.
        tools : list
            The full tool catalog (same list passed to :py:meth:`encode_tools`).

        Returns
        -------
        list
            A subset of *tools* (same element type) ranked by relevance.
        """
        if len(tools) <= self.top_k:
            return tools

        # Use pre-computed embeddings when available and the catalog matches.
        if (
            self._tool_embeddings is not None
            and self._tools is not None
            and len(self._tools) == len(tools)
        ):
            tool_embs = self._tool_embeddings
        else:
            # Fallback: compute on the fly (shouldn't normally happen).
            texts = [self._tool_text(t) for t in tools]
            tool_embs = self.model.encode(texts, convert_to_numpy=True)

        query_emb = self.model.encode([query], convert_to_numpy=True)

        # Cosine similarity (embeddings are already L2-normalised by default
        # for many sentence-transformer models, but we normalise explicitly to
        # be safe).
        tool_norms = np.linalg.norm(tool_embs, axis=1, keepdims=True)
        query_norm = np.linalg.norm(query_emb, axis=1, keepdims=True)
        tool_embs_normed = tool_embs / np.where(tool_norms == 0, 1, tool_norms)
        query_emb_normed = query_emb / np.where(query_norm == 0, 1, query_norm)

        similarities = (tool_embs_normed @ query_emb_normed.T).squeeze()

        top_indices = np.argsort(similarities)[::-1][: self.top_k]

        return [tools[i] for i in top_indices]
