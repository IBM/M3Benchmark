"""Data loader for candidate data."""

from pathlib import Path
from typing import Optional
import pandas as pd


class DataLoader:
    """Loads and caches candidate data from parquet file."""

    _instance: Optional['DataLoader'] = None
    _data: Optional[pd.DataFrame] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize the data loader."""
        if self._data is None:
            self._load_data()

    def _load_data(self):
        """Load candidate data from parquet file."""
        # Find data file - try multiple locations for different deployments
        possible_paths = [
            # apis/bpo/api/data_loader.py -> ../data/
            Path(__file__).parent.parent / "data" / "candidate_data.parquet",
            # Package installation: site-packages/bpo_benchmark/api/data/
            Path(__file__).parent / "data" / "candidate_data.parquet",
            # Current working directory relative to apis/bpo
            Path("apis/bpo/data/candidate_data.parquet"),
            # Main repo: api/data_loader.py -> data/
            Path(__file__).parent.parent.parent / "data" / "candidate_data.parquet",
            # Current working directory
            Path("./data/candidate_data.parquet"),
            # HuggingFace Spaces default path
            Path("/home/user/app/data/candidate_data.parquet"),
        ]

        data_file = None
        for path in possible_paths:
            if path.exists():
                data_file = path
                break

        if data_file is None:
            raise FileNotFoundError(
                f"Data file not found. Searched paths: {[str(p) for p in possible_paths]}"
            )

        self._data = pd.read_parquet(data_file)

        # Parse skills column (may be string representation of list or already parsed)
        import ast
        import numpy as np

        def parse_skills(x):
            # Handle None/NaN
            if x is None:
                return []
            # Check if it's a numpy/pandas array or list (already parsed)
            if isinstance(x, (list, np.ndarray)):
                return list(x) if isinstance(x, np.ndarray) else x
            # Handle string case
            if isinstance(x, str):
                if x == '':
                    return []
                try:
                    return ast.literal_eval(x)
                except (ValueError, SyntaxError):
                    return []
            # Scalar NaN case
            try:
                if pd.isna(x):
                    return []
            except (TypeError, ValueError):
                pass
            return []

        self._data['skills_parsed'] = self._data['skills'].apply(parse_skills)

    @property
    def data(self) -> pd.DataFrame:
        """Get the loaded data."""
        if self._data is None:
            self._load_data()
        return self._data

    def get_by_requisition(self, requisition_id: str) -> pd.DataFrame:
        """Get all candidates for a specific requisition."""
        return self.data[self.data['requisition_id'] == requisition_id].copy()

    def get_similar_requisitions(self, requisition_id: str) -> pd.DataFrame:
        """
        Get candidates from similar requisitions.

        Similarity is determined by matching requisition metadata:
        - Primary: requisition_template_id (most specific)
        - Fallback: department + seniority_level (broader matching)

        This enables data-driven similarity without hardcoded requisition lists.
        """
        # Get the reference requisition's metadata
        ref_rows = self.data[self.data['requisition_id'] == requisition_id]

        if ref_rows.empty:
            # Unknown requisition - return empty DataFrame
            return pd.DataFrame(columns=self.data.columns)

        # Extract metadata from first row (all rows for same req_id have same metadata)
        ref_row = ref_rows.iloc[0]
        ref_template_id = ref_row.get('requisition_template_id')

        # Primary: match by template ID if present
        if pd.notna(ref_template_id) and str(ref_template_id).strip():
            similar_mask = self.data['requisition_template_id'] == ref_template_id
            similar = self.data[similar_mask]
        else:
            similar = pd.DataFrame(columns=self.data.columns)

        # Fallback: if template match is missing/too small, match by dept + seniority
        if similar.empty or similar['requisition_id'].nunique() < 2:
            ref_department = ref_row.get('department')
            ref_seniority = ref_row.get('seniority_level')
            similar_mask = (
                (self.data['department'] == ref_department)
                & (self.data['seniority_level'] == ref_seniority)
            )
            similar = self.data[similar_mask]

        return similar.copy()

    def is_valid_requisition(self, requisition_id: str) -> bool:
        """Check if a requisition ID exists in the data."""
        return requisition_id in self.data['requisition_id'].values

    def get_suggested_requisitions(self, invalid_id: str, limit: int = 4) -> list:
        """
        Get a list of valid requisition IDs to suggest when an invalid ID is provided.

        Returns close-match IDs from the dataset.
        """
        valid_ids = list(self.data['requisition_id'].unique())
        try:
            from rapidfuzz import process, fuzz

            matches = process.extract(
                invalid_id,
                valid_ids,
                scorer=fuzz.WRatio,
                limit=limit,
            )
            return [match[0] for match in matches]
        except Exception:
            # Fall back to first few valid IDs if RapidFuzz isn't available
            return valid_ids[:limit]


# Singleton instance
_loader = DataLoader()


def get_data_loader() -> DataLoader:
    """Get the singleton data loader instance."""
    return _loader
