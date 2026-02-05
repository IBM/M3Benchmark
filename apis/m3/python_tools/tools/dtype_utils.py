"""
Utility functions for preserving pandas DataFrame dtypes through dict serialization.

This module implements a "metadata sidecar" pattern where dtype information is stored
alongside data dicts in a special "_dtypes" key. This prevents pandas from incorrectly
re-inferring types when converting dicts back to DataFrames.
"""

import pandas as pd
from typing import Optional


DTYPE_METADATA_KEY = "_dtypes"


def extract_dtypes(df: pd.DataFrame) -> dict[str, str]:
    """
    Extract dtype information from a DataFrame as a serializable dict.

    Args:
        df: The DataFrame to extract dtypes from

    Returns:
        A dict mapping column names to their dtype strings (e.g., {'col1': 'int64', 'col2': 'object'})
    """
    return {col: str(dtype) for col, dtype in df.dtypes.items()}


def apply_dtypes(data: dict, dtypes: dict[str, str]) -> pd.DataFrame:
    """
    Create a DataFrame from a data dict with explicit dtype specifications.

    This function ensures that columns maintain their original types instead of
    having pandas re-infer types (which can lead to incorrect conversions,
    especially for string columns with NaN values and numeric-looking content).

    Args:
        data: Dict with column names as keys and lists of values
        dtypes: Dict mapping column names to their dtype strings

    Returns:
        DataFrame with explicitly specified dtypes
    """
    # Create DataFrame without letting pandas infer types
    df = pd.DataFrame(data)

    # Apply the explicit dtypes
    for col, dtype_str in dtypes.items():
        if col in df.columns:
            # Handle special case: object dtype (pandas' string/mixed type)
            if dtype_str == "object":
                df[col] = df[col].astype("object")
            else:
                try:
                    df[col] = df[col].astype(dtype_str)
                except (ValueError, TypeError):
                    # If conversion fails, keep as object dtype
                    df[col] = df[col].astype("object")

    return df


def preserve_dtypes_in_dict(df: pd.DataFrame, original_dtypes: Optional[dict[str, str]] = None) -> dict:
    """
    Convert a DataFrame to dict format while preserving dtype metadata.

    This function adds dtype information to the output dict so that future
    operations can maintain the correct types.

    Args:
        df: The DataFrame to convert
        original_dtypes: Optional dict of original dtypes to preserve. If provided,
                        dtypes for columns not in the DataFrame will be dropped.
                        If None, dtypes are extracted from the current DataFrame.

    Returns:
        Dict with data in 'list' orient plus a '_dtypes' key containing dtype metadata
    """
    result_dict = df.to_dict(orient='list')

    # Use original dtypes if provided, otherwise extract from current df
    if original_dtypes is not None:
        # Only keep dtypes for columns that still exist in the DataFrame
        dtypes = {col: dtype for col, dtype in original_dtypes.items() if col in df.columns}
        # Add dtypes for any new columns
        for col in df.columns:
            if col not in dtypes:
                dtypes[col] = str(df[col].dtype)
    else:
        dtypes = extract_dtypes(df)

    result_dict[DTYPE_METADATA_KEY] = dtypes
    return result_dict


def has_dtype_metadata(data: dict) -> bool:
    """
    Check if a data dict contains dtype metadata.

    Args:
        data: Dict to check

    Returns:
        True if the dict contains dtype metadata, False otherwise
    """
    return DTYPE_METADATA_KEY in data


def extract_dtype_metadata(data: dict) -> Optional[dict[str, str]]:
    """
    Extract dtype metadata from a data dict.

    Args:
        data: Dict potentially containing dtype metadata

    Returns:
        Dict of dtypes if metadata exists, None otherwise
    """
    return data.get(DTYPE_METADATA_KEY)


def strip_dtype_metadata(data: dict) -> dict:
    """
    Create a copy of a data dict without the dtype metadata key.

    Args:
        data: Dict potentially containing dtype metadata

    Returns:
        New dict with all keys except the dtype metadata key
    """
    return {k: v for k, v in data.items() if k != DTYPE_METADATA_KEY}


def create_dataframe_with_metadata(data: dict) -> pd.DataFrame:
    """
    Create a DataFrame from a data dict, using dtype metadata if available.

    This is a convenience function that checks for metadata and applies it
    if present, otherwise falls back to standard DataFrame creation.

    Args:
        data: Dict potentially containing dtype metadata

    Returns:
        DataFrame with dtypes applied if metadata was available
    """
    if has_dtype_metadata(data):
        dtypes = extract_dtype_metadata(data)
        data_only = strip_dtype_metadata(data)
        return apply_dtypes(data_only, dtypes)
    else:
        return pd.DataFrame(data)
