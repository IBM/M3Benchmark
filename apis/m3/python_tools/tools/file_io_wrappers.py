
import pandas as pd
import functools
import json
import os
from uuid import uuid4
from typing import Any, Callable

import pandas as pd

FILEPATH_KEY="data_path"
DTYPE_METADATA_KEY="_dtypes"


def _write_dtype_sidecar(csv_filepath: str, dtypes: dict) -> None:
    """Write dtype metadata to a sidecar JSON file."""
    dtype_filepath = csv_filepath.replace('.csv', '_dtypes.json')
    with open(dtype_filepath, 'w') as f:
        json.dump(dtypes, f)


def _read_dtype_sidecar(csv_filepath: str) -> dict | None:
    """Read dtype metadata from a sidecar JSON file if it exists."""
    dtype_filepath = csv_filepath.replace('.csv', '_dtypes.json')
    if os.path.exists(dtype_filepath):
        with open(dtype_filepath, 'r') as f:
            return json.load(f)
    return None


def _load_csv_with_dtypes(csv_filepath: str, low_memory: bool = False) -> dict:
    """Load CSV and apply dtype metadata if sidecar file exists."""
    df = pd.read_csv(csv_filepath, low_memory=low_memory)
    result_dict = df.to_dict(orient='list')

    # Check for dtype sidecar file
    dtypes = _read_dtype_sidecar(csv_filepath)
    if dtypes is not None:
        result_dict[DTYPE_METADATA_KEY] = dtypes

    return result_dict


def _save_dict_to_csv_with_dtypes(data: dict, csv_filepath: str) -> None:
    """Save dict to CSV and write dtype metadata to sidecar file if present."""
    # Extract dtype metadata if present
    dtypes = data.get(DTYPE_METADATA_KEY)

    # Convert to DataFrame (strip dtype metadata key if present)
    data_for_csv = {k: v for k, v in data.items() if k != DTYPE_METADATA_KEY}
    pd_result = pd.DataFrame(data_for_csv)

    # Save CSV
    pd_result.to_csv(csv_filepath, index=False)

    # Save dtype sidecar if metadata exists
    if dtypes is not None:
        _write_dtype_sidecar(csv_filepath, dtypes)


peek_description = """
dict: The following summary of the available data is returned:
            {
                "data_path": location where full data can be accessed. 
                "num_records": number of individual records,
                "keys": ["col1", "col2", ...],
                "key_details": {
                    "key1": {
                        "dtype": "int64",
                        "first_3_values": [val1, val2, val3]
                    },
                    ...
                }
            }
"""

def peek_function(df: pd.DataFrame) -> dict:
    """
    Summarizes key information about a pandas DataFrame.

    Parameters:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        dict: A dictionary with the following structure:
            {
                "num_records": number of individual records,
                "key_details": {
                    "key1": {
                        "name": "key_name",
                        "dtype": "int64",
                        "first_3_values": [val1, val2, val3]
                    },
                    ...
                }
            }
    """
    return {
        "num_records": df.shape[0],  # Number of rows, not columns 
        "key_details": [
            {
                "name": col, 
                "dtype": str(df[col].dtype),
                "first_3_values": df[col].head(3).tolist()
            }
            for col in df.columns
        ]
    }

def update_docstring(docstring: str | None, input: bool = True, output: bool = True) -> str:
    if docstring is None:
        return ""
    # Update the docstring to reflect I/O wrappers. 
    lines = docstring.split('\n')
    nextline = False
    newlines = []
    for l in lines:
        if input and l.strip().startswith("data ("):
            l = "    data_source (str): The location of the data file in csv format. "
        if input and l.strip().startswith("data_1 ("):
            l = "    data_source_1 (str): The location of the first data file in csv format. "
        if input and l.strip().startswith("data_2 ("):
            l = "    data_source_2 (str): The location of the second data file in csv format. "
        if output and l.strip().startswith("Returns:"):
            nextline = True
            newlines.append(l)
            continue
        if nextline:
            l = peek_description
            nextline = False
        newlines.append(l)
    newstr = '\n'.join(newlines)
    return newstr

def load_csv_as_dataframe(func: Callable, tempfile_location: str) -> Callable:
    # Add a wrapper to input function to load the input from a csv, save the output as a csv, and return a peek
    @functools.wraps(func)
    def wrapper(data_source: str, *args, **kwargs) -> dict:
        file_path = os.path.join(tempfile_location, "temp_" + str(uuid4()) + ".csv")
        # Load CSV with dtype metadata if available
        dic = _load_csv_with_dtypes(data_source, low_memory=False)
        result = func(dic, *args, **kwargs)
        try:
            # If result is a Pydantic model, convert to dict
            if hasattr(result, 'model_dump'):
                result = result.model_dump()

            # Save result with dtype metadata if present
            _save_dict_to_csv_with_dtypes(result, file_path)
            # Create peek from the data (strip dtype metadata for peek)
            data_for_peek = {k: v for k, v in result.items() if k != DTYPE_METADATA_KEY}
            pd_result = pd.DataFrame(data_for_peek)
            peek = peek_function(pd_result)
            peek[FILEPATH_KEY] = file_path
            return peek
        except:
            return result
    wrapper.__doc__ = update_docstring(func.__doc__, input=True, output=True)
    wrapper.__name__ = func.__name__
    return wrapper

def load_multiple_csvs_as_dataframes(func: Callable, tempfile_location: str) -> Callable:
    # Add a wrapper to input function to load inputs from csv, save the output as a csv and return a peek
    @functools.wraps(func)
    def wrapper(data_source_1: str, data_source_2: str, *args, **kwargs) -> dict:
        file_path = os.path.join(tempfile_location, "temp_" + str(uuid4()) + ".csv")
        # Load CSVs with dtype metadata if available
        dic1 = _load_csv_with_dtypes(data_source_1, low_memory=False)
        dic2 = _load_csv_with_dtypes(data_source_2, low_memory=False)
        result = func(dic1, dic2, *args, **kwargs)
        try:
            # If result is a Pydantic model, convert to dict
            if hasattr(result, 'model_dump'):
                result = result.model_dump()

            # Save result with dtype metadata if present
            _save_dict_to_csv_with_dtypes(result, file_path)
            # Create peek from the data (strip dtype metadata for peek)
            data_for_peek = {k: v for k, v in result.items() if k != DTYPE_METADATA_KEY}
            pd_result = pd.DataFrame(data_for_peek)
            peek = peek_function(pd_result)
            peek[FILEPATH_KEY] = file_path
            return peek
        except:
            return result
    wrapper.__doc__ = update_docstring(func.__doc__, input=True, output=True)
    wrapper.__name__ = func.__name__
    return wrapper

def save_as_csv(func: Callable, tempfile_location: str) -> Callable:
    # Add a wrapper to input function to save the output as a csv and return a peek
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> dict:
        file_path = os.path.join(tempfile_location, "temp_" + str(uuid4()) + ".csv")
        result = func(*args, **kwargs)
        try:
            # If result is a Pydantic model, convert to dict
            if hasattr(result, 'model_dump'):
                result = result.model_dump()

            # Save result with dtype metadata if present
            _save_dict_to_csv_with_dtypes(result, file_path)
            # Create peek from the data (strip dtype metadata for peek)
            data_for_peek = {k: v for k, v in result.items() if k != DTYPE_METADATA_KEY}
            pd_result = pd.DataFrame(data_for_peek)
            peek = peek_function(pd_result)
            peek[FILEPATH_KEY] = file_path
            return peek
        except:
            return result
    wrapper.__doc__ = update_docstring(func.__doc__, input=False, output=True)
    wrapper.__name__ = func.__name__
    return wrapper


def load_from_csv(func: Callable) -> Callable:
    # Add a wrapper to input function to load the functions input data from a csv file.
    @functools.wraps(func)
    def wrapper(data_source: str, *args, **kwargs) -> Any:
        # Load CSV with dtype metadata if available
        dic = _load_csv_with_dtypes(data_source, low_memory=False)
        result = func(dic, *args, **kwargs)
        return result
    wrapper.__doc__ = update_docstring(func.__doc__, input=True, output=False)
    wrapper.__name__ = func.__name__
    return wrapper
