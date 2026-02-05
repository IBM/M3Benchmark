
from datetime import datetime
import math
import re
from typing import Any, Callable, List, Mapping, Union
from typing_extensions import Literal

import numpy as np
import pandas as pd

from .sql_tools import get_best_key
from .dtype_utils import (
    create_dataframe_with_metadata,
    preserve_dtypes_in_dict,
    extract_dtype_metadata,
)

def group_data_by(data: dict, key_name: str) -> dict:
    """Group data by values of a specified key
        Args:
            data (dict): Table of data to be grouped
            key_name (str): name of key to group by
        Returns:
            dict: data with additional layer of keys given by unique values in the column given by `key_name`
    """
    assert key_name in data.keys(), f"Key: {key_name} not found, choose from {data.keys()}. "

    # Extract and preserve dtype metadata
    original_dtypes = extract_dtype_metadata(data)
    df = create_dataframe_with_metadata(data)
    grouped_keys = list(set(data[key_name]))
    grouped_data = {}
    for key_val in grouped_keys:
        selected = df[df[key_name] == key_val]
        grouped_data[key_val] = preserve_dtypes_in_dict(selected, original_dtypes)
    return grouped_data


def retrieve_data(data: dict, key_name: Union[str, list[str]], distinct: bool = False, limit: int = -1) -> dict:
    """Returns contents of a data column or columns
        Args: 
            data (dict): input data table in dictionary format
            key_name (Union[str, list[str]]): key name or list of key names to select
            distinct (bool): whether to return only distinct values
            limit (int): only return the first 'limit' elements
        Returns: 
            dict: selected subset of input data
    """
    if isinstance(key_name, str):
        key_name = get_best_key(list(data.keys()), key_name)
        # TODO: add distinct for multiple columns
        if distinct:  # Only keep unique values, but preserve order
            data = {key_name: list(dict.fromkeys(data[key_name]))}
        key_name = [key_name]

    selected_data = {}
    for key in key_name:
        if limit >= 0:
            row = data[key][:limit]
        else:
            row = data[key]
        row = [None if isinstance(x, float) and math.isnan(x) else x for x in row]
        selected_data[key] = row
    return selected_data


def aggregate_list(data: list, agg_operation: Callable, distinct: bool = False, limit: int = -1) -> Union[float, int]:
    if distinct:
        data = select_unique_values(data)
    
    if limit >= 0:
        data = data[:limit]

    result = agg_operation(data)
    return result

AGGREGATION_OPERATIONS = Literal['min', 'max', 'sum', 'mean', 'count', 'std', 'argmin', 'argmax']
def aggregate_data(data: Union[dict, list], aggregation_operation: AGGREGATION_OPERATIONS, key_name: str = '', distinct: bool = False, limit: int = -1) -> Union[float, int, pd.Series]:
    """Perform an aggregation on input data. 
    If the input data is list-like, returns the value of the aggregation over the list index. 
    If the input data is tabular, returns a numerical value for the aggregation over a column. 
    If the data is grouped tables, then replace the values in the specified key with their aggregation result
        
        Args: 
            data (Union[dict, list]): data to be aggregated
            key_name (str): name of key to aggregate
            aggregation_operation (typing.Literal['min', 'max', 'sum', 'mean', 'count', 'std', 'argmin', 'argmax']): the aggregation operation to perform, must be one of ['min', 'max', 'sum', 'mean', 'count', 'std', 'argmin', 'argmax']
            distinct (bool): whether to aggregate only distinct values
            limit (int): limit the aggregation to the first 'limit' elements
        Returns: 
            dict: Result of aggregation. 
    """
    
    # Check request validity
    assert aggregation_operation in ['min', 'max', 'sum', 'mean', 'std', 'count', 'argmin', 'argmax'], f"{aggregation_operation} not a valid aggregation. "

    agg_operation = agg_dict[aggregation_operation]
    if isinstance(data, list):
        return aggregate_list(data, agg_operation, distinct=distinct, limit=limit)
    
    if key_name in ["", None]:
        assert aggregation_operation == 'count', f"Can't aggregate '*' with aggregation type {aggregation_operation}"
        k = list(data.keys())[0]
        return aggregate_list(data[k], agg_operation, distinct=distinct, limit=limit)

    else:
        key_name = get_best_key(list(data.keys()), key_name)
        assert isinstance(data[key_name], list)
        assert len(data[key_name]) > 0, "Empty data can't be aggregated. "

        return aggregate_data(data[key_name], aggregation_operation, key_name=key_name, distinct=distinct, limit=limit)
    # TODO: implement aggregate for groups
    # else:
    #     groups = list(data.keys())
    #     assert isinstance(data[groups[0]], dict)
    #     assert key_name in data[groups[0]].keys()
    #     for group in groups:
    #         vals = data[group][key_name]
    #         if distinct:
    #             vals = select_unique_values(vals)
    #         grouped_val = agg_operation(vals)
    #         lth = len(data[group][key_name])  # lth needs to match original length, not distinct length
    #         data[group][key_name] = [grouped_val] * lth

        # return data


OPERATION_TYPES = Literal['substring', 'abs', 'datetime']
def transform_data(data: dict, key_name: str, operation_type: OPERATION_TYPES, operation_args: Union[Mapping[str, Any], None]) -> dict:
    """
    Transform the data assigned to a key in the input table (dict) using the specified operation and operation_args
    
        Args:
            data (dict): data to be transformed
            key_name (str): name of key to transform
            operation_type (typing.Literal['substring', 'abs', 'datetime]): the type of operation to perform, must be one of ['substring', 'abs', 'datetime']
            operation_args (dict): any arguments required by the operation_type
        Returns:
            dict: original data with the values corresponding to the specified key transformed
    """

    assert operation_type in ['substring', 'abs', 'datetime'], f"Operation type {operation_type} not supported. "
    key_name = get_best_key(list(data.keys()), key_name)
        
    if operation_type == "substring":
        if operation_args is None:
            operation_args = {}
        start_index = operation_args.get('start_index', 0)
        end_index = operation_args.get('end_index', -1)
        data = transform_data_to_substring(data, key_name, start_index, end_index)
    elif operation_type == "abs":
        data = transform_data_to_absolute_value(data, key_name)
    elif operation_type == "datetime":
        if operation_args is None:
            operation_args = {"pattern": "%Y"}
        datetime_pattern = operation_args["pattern"]
        data = transform_data_to_datetime_part(data, key_name, datetime_pattern)
    return data


def transform_data_to_absolute_value(data: dict, key_name: str) -> dict:
    """
    Transform numeric values into their absolute value.

        Args:
            data (dict): table containing the data to be transformed
            key_name (str): name of string valued key to take absolute value of
        Returns:
            dict: original table (dict) with the data associated with key_name replaced with its absolute value
    """
    key_name = get_best_key(list(data.keys()), key_name)
    data[key_name] = [abs(val) for val in data[key_name]]

    # Update dtype metadata if present - absolute value preserves numeric type
    dtypes = extract_dtype_metadata(data)
    if dtypes is not None and key_name in dtypes:
        # Dtype should remain the same for absolute value (int stays int, float stays float)
        pass

    return data


def datetime_extraction(value: str, pattern: str):
    try:
        dt = datetime.fromisoformat(value)
        if pattern == "%Y":
            return dt.year
        elif pattern == "%m":
            return dt.month
        elif pattern == "%d":
            return dt.day
    except (ValueError, TypeError):
        return value

def transform_data_to_datetime_part(data: dict, key_name: str, datetime_pattern: str) -> dict:
    """
    Transform list of datetime strings into specified subpart.

        Args:
            data (dict): table containing the data to be transformed
            key_name (str): name of string valued key to transform
            datetime_pattern (str): ISO datetime pattern to extract
        Returns:
            dict: original table (dict) with the specified key values transformed to the specified datetime pattern
    """
    key_name = get_best_key(list(data.keys()), key_name)
    data[key_name] = [datetime_extraction(val, datetime_pattern) for val in data[key_name]]

    # Update dtype metadata if present - datetime extraction produces integers
    dtypes = extract_dtype_metadata(data)
    if dtypes is not None and key_name in dtypes:
        # Year/month/day extraction produces integer values
        data['_dtypes'][key_name] = 'int64'

    return data



def transform_data_to_substring(data: dict, key_name: str, start_index: int, end_index: int) -> dict:
    """
    Transform list of string values by taking substrings

        Args:
            data (dict): table containing the data to be transformed
            key_name (str): name of string valued key to transform
            start_index (int): start of substring
            end_index (int): end of substring, must be >= start_index
        Returns:
            dict: original table (dict) with the specified key values transformed
    """
    key_name = get_best_key(list(data.keys()), key_name)
    data[key_name] = [val[start_index:end_index] for val in data[key_name]]

    # Update dtype metadata if present - substring extraction keeps string type
    dtypes = extract_dtype_metadata(data)
    if dtypes is not None and key_name in dtypes:
        # Substring extraction keeps the dtype as object (string)
        data['_dtypes'][key_name] = 'object'

    return data


        
def concatenate_data(data_1: dict, data_2: dict) -> dict:
    """
    Concatenates two tables along axis 0 (rows)
        Args:
            data_1 (dict): First Table.
            data_2 (dict): Second table.

        Returns:
            dict: A new table with the same columns as the input tables and the sum of their rows.
    """
    # Extract dtype metadata from both inputs
    dtypes_1 = extract_dtype_metadata(data_1)
    dtypes_2 = extract_dtype_metadata(data_2)

    # Strip dtype metadata for comparison
    from .dtype_utils import strip_dtype_metadata
    data_1_keys = strip_dtype_metadata(data_1).keys() if dtypes_1 else data_1.keys()
    data_2_keys = strip_dtype_metadata(data_2).keys() if dtypes_2 else data_2.keys()
    assert set(data_1_keys) == set(data_2_keys)

    df_1 = create_dataframe_with_metadata(data_1)
    df_2 = create_dataframe_with_metadata(data_2)
    df_concat = pd.concat([df_1, df_2], axis=0, ignore_index=True)

    # Use dtypes from first table (both should have same columns and dtypes)
    return preserve_dtypes_in_dict(df_concat, dtypes_1)


CONDITIONS = Literal[
    "equal_to",
    "not_equal_to",
    "greater_than",
    "less_than",
    "greater_than_equal_to",
    "less_than_equal_to",
    "contains",
    "like",
    ]

CALCULATOR_OPERATIONS = Literal['add', 'subtract', 'multiply', 'divide']

def filter_data(data: Mapping[str, list[Any]], key_name: str, value: str, condition: CONDITIONS) -> Mapping[str, list[Any]]:
    """

    This function applies a filter on the given key of the input data
    based on the provided condition and value. 
    It returns a new table (dict) with the rows that meet the condition.

    Args:
        data (dict): The input data to filter. 
        key_name (str): The key on which the filter will be applied.
        value: The value to compare against in the filtering operation.
        condition (typing.Literal['equal_to', 'not_equal_to', 'greater_than', 'less_than', 'greater_than_equal_to', 'less_than_equal_to', 'contains', 'like']): The condition to apply for filtering. It must be one of the following:
            - 'equal_to': Filters rows where the column's value is equal to the given value.
            - 'not_equal_to': Filters rows where the column's value is not equal to the given value.
            - 'greater_than': Filters rows where the column's value is greater than the given value.
            - 'less_than': Filters rows where the column's value is less than the given value.
            - 'greater_than_equal_to': Filters rows where the column's value is greater than or equal to the given value.
            - 'less_than_equal_to': Filters rows where the column's value is less than or equal to the given value.
            - 'contains': Filters rows where the column's value contains the given value (applies to strings).
            - 'like': Filters rows where the column's value matches a regex pattern (applies to strings).

    Returns:
        dict: A new table (dict) containing the rows from the input data that meet the specified condition.

    Raises:
        ValueError: If the `condition` is not one of the recognized options.
        KeyError: If `key_name` does not exist in the `data`.
    """
    key_name = get_best_key(list(data.keys()), key_name)
    # Deal with str to int comparisons
    if type(data[key_name][0]) == str and type(value) == int:
        value = str(value)

    # Extract and preserve dtype metadata
    original_dtypes = extract_dtype_metadata(data)
    df = create_dataframe_with_metadata(data)
    
    if isinstance(value, str) and df[key_name].dtype == float:
        value = float(value)
    elif isinstance(value, str) and df[key_name].dtype == int:
        value = int(value)
    elif isinstance(value, float) and df[key_name].dtype == int:
        value = int(value)
    if condition == 'equal_to':
        if value is None:
            df = df[pd.isna(df[key_name])]
        else:
            df = df[df[key_name] == value]
    elif condition == 'not_equal_to':
        if value is None:
            df = df[pd.notna(df[key_name])]
        else:
            df = df[df[key_name] != value]
    elif condition == 'greater_than':
        df = df[(df[key_name].notna()) & (df[key_name] > value)]
    elif condition == 'less_than':
        df = df[(df[key_name].notna()) & (df[key_name] < value)]
    elif condition == 'greater_than_equal_to':
        df = df[(df[key_name].notna()) & (df[key_name] >= value)]
    elif condition == 'less_than_equal_to':
        df = df[(df[key_name].notna()) & (df[key_name] <= value)]
    elif condition == 'contains':
        df = df[df[key_name].str.contains(value, case=False, na=False)]
    elif condition == 'like':
        pattern = _process_pattern_to_regex(value)
        df = df[df[key_name].apply(lambda x: compare_like_pattern(pattern, x))]

    return preserve_dtypes_in_dict(df, original_dtypes)

def _process_pattern_to_regex(pattern: str) -> str:
    pattern = pattern.replace('%', '.*')
    pattern = pattern.replace('_', '.')

    # Escape special regex characters in the pattern, except for % and _
    pattern = re.escape(pattern)

    # Undo escaping of the regex parts we inserted
    pattern = pattern.replace(r'\.\*', '.*')  # unescape .*
    pattern = pattern.replace(r'\.', '.')     # unescape .

    # Replace SQL wildcards with equivalent regex patterns
    pattern = pattern.replace(r'\%', '.*')  # % becomes .*
    pattern = pattern.replace(r'\_', '.')   # _ becomes .

    # Add start and end anchors to match the whole string
    pattern = '^' + pattern + '$'
    return pattern

def sort_data(
    data: Mapping[str, list[Any]],
    key_name: str = '',
    ascending: bool = False,
    ranking_array: Union[List[float], List[int], None] = None
) -> dict:
    """Sort data by the values associated with the chosen key='key_name'
    If the input data is list-like, returns the sorted list.
    If the input data is tabular, returns the table with rows sorted by the values in column 'key_name'.
    If the data is grouped tables, then sort the groups by the value in 'key_name'

        Args:
            data (dict): table in json format
            key_name (str): name of key to sort by
            ascending (bool): whether to sort by ascending order
            ranking_array (Union[List[float], List[int], None]): Optional array of numeric values to use for sorting.
                If provided, data will be sorted by these values instead of key_name.
                Must have the same length as the number of rows in data.
                Typically used when sorting by computed values from arithmetic expressions.

        Returns:
            dict: data sorted by chosen key
    """

    # Validate ranking_array if provided
    if ranking_array is not None:
        # ranking_array only works with tabular data
        if isinstance(data, list):
            raise ValueError("ranking_array cannot be used with list data, only tabular data")

        # Check for empty array first
        if not ranking_array:  # Empty list
            raise ValueError("ranking_array cannot be empty")

        # Check length match
        data_row_count = len(next(iter(data.values()))) if data else 0
        ranking_array_length = len(ranking_array)

        if ranking_array_length != data_row_count:
            raise ValueError(
                f"ranking_array length ({ranking_array_length}) must match "
                f"data row count ({data_row_count})"
            )

    # Handle strings and currency
    # if isinstance(data[key_name][0], str):
    #     try:
    #         data[key_name] = [float(v.strip().lstrip('$').replace(',','')) for v in data[key_name]]
    #     except:
    #         pass

    if isinstance(data, list):
        reverse = not ascending
        return sorted(data, reverse=reverse)

    # Extract and preserve dtype metadata
    original_dtypes = extract_dtype_metadata(data)
    df = create_dataframe_with_metadata(data)

    # Determine sort key
    if ranking_array is not None:
        # Use ranking_array for sorting via temporary column
        temp_col_name = '__temp_ranking__'
        df[temp_col_name] = ranking_array
        sort_key = temp_col_name
    else:
        # Use existing column (current behavior)
        sort_key = get_best_key(list(data.keys()), key_name)
        assert isinstance(data[sort_key], list)
        assert len(data[sort_key]) > 0, "Empty data can't be sorted. "

    # Perform stable sort
    sorted_data = df.sort_values(by=sort_key, ascending=ascending, kind='stable')

    # Remove temporary column if it was added
    if ranking_array is not None:
        sorted_data = sorted_data.drop(columns=[temp_col_name])

    return preserve_dtypes_in_dict(sorted_data, original_dtypes)

    # else:
    #     groups = list(data.keys())
    #     assert isinstance(data[groups[0]], dict)
    #     assert key_name in data[groups[0]].keys()
    #     for group in groups:
    #         vals = data[group][key_name]
    #         grouped_val = agg_operation(vals)
    #         lth = len(data[group][key_name])  # lth needs to match original length, not distinct length
    #         data[group][key_name] = [grouped_val] * lth

    #     return data

    # df = pd.DataFrame(data)
    # sorted_data = df.sort_values(by=key_name, ascending=ascending, kind='mergesort')
    # return sorted_data.to_dict(orient='list')


def Calculator(input_1: Union[str, int, float, List[str], List[int], List[float]],
               input_2: Union[str, int, float, List[str], List[int], List[float]],
               operation: CALCULATOR_OPERATIONS) -> Union[float, int, List[float], List[int]]:
    """Perform arithmetic operations on scalar or list inputs.

    Supports element-wise operations on lists and scalar-scalar operations.
    String inputs are converted to numeric types where possible.

    Args:
        input_1 (Union[str, int, float, List[str], List[int], List[float]]): First operand
        input_2 (Union[str, int, float, List[str], List[int], List[float]]): Second operand
        operation (typing.Literal['add', 'subtract', 'multiply', 'divide']): The arithmetic operation to perform, must be one of ['add', 'subtract', 'multiply', 'divide']

    Returns:
        Union[float, int, List[float], List[int]]: Result of the arithmetic operation

    Raises:
        ValueError: If the operation is not supported or inputs have incompatible shapes
        ZeroDivisionError: If division by zero is attempted
    """
    assert operation in ['add', 'subtract', 'multiply', 'divide'], f"{operation} not a valid operation."

    # Convert string inputs to numeric
    def to_numeric(val):
        if isinstance(val, str):
            try:
                return int(val)
            except ValueError:
                try:
                    return float(val)
                except ValueError:
                    raise ValueError(f"Cannot convert '{val}' to numeric type")
        return val

    # Handle list inputs
    if isinstance(input_1, list) or isinstance(input_2, list):
        # Convert to lists if needed
        list_1 = input_1 if isinstance(input_1, list) else [input_1]
        list_2 = input_2 if isinstance(input_2, list) else [input_2]

        # Convert string elements to numeric
        list_1 = [to_numeric(x) for x in list_1]
        list_2 = [to_numeric(x) for x in list_2]

        # Handle scalar broadcast
        if len(list_1) == 1 and len(list_2) > 1:
            list_1 = list_1 * len(list_2)
        elif len(list_2) == 1 and len(list_1) > 1:
            list_2 = list_2 * len(list_1)

        # Check length compatibility
        if len(list_1) != len(list_2):
            raise ValueError(f"Incompatible list lengths: {len(list_1)} and {len(list_2)}")

        # Perform element-wise operation
        result = []
        for a, b in zip(list_1, list_2):
            # Handle None values - return NaN for any operation with None
            if a is None or b is None:
                result.append(float('nan'))
                continue

            if operation == 'add':
                result.append(a + b)
            elif operation == 'subtract':
                result.append(a - b)
            elif operation == 'multiply':
                result.append(a * b)
            elif operation == 'divide':
                if b > 0:
                    result.append(a / b)
                else:
                    result.append(float('nan'))

        return result

    # Handle scalar inputs
    val_1 = to_numeric(input_1)
    val_2 = to_numeric(input_2)

    if operation == 'add':
        result = val_1 + val_2
    elif operation == 'subtract':
        result = val_1 - val_2
    elif operation == 'multiply':
        result = val_1 * val_2
    elif operation == 'divide':
        if val_2 > 0:
            result = val_1 / val_2
        else:
            result = float('Nan')

    return result


def select_unique_values(unique_array: list) -> list:
    """Return only the distinct elements from the input list.

    Args:
        unique_array (list): A list of input data

    Returns:
        list: The distinct elements of the input list
    """

    return list(dict.fromkeys(unique_array))


def truncate(truncate_array, n: int) -> Union[dict, list]:
    """Return the first `n` elements of a list-like object.

    Args:
        truncate_array: A list-like object.
        n (int): The number of rows/elements to return.

    Returns:
        list: The first `n` elements of the list-like object.

    Raises:
        ValueError: If n is negative.
        TypeError: If the input is not a list-like object.
    """
    if n < 0:
        raise ValueError("n must be a non-negative integer.")
    
    return truncate_array[:n]


def compare_like_pattern(pattern: str, comparison_value: str) -> bool:
    if isinstance(comparison_value, float) and np.isnan(comparison_value):
        return False

    # Compile and use the regex to match the string
    matches = bool(re.match(pattern, comparison_value))
    return matches


##############################################################################################################


# Aggregation related functions


##############################################################################################################

def get_min(data: list[float]) -> float:
    """Return the minimum value from a list.

    Args:
        data (list): A list of numbers.

    Returns:
        float: The minimum value in the input data.
    """
    return float(np.nanmin(data))


def get_max(data: list[float]) -> float:
    """Return the maximum value from a list.

    Args:
        data (list): A list of numbers.

    Returns:
        float: The maximum value in the input data.
    """
    return float(np.nanmax(data))


def get_count(data: list[float]) -> int:
    """Return the number of elements in a list.

    Args:
        data (list): A list of numbers.

    Returns:
        int: The number of elements in the input data.
    """
    return len(data)


def get_sum(data: list[float]) -> float:
    """Return the sum of values in a list.

    Args:
        data (list): A list of numbers.

    Returns:
        float: The sum of the input data.
    """
    return float(np.nansum(data))


def get_mean(data: list[float]) -> float:
    """Return the mean of values in a list.

    Args:
        data (list): A list of numbers.

    Returns:
        float: The mean of the input data.
    """
    return float(np.nanmean(data))


def get_std(data: list[float]) -> float:
    """Return the standard deviation of values in a list.

    Args:
        data (list): A list of numbers.

    Returns:
        float: The standard deviation of the input data.
    """
    return float(np.nanstd(data))


def get_argmin(data: list[float]) -> int:
    """Return the index of the minimum value in a list.

    Args:
        data (list): A list of numbers.

    Returns:
        int: The index of the minimum value in the input data.
    """
    return list(data).index(min(data))

def get_argmax(data: list[float]) -> int:
    """Return the index of the maximum value in a list.

    Args:
        data (list): A list of numbers.

    Returns:
        int: The index of the maximum value in the input data.
    """
    # Cast to list makes this work if data is a 1-D numpy array. 
    return list(data).index(max(data))

agg_dict = {
    'min': get_min,
    'max': get_max,
    'count': get_count,
    'sum': get_sum,
    'mean': get_mean,
    'std': get_std,
    'argmin': get_argmin,
    'argmax': get_argmax,
}
