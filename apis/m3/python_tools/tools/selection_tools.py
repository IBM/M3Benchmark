
from functools import partial, update_wrapper
from typing import Callable, get_args

from .slot_filling_tools import (
    sort_data,
    filter_data,
    aggregate_data,
    CONDITIONS,
    AGGREGATION_OPERATIONS,
)

# Condition descriptions from filter_data docstring
CONDITION_DESCRIPTIONS = {
    "equal_to": "Filters rows where the column's value is equal to the given value.",
    "not_equal_to": "Filters rows where the column's value is not equal to the given value.",
    "greater_than": "Filters rows where the column's value is greater than the given value.",
    "less_than": "Filters rows where the column's value is less than the given value.",
    "greater_than_equal_to": "Filters rows where the column's value is greater than or equal to the given value.",
    "less_than_equal_to": "Filters rows where the column's value is less than or equal to the given value.",
    "contains": "Filters rows where the column's value contains the given value (applies to strings).",
    "like": "Filters rows where the column's value matches a regex pattern (applies to strings).",
}

AGGREGATION_DESCRIPTIONS = {
    "min": "minimum",
    "max": "maximum",
    "sum": "sum",
    "mean": "mean",
    "count": "count",
    "std": "standard deviation",
    "argmin": "index of the minimum",
    "argmax": "index of the maximum",
}


def _create_filter_docstring(condition: str) -> str:
    """Generate docstring for a specific filter condition."""
    desc = CONDITION_DESCRIPTIONS.get(
        condition, f"Filters data using {condition} condition."
    )
    return f"""{desc}

    Args:
        data (dict): The input data to filter.
        key_name (str): The key on which the filter will be applied.
        value: The value to compare against in the filtering operation.

    Returns:
        dict: A new table (dict) containing the rows from the input data that meet the specified condition.

    Raises:
        KeyError: If `key_name` does not exist in the `data`.
    """


def _create_sort_docstring(direction: str) -> str:
    """Generate docstring for ascending/descending sort."""
    return f"""Sort data in {direction} order by the values associated with the chosen key='key_name'
If the input data is list-like, returns the sorted list.
If the input data is tabular, returns the table with rows sorted by the values in column 'key_name'.
If the data is grouped tables, then sort the groups by the value in 'key_name'

    Args:
        data (dict): table in json format
        key_name (str): name of key to sort by

    Returns:
        dict: data sorted in {direction} order
    """


def _create_aggregate_docstring(operation: str) -> str:
    """Generate docstring for a specific aggregation operation."""
    op_desc = AGGREGATION_DESCRIPTIONS.get(operation, operation)
    return f"""Return the {op_desc} of the input data over the specified key.

    Args:
        data (dict): input data
        key_name (str): name of key to apply {operation}
        distinct (bool): whether to apply {operation} to only distinct values

    Returns:
        Union[float, int]: Result of {operation}.
    """


def _create_partial_function(
    base_function: Callable,
    function_name: str,
    docstring: str,
    **bound_params,
) -> Callable:
    """
    Create a partial function with custom name and docstring.

    Args:
        base_function: The function to bind parameters to
        function_name: Name for the new function
        docstring: Docstring for the new function
        **bound_params: Parameters to bind to the function

    Returns:
        Callable: New partial function with updated metadata
    """
    partial_fcn = partial(base_function, **bound_params)
    update_wrapper(partial_fcn, base_function)
    partial_fcn.__name__ = function_name
    partial_fcn.__doc__ = docstring
    return partial_fcn


def fill_slots_filter_data() -> dict[str, Callable]:
    """Create specialized filter functions for each condition type."""
    filled_in_fcns = {}
    for condition in get_args(CONDITIONS):
        name = f"select_data_{condition}"
        docstring = _create_filter_docstring(condition)
        fcn = _create_partial_function(
            filter_data, function_name=name, docstring=docstring, condition=condition
        )
        filled_in_fcns[name] = fcn
    return filled_in_fcns

def fill_slots_sort_data() -> dict[str, Callable]:
    """Create specialized sort functions for ascending/descending."""
    filled_in_fcns = {}
    for direction in ["ascending", "descending"]:
        name = f"sort_data_{direction}"
        docstring = _create_sort_docstring(direction)
        ascending = direction == "ascending"
        fcn = _create_partial_function(
            sort_data, function_name=name, docstring=docstring, ascending=ascending
        )
        filled_in_fcns[name] = fcn
    return filled_in_fcns

def fill_slots_aggregate_data() -> dict[str, Callable]:
    """Create specialized aggregate functions for each operation."""
    filled_in_fcns = {}
    for operation in get_args(AGGREGATION_OPERATIONS):
        name = f"compute_data_{operation}"
        docstring = _create_aggregate_docstring(operation)
        fcn = _create_partial_function(
            aggregate_data,
            function_name=name,
            docstring=docstring,
            aggregation_operation=operation,
            limit=-1,
        )
        filled_in_fcns[name] = fcn
    return filled_in_fcns

