
from collections import defaultdict
import logging
import math
import os
from typing import Callable, Literal, Union

logger = logging.getLogger(__name__)

import pandas as pd

from .sql_query_components import (
    safe_name_columns,
    make_safe,
    database_get_connection,
    database_get_table,
    database_close_connection,
)
from .dtype_utils import preserve_dtypes_in_dict, DTYPE_METADATA_KEY



open_api_conversion_dict = {
    "str": "string",
    "float": "number",
    "int": "integer",
    "bool": "boolean", 
    "list": "array",
    "dict": "object"
}

def translate_data_type(data_type: str):
    enum = None
    data_type = data_type.replace("<class", "").replace(">", "").strip()
    data_type = data_type.replace("'", "").replace('"', '')
    if data_type.startswith("Union"):
        data_type = data_type.replace("Union", "").replace("[", "").replace("]", "")
        subtypes = data_type.split(', ')
        data_type = [{"type": open_api_conversion_dict.get(s, 'object')} for s in subtypes]
    elif data_type.startswith("typing.Literal"):
        enum = data_type.replace("typing.Literal[", "").replace(']', '').split(', ')
        data_type = 'string'
    else:
        data_type = open_api_conversion_dict.get(data_type, 'object')
    return data_type, enum


def clean_for_json(obj):
    if isinstance(obj, dict):
        return {str(k): clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(v) for v in obj]
    elif isinstance(obj, set):
        return [clean_for_json(v) for v in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, (int, str, bool)) or obj is None:
        return obj
    else:
        # fallback: convert unknown types to string
        return str(obj)

def get_best_key(keys: list[str], key_name: str) -> Union[str, None]:
    if key_name in keys:
        return key_name
    else:
        try:
            index = [k.lower() for k in keys].index(key_name.lower())
            key_name = keys[index]
            return key_name
        except ValueError:
            raise KeyError(f"Key: {key_name} not found in table. Choose from available keys: {keys}")



def rewrite_table_alias_column(identifier: str, alias_to_name_dict: dict) -> str:
    rewritten = identifier.split('.')
    table_alias = rewritten[0]
    rewritten = alias_to_name_dict[table_alias]['modified_table_name'] + "_" + make_safe(rewritten[1])
    return table_alias, rewritten
 

JOIN_TYPES = Literal['inner', 'outer', 'left', 'right']
def data_join(data_a: pd.DataFrame, data_b: pd.DataFrame, left_column_name: str, right_column_name: str, join_type: JOIN_TYPES = 'inner') -> pd.DataFrame:
    """
    Joins two Pandas DataFrames on a specified column.

        Args:
            data_a: First DataFrame.
            data_b: Second DataFrame.
            left_column_name: The column name to join on from the first data. None indicates to use the data index. 
            right_column_name: The column name to join on from the second data. None indicates to use the data index. 
            join_type: Type of join: 'inner', 'outer', 'left', or 'right'. Default is 'inner'.

        Returns:
            A new DataFrame resulting from the join.
    """
    if left_column_name is not None and right_column_name is not None:
        left_key = data_a[left_column_name]
        right_key = data_b[right_column_name]
        if left_key.dtype != right_key.dtype:
            try:
                data_b = data_b.assign(**{right_column_name: right_key.astype(left_key.dtype)})
            except (ValueError, TypeError):
                data_a = data_a.assign(**{left_column_name: left_key.astype(str)})
                data_b = data_b.assign(**{right_column_name: right_key.astype(str)})
        joined_df = pd.merge(data_a, data_b, left_on=left_column_name, right_on=right_column_name, how=join_type)
    elif left_column_name is None and right_column_name is not None:
        joined_df = pd.merge(data_a, data_b, left_index=True, right_on=right_column_name, how=join_type)
    elif left_column_name is not None and right_column_name is None:
        joined_df = pd.merge(data_a, data_b, left_on=left_column_name, right_index=True, how=join_type)
    else:
        joined_df = pd.merge(data_a, data_b, left_index=True, right_index=True, how=join_type)
    return joined_df


def initialize_active_data(condition_sequence: list, alias_to_table_dict: dict, database_path: str) -> dict:
    """
    Initializes active data based on the provided condition sequence, alias-to-table mapping, and database path.

    This function checks the validity of the database file at the specified path and raises an exception if the file is not found.
    After validating the database path, the function processes the condition sequence and alias-to-table dictionary
    to return a dictionary of active data.

    Args:
        condition_sequence (list): A list of conditions (joins) to be processed for initializing the data.
        alias_to_table_dict (dict): A dictionary mapping aliases to their respective tables.
        database_path (str): The file path to the database that will be used for the initialization.

    Raises:
        Exception: If the database file cannot be found at the specified location.

    Returns:
        dict: A dictionary representing the initialized active data based on the conditions and table mappings.
    """
    
    if not os.path.isfile(database_path):
        raise Exception(f"failed, can't find database at location ({database_path})")

    connection = database_get_connection(database_path)

    if len(condition_sequence) == 0:
        # Single-table path: preserve original behavior
        for alias, table_data in alias_to_table_dict.items():
            table = database_get_table(connection, table_data['original_table_name'])
            table = safe_name_columns(table)
            modified_table = table.add_prefix(table_data['modified_table_name'] + "_")
        database_close_connection(connection)
        if alias == '':
            return preserve_dtypes_in_dict(table)
        else:
            return preserve_dtypes_in_dict(modified_table)

    # Multi-table: push joins into SQLite to avoid loading full tables into memory.
    # Use PRAGMA to fetch actual column names for case-insensitive resolution.
    cursor = connection.cursor()
    table_columns: dict[str, list[str]] = {}
    for alias, table_data in alias_to_table_dict.items():
        cursor.execute(f'PRAGMA table_info("{table_data["original_table_name"]}")')
        table_columns[alias] = [row[1] for row in cursor.fetchall()]

    # SELECT "alias"."col" AS "prefix_safe_col" for every column of every table
    select_parts = []
    for alias, table_data in alias_to_table_dict.items():
        prefix = table_data['modified_table_name']
        for col in table_columns[alias]:
            select_parts.append(f'"{alias}"."{col}" AS "{prefix}_{make_safe(col)}"')

    # FROM clause anchored on the first alias
    first_alias = list(alias_to_table_dict.keys())[0]
    first_orig = alias_to_table_dict[first_alias]['original_table_name']
    from_clause = f'"{first_orig}" AS "{first_alias}"'

    # Build INNER JOIN clauses from condition_sequence
    joined_aliases = {first_alias}
    join_clauses = []
    for j in condition_sequence:
        left_alias, left_col_raw = j[0].split('.', 1)
        right_alias, right_col_raw = j[1].split('.', 1)

        # Case-insensitive resolution of column names against actual schema
        left_col = next(
            (c for c in table_columns.get(left_alias, []) if c.lower() == left_col_raw.lower()),
            left_col_raw,
        )
        right_col = next(
            (c for c in table_columns.get(right_alias, []) if c.lower() == right_col_raw.lower()),
            right_col_raw,
        )

        if right_alias not in joined_aliases and right_alias in alias_to_table_dict:
            join_table = alias_to_table_dict[right_alias]['original_table_name']
            join_clauses.append(
                f'INNER JOIN "{join_table}" AS "{right_alias}"'
                f' ON "{left_alias}"."{left_col}" = "{right_alias}"."{right_col}"'
            )
            joined_aliases.add(right_alias)
        elif left_alias not in joined_aliases and left_alias in alias_to_table_dict:
            join_table = alias_to_table_dict[left_alias]['original_table_name']
            join_clauses.append(
                f'INNER JOIN "{join_table}" AS "{left_alias}"'
                f' ON "{right_alias}"."{right_col}" = "{left_alias}"."{left_col}"'
            )
            joined_aliases.add(left_alias)

    query = f'SELECT {", ".join(select_parts)} FROM {from_clause} {" ".join(join_clauses)}'
    df = pd.read_sql_query(query, connection)
    database_close_connection(connection)

    return preserve_dtypes_in_dict(df)


def set_query_specific_columns_and_descriptions(join_sequence: list, alias_dict: dict, database_path: str, table_descriptions: dict) -> list[dict]:
    
    current_table = initialize_active_data(join_sequence, alias_dict, database_path)
    current_table_keys = [k for k in current_table.keys() if k != DTYPE_METADATA_KEY]

    # Invert the alias and table names to search data loader
    table_to_alias_dict = defaultdict(list)
    for k in alias_dict.keys():
        table_to_alias_dict[alias_dict[k]['original_table_name']].append(k)

    # Construct the list of query-specific names and their descriptions
    query_specific_columns_and_descriptions = []
    for table in table_to_alias_dict.keys():
        # Fix for mislabeled tables
        temp_tablename = get_best_key(list(table_descriptions.keys()), table)
        metadata = table_descriptions[temp_tablename]
        aliases = table_to_alias_dict[table]
        if aliases is None:  # Can't iterate over None
            aliases = ['']
        for alias in aliases:
            for m in metadata:
                col_name = m['column_name']
                col_desc = m['column_description']
                col_dtype = m['column_dtype']
                col_dtype, _ = translate_data_type(str(col_dtype))
                modified_col_name = alias_dict[alias]['modified_table_name'] + "_" + col_name if alias != '' else col_name
                if modified_col_name in current_table_keys:
                    query_specific_columns_and_descriptions.append({'key_name': modified_col_name, 'description': col_desc, 'dtype': col_dtype})
    try:
        assert len(query_specific_columns_and_descriptions) == len(current_table_keys)
    except:
        logger.error("Failed to build name-description list. There were %d api calls and %d columns", len(query_specific_columns_and_descriptions), len(current_table_keys))
        raise Exception("Didn't create the right number of getters. ")
    
    query_specific_columns_and_descriptions = clean_for_json(query_specific_columns_and_descriptions)
    return query_specific_columns_and_descriptions
    

def get_column(data: dict, key_name: str) -> list:
        row = [None if isinstance(x, float) and math.isnan(x) else x for x in data[key_name]]
        return row

def create_getter(column: str, column_description: str, column_dtype: str | None = None) -> Callable[[dict], list]:

    def getter_func(data: dict) -> list:
        return get_column(data, key_name=column)
    getter_name = f"get_{column}s"
    if column_dtype is None:
        getter_return_type = list
    else:
        getter_return_type = list[column_dtype]
    getter_docstring = f"Lookup data {column}: {column_description}"
    getter_docstring = getter_docstring + '\n\n\tArgs:\n\t\t' + "data (dict): table containing the data to be retrieved"
    getter_docstring = getter_docstring + '\n\tReturns:\n\t\t' + f"{getter_return_type}" + f": list of {column}s"
    getter_func.__name__ = getter_name
    getter_func.__doc__ = getter_docstring
    return getter_func
 
def create_getters_from_table(table_metadata: list[dict]) -> dict[str, Callable]:
    # Initialize the list of API calls that will be translated to open API specs to build the dataset
    template_api_calls = {}

    for column in table_metadata:
        col_name = column['column_name']
        col_desc = column['column_description']
        getter = create_getter(col_name, col_desc)
        template_api_calls[col_name] = getter
    return template_api_calls
