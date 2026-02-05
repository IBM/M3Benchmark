
from typing import Any, List
import re
import sqlite3

import pandas as pd


sqlite_reserved_keywords = {
    "ABORT", "ADD", "AFTER", "ALL", "ALTER", "ANALYZE", "AND", "AS",
    "ASC", "ATTACH", "AUTOINCREMENT", "BEFORE", "BETWEEN", "BY", "CASE",
    "CAST", "CHECK", "COLLATE", "COLUMN", "CONSTRAINT", "CREATE", "CROSS",
    "CURRENT", "CURRENT_TIME", "CURRENT_TIMESTAMP", "DATABASE", "DEFAULT",
    "DEFERRABLE", "DEFERRED", "DELETE", "DESC", "DETACH", "DISTINCT",
    "DROP", "EACH", "ELSE", "END", "ESCAPE", "EXCEPT", "EXCLUSIVE", "EXISTS",
    "EXPLAIN", "FAIL", "FILTER", "FOR", "FOREIGN", "FROM", "GLOB", "GROUP",
    "HAVING", "IF", "IGNORE", "IMMEDIATE", "IN", "INDEX", "INDEXED", "INNER",
    "INSERT", "INSTEAD", "INTERSECT", "INTO", "IS", "ISNULL", "JOIN", "KEY",
    "LEFT", "LIKE", "LIMIT", "MATCH", "NATURAL", "NO", "NOT", "NOTNULL", "NULL",
    "OF", "OFFSET", "ON", "OR", "ORDER", "OUTER", "PLAN", "PRAGMA", "PRIMARY",
    "QUERY", "RAISE", "RECURSIVE", "REFERENCES", "REGEXP", "REINDEX", "RELEASE",
    "ROLLBACK", "ROW", "SAVEPOINT", "SELECT", "SET", "TABLE", "TEMP", "TEMPORARY",
    "THEN", "TO", "TRANSACTION", "TRIGGER", "UNION", "UNIQUE", "UPDATE", "USING",
    "VACUUM", "VALUES", "VIEW", "VIRTUAL", "SQLITE_SEQUENCE"
}

def is_reserved_keyword(table_name: str) -> bool:
    """
    Check if the table name is a reserved keyword in SQLite.
    
    Args:
    - table_name (str): The name of the table to check.
    
    Returns:
    - bool: True if the table name is a reserved keyword, False otherwise.
    """
    return table_name.upper() in sqlite_reserved_keywords

def is_unsafe_table_name(table_name: str) -> bool:
    keyword = is_reserved_keyword(table_name)
    unsafe = table_name != make_safe(table_name)
    if keyword or unsafe:
        return True
    return False


# Process strings to remove problematic characters

def make_safe(exp: str) -> str:
    # Convert to string if not already a string (handles float, NaN, None, etc.)
    if not isinstance(exp, str):
        exp = str(exp)

    safestr = exp.strip().replace(' ', '_')
    safestr = safestr.replace('(', '').replace(')','')
    safestr = safestr.replace('/', '_')
    safestr = safestr.replace('-', '_')
    return safestr

def safe_name_columns(table: pd.DataFrame) -> pd.DataFrame:
    column_renaming = {}
    for c in table.columns:
        c_safe = make_safe(c)
        if c != c_safe:
            column_renaming[c] = c_safe
    
    return table.rename(columns=column_renaming)

def make_query_safe(query: str) -> str:
    # Regular expression pattern to match text between backticks
    pattern = r'`([^`]*)`'

    # Use re.findall to find all substrings between backticks
    result = re.findall(pattern, query)

    for unsafe_column_name in result:
        escaped_column_name = make_safe(unsafe_column_name)
        query = query.replace(f"`{unsafe_column_name}`", escaped_column_name)
    return query

# Basic database operations

def database_get_connection(database_path: str) -> sqlite3.Connection:
    """Endpoint to create a connection to a database located at 'database_path'
    Arguments: database_path: the location of the .sqlite or .db file
    Returns: An sqlite.Connection object pointing to the database
    """ 
    conn = sqlite3.connect(database_path)    
    return conn

def database_lookup_tables(connection: sqlite3.Connection) -> List[str]:
    """Lookup the names of all tables in the database pointed to by connection
    Arguments: connection: an sqlite.Connection instance pointing to a database
    Returns: table_names: list of string labels for each table in the database
    """
    cursor = connection.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    table_names = cursor.fetchall()
    table_names = [t[0] for t in table_names]  # Return strings, not tuples
    table_names = [t for t in table_names if t != 'sqlite_sequence'] # Don't return special tables
    return table_names

def database_get_table(connection: sqlite3.Connection, table_name: str) -> pd.DataFrame:
    """Load a table from an open database connection and return as a pandas DataFrame
    Arguments: connection: sqlite.Connection to the database
    Returns: df -> a pandas DataFrame
    """
    try:
        if is_unsafe_table_name(table_name):
            query =  f'SELECT * FROM `{table_name}`;'
        else:
            query = f"SELECT * FROM {table_name};"
        df = pd.read_sql_query(query, connection)
    except:
        query = "SELECT name FROM sqlite_master WHERE type='table';"
        df = pd.read_sql_query(query, connection)
        raise Exception(f"Did not find table '{table_name}' in connection. Only found {df['name'].values}")
    return df

def database_close_connection(connection: sqlite3.Connection) -> None:
    """Close an open connection to an sql database
    Arguments: connection: sqlite.Connection to the database
    Returns: None
    """
    connection.close()

def database_drop_table(connection: sqlite3.Connection, table_name: str):
    cursor = connection.cursor()

    # Drop the table if it exists
    cursor.execute(f'DROP TABLE IF EXISTS {table_name}')

    # Commit and close connection
    connection.commit()
    connection.close()

def table_lookup_columns(df: pd.DataFrame) -> List[str]:
    """Lookup the list of columns in a table or dataframe
    Arguments: df: pandas DataFrame
    Returns: list of column names
    """
    return list(df.columns)

def table_get_column(df: pd.DataFrame, column_name: str) -> List[Any]:
    """Lookup the list of columns in a table or dataframe
    Arguments: df: pandas DataFrame
    Returns: list of column names
    """
    return df[column_name].tolist()
