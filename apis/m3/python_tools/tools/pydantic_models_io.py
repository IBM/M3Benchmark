
from enum import Enum
from typing import Any, List, Union, Mapping
from pydantic import BaseModel, ConfigDict, Field, RootModel

from .pydantic_wrapper import KeySummary

# retrieve_data

class RetrieveDataInput(BaseModel):
    model_config = ConfigDict(
        title="RetrieveDataInput",
        extra="forbid", 
        json_schema_extra={
            "description": "Returns the values associated with a key or keys from the input data source. ",
        },
    )
    # Fields
    data_source: str = Field(..., description="Path to data object to select from. ")
    key_name: Union[str, List[str]] = Field(..., description="The key or keys to retrieve the values of from the specified data path. ")
    distinct: bool = Field(False, description="whether to return only distinct values")
    limit: int = Field(-1, description="only return the first 'limit' elements, (-1) indicates no limit on number of returned items")

class RetrieveDataResult(RootModel[Mapping[str, List[Any]]]):
    model_config = ConfigDict(
        title="RetrieveDataResult",
        json_schema_extra={
            "description": "The values associated with the key or keys",
        },
    )


# filter_data

class FilterCondition(str, Enum):
    """
    Available conditions on which the filter can be applied: 

    - 'equal_to': Filters rows where the column's value is equal to the given value.
    - 'not_equal_to': Filters rows where the column's value is not equal to the given value.
    - 'greater_than': Filters rows where the column's value is greater than the given value.
    - 'less_than': Filters rows where the column's value is less than the given value.
    - 'greater_than_equal_to': Filters rows where the column's value is greater than or equal to the given value.
    - 'less_than_equal_to': Filters rows where the column's value is less than or equal to the given value.
    - 'contains': Filters rows where the column's value contains the given value (applies to strings).
    - 'like': Filters rows where the column's value matches a regex pattern (applies to strings).
    """

    EQUAL_TO = "equal_to"
    NOT_EQUAL_TO = "not_equal_to"
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    GREATER_THAN_EQUAL_TO = "greater_than_equal_to"
    LESS_THAN_EQUAL_TO = "less_than_equal_to"
    CONTAINS = "contains"
    LIKE = "like"

class FilterDataInput(BaseModel):
    model_config = ConfigDict(
        title="FilterDataInput",
        extra="forbid", 
        json_schema_extra={
            "description": "Apply a filter based on a specified condition and value to the data at the data_path. ",
        },
    )
    # Fields
    data_source: str = Field(..., description="Path to data object to be filtered. ")
    key_name: str = Field(..., description="The key on which the filter will be applied.")
    value: str = Field(..., description="The value to compare against in the filtering operation.")
    condition: FilterCondition = Field(..., description=(
        "Condition used in the filter. \n\n"
        "Options:\n"
        "- 'equal_to': Filters rows where the column's value is equal to the given value."
        "- 'not_equal_to': Filters rows where the column's value is not equal to the given value."
        "- 'greater_than': Filters rows where the column's value is greater than the given value."
        "- 'less_than': Filters rows where the column's value is less than the given value."
        "- 'greater_than_equal_to': Filters rows where the column's value is greater than or equal to the given value."
        "- 'less_than_equal_to': Filters rows where the column's value is less than or equal to the given value."
        "- 'contains': Filters rows where the column's value contains the given value (applies to strings)."
        "- 'like': Filters rows where the column's value matches a regex pattern (applies to strings)."
    ),
    )

class FilterDataResult(BaseModel):
    model_config = ConfigDict(
        title="FilterDataResult",
        extra="forbid", 
        json_schema_extra={
            "description": "Summary and location of data after filtering. ",
        },
    )
    # Fields
    data_path: str = Field(..., description="Location/path of filtered data")
    num_records: int = Field(..., description="number of individual records in filtered data")
    key_details: List[KeySummary] = Field(..., description="Summary info about each key or attribute in the filtered data. ")


# transform_data

class TransformOperation(str, Enum):
    """
    Available operations to apply to the data: 

    - 'substring': Extract a substring from a string value. Requires operation_args={'start_index': (int), 'end_index': (int)}
    - 'abs': Replace the (numerical) value of each record with its absolute value.
    - 'datetime': Extract an ISO datetime pattern (e.g. '%Y') from the record value. Requires operation_args={'pattern': 'ISO Pattern'}
    """

    SUBSTRING = "substring"
    ABS = "abs"
    DATETIME = "datetime"

class TransformDataInput(BaseModel):
    model_config = ConfigDict(
        title="TransformDataInput",
        extra="forbid", 
        json_schema_extra={
            "description": "Apply a specified transformation to the data at the data_path. ",
        },
    )
    # Fields
    data_source: str = Field(..., description="Path to data object to be transformed. ")
    key_name: str = Field(..., description="The key specifying which values the transformation will be applied to.")
    operation_type: TransformOperation = Field(..., description=(
        "Type of transformation to apply. \n\n"
        "Options:\n"
        "- 'substring': Extract a substring from a string value. Requires operation_args={'start_index': (int), 'end_index': (int)}"
        "- 'abs': Replace the (numerical) value of each record with its absolute value."
        "- 'datetime': Extract an ISO datetime pattern (e.g. '%Y') from the record value. Requires operation_args={'pattern': 'ISO Pattern'}"
    ),
    )
    operation_args: Union[Mapping[str, Any], None] = Field(None, description="Dictionary of keywords args required by the transformation, if any. ")



class TransformDataResult(BaseModel):
    model_config = ConfigDict(
        title="TransformDataResult",
        extra="forbid", 
        json_schema_extra={
            "description": "Summary and location of data after applying transformation. ",
        },
    )
    # Fields
    data_path: str = Field(..., description="Location/path of transformed data")
    num_records: int = Field(..., description="number of individual records in transformed data")
    key_details: List[KeySummary] = Field(..., description="Summary info about each key or attribute in the transformed data. ")


# sort_data

class SortDataInput(BaseModel):
    model_config = ConfigDict(
        title="SortDataInput",
        extra="forbid", 
        json_schema_extra={
            "description": "Sort the data at the data_path in either descending (default) or ascending order. ",
        },
    )
    # Fields
    data_source: str = Field(..., description="Path to data object to be sorted. ")
    key_name: str = Field(..., description="The key on which the sort will be applied.")
    ascending: bool = Field(False, description=(
        "Sort from smallest to largest (default is largest to smallest). \n\n"
    ),
    )

class SortDataResult(BaseModel):
    model_config = ConfigDict(
        title="SortDataResult",
        extra="forbid", 
        json_schema_extra={
            "description": "Summary and location of data after sorting. ",
        },
    )
    # Fields
    data_path: str = Field(..., description="Location/path of sorted data")
    num_records: int = Field(..., description="number of individual records in sorted data")
    key_details: List[KeySummary] = Field(..., description="Summary info about each key or attribute in the sorted data. ")


# aggregate_data

class AggregationOperation(str, Enum):
    """
    Available aggregation operations: min, max, sum, mean, count, std, argmin, argmax
    """

    MIN = "min"
    MAX = "max"
    SUM = "sum"
    MEAN = "mean"
    COUNT = "count"
    STD = "std"
    ARGMIN = "argmin"
    ARGMAX = "argmax"

class AggregateDataInput(BaseModel):
    model_config = ConfigDict(
        title="AggregateDataInput",
        extra="forbid", 
        json_schema_extra={
            "description": "Apply a specified aggregation operation to the values in a single key of the data at data_path. ",
        },
    )
    # Fields
    data_source: str = Field(..., description="Path to data object to be aggregated. ")
    key_name: str = Field("", description="The key on which the aggregation will be applied.")
    aggregation_operation: AggregationOperation = Field(..., description=(
        "Aggregation operation to perform. Available operations: min, max, sum, mean, count, std, argmin, argmax"),
    )
    distinct: bool = Field(False, description="Whether to aggregate only unique values. ")
    limit: int = Field(-1, description="The maximum number of entries to include in the aggregation (default=-1 includes all available entries)")

class AggregateDataResult(RootModel[float | int]):
    model_config = ConfigDict(
        title="AggregateDataResult",
        json_schema_extra={
            "description": "Result of the aggregation operation",
        },
    )

# concatenate_data

class ConcatenateDataInput(BaseModel):
    model_config = ConfigDict(
        title="ConcatenateDataInput",
        extra="forbid", 
        json_schema_extra={
            "description": "Concatenates two data tables along axis 0 (rows). ",
        },
    )
    # Fields
    data_source_1: str = Field(..., description="Path to first data object to be concatenated. ")
    data_source_2: str = Field(..., description="Path to second data object to be concatenated. ")

class ConcatenateDataResult(BaseModel):
    model_config = ConfigDict(
        title="ConcatenateDataResult",
        extra="forbid", 
        json_schema_extra={
            "description": "Summary and location of combined data after concatenating. ",
        },
    )
    # Fields
    data_path: str = Field(..., description="Location/path of combined data")
    num_records: int = Field(..., description="number of individual records in combined data")
    key_details: List[KeySummary] = Field(..., description="Summary info about each key or attribute in the combined data. ")


# select_unique_values

class SelectUniqueDataInput(BaseModel):
    model_config = ConfigDict(
        title="SelectUniqueDataInput",
        extra="forbid", 
        json_schema_extra={
            "description": "Returns only the unique values from the input array. ",
        },
    )
    # Fields
    unique_array: List[Any] = Field(..., description="Array of objects from which duplicates will be removed. ")

class SelectUniqueDataResult(RootModel[List[Any]]):
    model_config = ConfigDict(
        title="SelectUniqueDataResult",
        json_schema_extra={
            "description": "Input array with all duplicate values removed",
        },
    )
