
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Mapping, Literal
from pydantic import BaseModel, ConfigDict, Field, RootModel, model_validator

from environment.m3.python_tools.tools.dtype_utils import DTYPE_METADATA_KEY


# retrieve_data
class RetrieveDataInput(BaseModel):
    model_config = ConfigDict(
        title="RetrieveDataInput",
        extra="forbid",
        json_schema_extra={
            "description": "Returns the values associated with a key_name (or list of key_names) from the input data source. ",
        },
    )
    # Fields
    data: Mapping[str, Union[List[Any], Dict[str, str]]] = Field(..., description="Data object to select from. ")
    key_name: Union[str, List[str]] = Field(..., description="The key or keys to retrieve the values of from the specified data path. ")
    distinct: bool = Field(False, description="whether to return only distinct values")
    limit: int = Field(-1, description="only return the first 'limit' elements, (-1) indicates no limit on number of returned items")



class RetrieveDataResult(BaseModel):
    model_config = ConfigDict(
        title="RetrieveDataResult",
        extra="forbid",
        json_schema_extra={
            "description": "The values associated with the key_name (or key_names)",
        },
    )

    data: Mapping[str, Union[List[Any], Dict[str, str]]] = Field(..., description="Retrieved data table in dictionary format")



# filter_data
class FilterCondition(str, Enum):
    """
    Available conditions on which the filter can be applied: 

    - 'equal_to': Filters rows where key_name's value is equal to the given value.
    - 'not_equal_to': Filters rows where key_name's value is not equal to the given value.
    - 'greater_than': Filters rows where key_name's value is greater than the given value.
    - 'less_than': Filters rows where key_name's value is less than the given value.
    - 'greater_than_equal_to': Filters rows where key_name's value is greater than or equal to the given value.
    - 'less_than_equal_to': Filters rows where key_name's value is less than or equal to the given value.
    - 'contains': Filters rows where key_name's value contains the given value (applies to strings).
    - 'like': Filters rows where key_name's value matches a regex pattern (applies to strings).
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
            "description": "Apply a filter based on a specified condition and value to the input data. ",
        },
    )
    # Fields
    data: Mapping[str, Union[List[Any], Dict[str, str]]] = Field(..., description="Data object to be filtered. ")
    key_name: str = Field(..., description="The key on which the filter will be applied.")
    value: Union[str, float, int, None] = Field(..., description="The value to compare against in the filtering operation.")
    condition: FilterCondition = Field(..., description=(
        "Condition used in the filter. \n\n"
        "Options:\n"
        "- 'equal_to': Filters rows where key_name's value is equal to the given value."
        "- 'not_equal_to': Filters rows where key_name's value is not equal to the given value."
        "- 'greater_than': Filters rows where key_name's value is greater than the given value."
        "- 'less_than': Filters rows where key_name's value is less than the given value."
        "- 'greater_than_equal_to': Filters rows where key_name's value is greater than or equal to the given value."
        "- 'less_than_equal_to': Filters rows where key_name's value is less than or equal to the given value."
        "- 'contains': Filters rows where key_name's value contains the given value (applies to strings)."
        "- 'like': Filters rows where key_name's value matches a regex pattern (applies to strings)."
    ),
    )


class FilterDataResult(BaseModel):
    model_config = ConfigDict(
        title="FilterDataResult",
        extra="forbid",
        json_schema_extra={
            "description": "Data object after filtering. ",
        },
    )

    data: Mapping[str, Union[List[Any], Dict[str, str]]] = Field(..., description="Filtered data table in dictionary format")



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
            "description": "Apply a specified transformation to the input data. ",
        },
    )
    # Fields
    data: Mapping[str, Union[List[Any], Dict[str, str]]] = Field(..., description="Data object to be transformed. ")
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
            "description": "Data object after applying transformation. ",
        },
    )

    data: Mapping[str, Union[List[Any], Dict[str, str]]] = Field(..., description="Transformed data table in dictionary format")



# sort_data
class SortDataInput(BaseModel):
    model_config = ConfigDict(
        title="SortDataInput",
        extra="forbid",
        json_schema_extra={
            "description": "Sort the input data in either descending (default) or ascending order. ",
        },
    )
    # Fields
    data: Mapping[str, Union[List[Any], Dict[str, str]]] = Field(..., description="Data object to be sorted. ")
    key_name: str = Field(
        default='',
        description="The key on which the sort will be applied. Not required if ranking_array is provided."
    )
    ascending: bool = Field(
        False,
        description="Sort from smallest to largest (default is largest to smallest). \n\n"
    )
    ranking_array: Union[List[float], List[int], None] = Field(
        default=None,
        description=(
            "Optional array of numeric values to use for sorting. "
            "If provided, data will be sorted by these values instead of key_name. "
            "Must have the same length as the number of rows in data. "
            "Typically used when sorting by computed values from arithmetic expressions."
        )
    )

    @model_validator(mode='after')
    def validate_sort_key(self) -> 'SortDataInput':
        """Ensure either key_name or ranking_array is provided."""
        if not self.key_name and self.ranking_array is None:
            raise ValueError("Either key_name or ranking_array must be provided")
        return self


class SortDataResult(BaseModel):
    model_config = ConfigDict(
        title="SortDataResult",
        extra="forbid",
        json_schema_extra={
            "description": "Data object after sorting. ",
        },
    )

    data: Mapping[str, Union[List[Any], Dict[str, str]]] = Field(..., description="Sorted data table in dictionary format")



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
            "description": "Apply a specified aggregation operation to the values in a single key of the data object. ",
        },
    )
    # Fields
    data: Mapping[str, Union[List[Any], Dict[str, str]]] = Field(..., description="Data object to be aggregated. ")
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

class StdInput(BaseModel):
    model_config = ConfigDict(
        title="StdInput",
        extra="forbid",
        json_schema_extra={
            "description": "Get the standard deviation of the values in a single key of the data object. ",
        },
    )
    # Fields
    data: Mapping[str, Union[List[Any], Dict[str, str]]] = Field(..., description="Data object to be aggregated. ")
    key_name: str = Field("", description="The key on which the aggregation will be applied.")


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
    data_1: Mapping[str, Union[List[Any], Dict[str, str]]] = Field(..., description="First data object to be concatenated. ")
    data_2: Mapping[str, Union[List[Any], Dict[str, str]]] = Field(..., description="Second data object to be concatenated. ")


class ConcatenateDataResult(BaseModel):
    model_config = ConfigDict(
        title="ConcatenateDataResult",
        extra="forbid",
        json_schema_extra={
            "description": "Concatenated data objects. ",
        },
    )

    data: Mapping[str, Union[List[Any], Dict[str, str]]] = Field(..., description="Concatenated data table in dictionary format")



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


# ============================================================================
# Specialized Selection Tool Models
# ============================================================================
# These models inherit from base models with pre-filled parameters for
# specialized functions created via functools.partial()


# Filter Models - inherit from FilterDataInput with condition pre-filled

class SelectDataEqualToInput(FilterDataInput):
    """Input for select_data_equal_to function.

    Equivalent to FilterDataInput with condition pre-set to 'equal_to'.
    """
    condition: Literal[FilterCondition.EQUAL_TO] = Field(
        default=FilterCondition.EQUAL_TO,
        description="Condition is pre-set to 'equal_to' for this specialized function.",
    )


class SelectDataNotEqualToInput(FilterDataInput):
    """Input for select_data_not_equal_to function.

    Equivalent to FilterDataInput with condition pre-set to 'not_equal_to'.
    """
    condition: Literal[FilterCondition.NOT_EQUAL_TO] = Field(
        default=FilterCondition.NOT_EQUAL_TO,
        description="Condition is pre-set to 'not_equal_to' for this specialized function.",
    )


class SelectDataGreaterThanInput(FilterDataInput):
    """Input for select_data_greater_than function.

    Equivalent to FilterDataInput with condition pre-set to 'greater_than'.
    """
    condition: Literal[FilterCondition.GREATER_THAN] = Field(
        default=FilterCondition.GREATER_THAN,
        description="Condition is pre-set to 'greater_than' for this specialized function.",
    )


class SelectDataLessThanInput(FilterDataInput):
    """Input for select_data_less_than function.

    Equivalent to FilterDataInput with condition pre-set to 'less_than'.
    """
    condition: Literal[FilterCondition.LESS_THAN] = Field(
        default=FilterCondition.LESS_THAN,
        description="Condition is pre-set to 'less_than' for this specialized function.",
    )


class SelectDataGreaterThanEqualToInput(FilterDataInput):
    """Input for select_data_greater_than_equal_to function.

    Equivalent to FilterDataInput with condition pre-set to 'greater_than_equal_to'.
    """
    condition: Literal[FilterCondition.GREATER_THAN_EQUAL_TO] = Field(
        default=FilterCondition.GREATER_THAN_EQUAL_TO,
        description="Condition is pre-set to 'greater_than_equal_to' for this specialized function.",
    )


class SelectDataLessThanEqualToInput(FilterDataInput):
    """Input for select_data_less_than_equal_to function.

    Equivalent to FilterDataInput with condition pre-set to 'less_than_equal_to'.
    """
    condition: Literal[FilterCondition.LESS_THAN_EQUAL_TO] = Field(
        default=FilterCondition.LESS_THAN_EQUAL_TO,
        description="Condition is pre-set to 'less_than_equal_to' for this specialized function.",
    )


class SelectDataContainsInput(FilterDataInput):
    """Input for select_data_contains function.

    Equivalent to FilterDataInput with condition pre-set to 'contains'.
    """
    condition: Literal[FilterCondition.CONTAINS] = Field(
        default=FilterCondition.CONTAINS,
        description="Condition is pre-set to 'contains' for this specialized function.",
    )


class SelectDataLikeInput(FilterDataInput):
    """Input for select_data_like function.

    Equivalent to FilterDataInput with condition pre-set to 'like'.
    """
    condition: Literal[FilterCondition.LIKE] = Field(
        default=FilterCondition.LIKE,
        description="Condition is pre-set to 'like' for this specialized function.",
    )


# Sort Models - inherit from SortDataInput with ascending pre-filled

class SortDataAscendingInput(SortDataInput):
    """Input for sort_data_ascending function.

    Equivalent to SortDataInput with ascending pre-set to True.
    """
    ascending: Literal[True] = Field(
        default=True,
        description="Sort order is pre-set to ascending for this specialized function.",
    )


class SortDataDescendingInput(SortDataInput):
    """Input for sort_data_descending function.

    Equivalent to SortDataInput with ascending pre-set to False.
    """
    ascending: Literal[False] = Field(
        default=False,
        description="Sort order is pre-set to descending for this specialized function.",
    )


# Aggregate Models - inherit from AggregateDataInput with aggregation_operation and limit pre-filled

class ComputeDataMinInput(AggregateDataInput):
    """Input for compute_data_min function.

    Equivalent to AggregateDataInput with aggregation_operation='min' and limit=-1.
    """
    aggregation_operation: Literal[AggregationOperation.MIN] = Field(
        default=AggregationOperation.MIN,
        description="Aggregation operation is pre-set to 'min' for this specialized function.",
    )
    limit: Literal[-1] = Field(
        default=-1,
        description="Limit is pre-set to -1 (no limit) for this specialized function.",
    )


class ComputeDataMaxInput(AggregateDataInput):
    """Input for compute_data_max function.

    Equivalent to AggregateDataInput with aggregation_operation='max' and limit=-1.
    """
    aggregation_operation: Literal[AggregationOperation.MAX] = Field(
        default=AggregationOperation.MAX,
        description="Aggregation operation is pre-set to 'max' for this specialized function.",
    )
    limit: Literal[-1] = Field(
        default=-1,
        description="Limit is pre-set to -1 (no limit) for this specialized function.",
    )


class ComputeDataSumInput(AggregateDataInput):
    """Input for compute_data_sum function.

    Equivalent to AggregateDataInput with aggregation_operation='sum' and limit=-1.
    """
    aggregation_operation: Literal[AggregationOperation.SUM] = Field(
        default=AggregationOperation.SUM,
        description="Aggregation operation is pre-set to 'sum' for this specialized function.",
    )
    limit: Literal[-1] = Field(
        default=-1,
        description="Limit is pre-set to -1 (no limit) for this specialized function.",
    )


class ComputeDataMeanInput(AggregateDataInput):
    """Input for compute_data_mean function.

    Equivalent to AggregateDataInput with aggregation_operation='mean' and limit=-1.
    """
    aggregation_operation: Literal[AggregationOperation.MEAN] = Field(
        default=AggregationOperation.MEAN,
        description="Aggregation operation is pre-set to 'mean' for this specialized function.",
    )
    limit: Literal[-1] = Field(
        default=-1,
        description="Limit is pre-set to -1 (no limit) for this specialized function.",
    )


class ComputeDataCountInput(AggregateDataInput):
    """Input for compute_data_count function.

    Equivalent to AggregateDataInput with aggregation_operation='count' and limit=-1.
    """
    aggregation_operation: Literal[AggregationOperation.COUNT] = Field(
        default=AggregationOperation.COUNT,
        description="Aggregation operation is pre-set to 'count' for this specialized function.",
    )
    limit: Literal[-1] = Field(
        default=-1,
        description="Limit is pre-set to -1 (no limit) for this specialized function.",
    )


class ComputeDataStdInput(AggregateDataInput):
    """Input for compute_data_std function.

    Equivalent to AggregateDataInput with aggregation_operation='std' and limit=-1.
    """
    aggregation_operation: Literal[AggregationOperation.STD] = Field(
        default=AggregationOperation.STD,
        description="Aggregation operation is pre-set to 'std' for this specialized function.",
    )
    limit: Literal[-1] = Field(
        default=-1,
        description="Limit is pre-set to -1 (no limit) for this specialized function.",
    )


class ComputeDataArgminInput(AggregateDataInput):
    """Input for compute_data_argmin function.

    Equivalent to AggregateDataInput with aggregation_operation='argmin' and limit=-1.
    """
    aggregation_operation: Literal[AggregationOperation.ARGMIN] = Field(
        default=AggregationOperation.ARGMIN,
        description="Aggregation operation is pre-set to 'argmin' for this specialized function.",
    )
    limit: Literal[-1] = Field(
        default=-1,
        description="Limit is pre-set to -1 (no limit) for this specialized function.",
    )


class ComputeDataArgmaxInput(AggregateDataInput):
    """Input for compute_data_argmax function.

    Equivalent to AggregateDataInput with aggregation_operation='argmax' and limit=-1.
    """
    aggregation_operation: Literal[AggregationOperation.ARGMAX] = Field(
        default=AggregationOperation.ARGMAX,
        description="Aggregation operation is pre-set to 'argmax' for this specialized function.",
    )
    limit: Literal[-1] = Field(
        default=-1,
        description="Limit is pre-set to -1 (no limit) for this specialized function.",
    )


# Generic Getter Models
class GetterInput(BaseModel):
    """Generic input model for dynamically created getter functions.

    Getter functions retrieve a specific column from the input table data.
    They are created dynamically based on database schema using create_getter().
    """
    model_config = ConfigDict(
        title="GetterInput",
        extra="forbid",
        json_schema_extra={
            "description": "Input for getter functions that retrieve a specific column from table data.",
        },
    )

    data: Mapping[str, List[Any]] = Field(
        ...,
        description="Table data containing key_name to retrieve. Each key_name has an associated list of values."
    )

    @model_validator(mode='before')
    @classmethod
    def strip_dtype_metadata(cls, values: Any) -> Any:
        if isinstance(values, dict) and 'data' in values:
            data = values['data']
            if isinstance(data, dict) and DTYPE_METADATA_KEY in data:
                values = {**values, 'data': {k: v for k, v in data.items() if k != DTYPE_METADATA_KEY}}
        return values



class GetterResult(RootModel[List[Any]]):
    """Generic output model for dynamically created getter functions.

    Returns a list of values from the specified key_name.
    """
    model_config = ConfigDict(
        title="GetterResult",
        json_schema_extra={
            "description": "List of values from the retrieved key_name.",
        },
    )
    root: List[Any]


# ============================================================================
# Transform and Truncate Tool Models
# ============================================================================

# truncate
class TruncateInput(BaseModel):
    """Input for truncate function.

    Returns the first n elements of a list-like object.
    """
    model_config = ConfigDict(
        title="TruncateInput",
        extra="forbid",
        json_schema_extra={
            "description": "Return the first n elements of a list-like object.",
        },
    )

    truncate_array: Union[Mapping[str, List[Any]], List[Any]] = Field(
        ...,
        description="A list-like object or table to truncate."
    )
    n: int = Field(
        ...,
        description="The number of rows/elements to return. Must be non-negative.",
        ge=0
    )



class TruncateResult(BaseModel):
    """Output for truncate function.

    Returns the first n elements of the input.
    """
    model_config = ConfigDict(
        title="TruncateResult",
        extra="forbid",
        json_schema_extra={
            "description": "The first n elements of the list-like object.",
        },
    )

    data: Union[Mapping[str, List[Any]], List[Any]] = Field(..., description="Truncated data (table or list)")



# transform_data_to_substring
class TransformDataToSubstringInput(BaseModel):
    """Input for transform_data_to_substring function.

    Transform list of string values by taking substrings.
    """
    model_config = ConfigDict(
        title="TransformDataToSubstringInput",
        extra="forbid",
        json_schema_extra={
            "description": "Transform list of string values by taking substrings.",
        },
    )

    data: Mapping[str, List[Any]] = Field(
        ...,
        description="Table containing the data to be transformed."
    )
    key_name: str = Field(
        ...,
        description="Name of string valued key to transform."
    )
    start_index: int = Field(
        ...,
        description="Start index of substring."
    )
    end_index: int = Field(
        ...,
        description="End index of substring. Must be >= start_index."
    )



class TransformDataToSubstringResult(BaseModel):
    """Output for transform_data_to_substring function.

    Returns the table with transformed substring values.
    """
    model_config = ConfigDict(
        title="TransformDataToSubstringResult",
        extra="forbid",
        json_schema_extra={
            "description": "Original table with the specified key values transformed to substrings.",
        },
    )

    data: Mapping[str, Union[List[Any], Dict[str, str]]] = Field(..., description="Table with transformed substring values")



# transform_data_to_absolute_value
class TransformDataToAbsoluteValueInput(BaseModel):
    """Input for transform_data_to_absolute_value function.

    Transform numeric values into their absolute value.
    """
    model_config = ConfigDict(
        title="TransformDataToAbsoluteValueInput",
        extra="forbid",
        json_schema_extra={
            "description": "Transform numeric values into their absolute value.",
        },
    )

    data: Mapping[str, List[Any]] = Field(
        ...,
        description="Table containing the data to be transformed."
    )
    key_name: str = Field(
        ...,
        description="Name of numeric valued key to take absolute value of."
    )



class TransformDataToAbsoluteValueResult(BaseModel):
    """Output for transform_data_to_absolute_value function.

    Returns the table with transformed absolute values.
    """
    model_config = ConfigDict(
        title="TransformDataToAbsoluteValueResult",
        extra="forbid",
        json_schema_extra={
            "description": "Original table with the data associated with key_name replaced with its absolute value.",
        },
    )

    data: Mapping[str, Union[List[Any], Dict[str, str]]] = Field(..., description="Table with transformed absolute values")



# transform_data_to_datetime_part
class TransformDataToDatetimePartInput(BaseModel):
    """Input for transform_data_to_datetime_part function.

    Transform list of datetime strings into specified subpart.
    """
    model_config = ConfigDict(
        title="TransformDataToDatetimePartInput",
        extra="forbid",
        json_schema_extra={
            "description": "Transform list of datetime strings into specified subpart.",
        },
    )

    data: Mapping[str, List[Any]] = Field(
        ...,
        description="Table containing the data to be transformed."
    )
    key_name: str = Field(
        ...,
        description="Name of string valued key containing datetime values to transform."
    )
    datetime_pattern: str = Field(
        ...,
        description="ISO datetime pattern to extract (e.g., '%Y' for year, '%m' for month, '%d' for day)."
    )



class TransformDataToDatetimePartResult(BaseModel):
    """Output for transform_data_to_datetime_part function.

    Returns the table with transformed datetime parts.
    """
    model_config = ConfigDict(
        title="TransformDataToDatetimePartResult",
        extra="forbid",
        json_schema_extra={
            "description": "Original table with the specified key values transformed to the specified datetime pattern.",
        },
    )

    data: Mapping[str, Union[List[Any], Dict[str, str]]] = Field(..., description="Table with transformed datetime parts")



# ============================================================================
# Calculator Tool Models
# ============================================================================

class CalculatorOperation(str, Enum):
    """
    Available arithmetic operations for the Calculator tool:
    - 'add': Addition (input_1 + input_2)
    - 'subtract': Subtraction (input_1 - input_2)
    - 'multiply': Multiplication (input_1 * input_2)
    - 'divide': Division (input_1 / input_2)
    """
    ADD = "add"
    SUBTRACT = "subtract"
    MULTIPLY = "multiply"
    DIVIDE = "divide"


class CalculatorInput(BaseModel):
    """Input for Calculator function.

    Perform arithmetic operations on scalar or list inputs.
    Supports element-wise operations on lists.
    """
    model_config = ConfigDict(
        title="CalculatorInput",
        extra="forbid",
        json_schema_extra={
            "description": "Perform arithmetic operations on scalar or list inputs.",
        },
    )

    input_1: Union[int, float, List[Union[int, float, None]], List[int], List[float]] = Field(
        ...,
        description="First operand (scalar or list of numeric values, may contain None)"
    )
    input_2: Union[int, float, List[Union[int, float, None]], List[int], List[float]] = Field(
        ...,
        description="Second operand (scalar or list of numeric values, may contain None)"
    )
    operation: CalculatorOperation = Field(
        ...,
        description=(
            "Arithmetic operation to perform.\n\n"
            "Options:\n"
            "- 'add': Addition (input_1 + input_2)\n"
            "- 'subtract': Subtraction (input_1 - input_2)\n"
            "- 'multiply': Multiplication (input_1 * input_2)\n"
            "- 'divide': Division (input_1 / input_2)"
        )
    )


class CalculatorResult(RootModel[Union[int, float, List[int], List[float]]]):
    """Output for Calculator function.

    Returns the result of the arithmetic operation.
    """
    model_config = ConfigDict(
        title="CalculatorResult",
        json_schema_extra={
            "description": "Result of the arithmetic operation (scalar or list)",
        },
    )
    root: Union[int, float, List[int], List[float]]
