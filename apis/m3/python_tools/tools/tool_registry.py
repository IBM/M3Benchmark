
from copy import deepcopy
import inspect
import logging
import os
from typing import Callable

logger = logging.getLogger(__name__)

from .sql_tools import initialize_active_data, create_getter
from .slot_filling_tools import (
    filter_data,
    aggregate_data,
    sort_data,
    transform_data,
    transform_data_to_absolute_value,
    transform_data_to_datetime_part,
    transform_data_to_substring,
    truncate,
    group_data_by,
    retrieve_data,
    concatenate_data,
    select_unique_values,
    Calculator
)
from .selection_tools import fill_slots_aggregate_data, fill_slots_filter_data, fill_slots_sort_data
from .file_io_wrappers import load_csv_as_dataframe, load_from_csv, load_multiple_csvs_as_dataframes, save_as_csv
from .pydantic_models import (
    RetrieveDataInput, RetrieveDataResult,
    FilterDataInput, FilterDataResult,
    TransformDataInput, TransformDataResult,
    SortDataInput, SortDataResult,
    AggregateDataInput, AggregateDataResult,
    ConcatenateDataInput, ConcatenateDataResult,
    SelectUniqueDataInput, SelectUniqueDataResult,
    # Calculator models
    CalculatorInput,
    CalculatorResult,
    # Getter models
    GetterInput,
    GetterResult,
    # Specialized filter models
    SelectDataEqualToInput,
    SelectDataNotEqualToInput,
    SelectDataGreaterThanInput,
    SelectDataLessThanInput,
    SelectDataGreaterThanEqualToInput,
    SelectDataLessThanEqualToInput,
    SelectDataContainsInput,
    SelectDataLikeInput,
    # Specialized sort models
    SortDataAscendingInput,
    SortDataDescendingInput,
    # Specialized aggregate models
    ComputeDataMinInput,
    ComputeDataMaxInput,
    ComputeDataSumInput,
    ComputeDataMeanInput,
    ComputeDataCountInput,
    ComputeDataStdInput,
    ComputeDataArgminInput,
    ComputeDataArgmaxInput,
    # Transform and truncate models
    TruncateInput,
    TruncateResult,
    TransformDataToSubstringInput,
    TransformDataToSubstringResult,
    TransformDataToAbsoluteValueInput,
    TransformDataToAbsoluteValueResult,
    TransformDataToDatetimePartInput,
    TransformDataToDatetimePartResult,
)
from .pydantic_wrapper import wrap_with_models, PeekInput, PeekResult, peek_function

AGGREGATE_DATA="aggregate_data"
CALCULATOR="Calculator"
CONCATENATE_DATA="concatenate_data"
FILTER_DATA="filter_data"
GROUP_DATA_BY="group_data_by"
INITIALIZE_ACTIVE_DATA="initialize_active_data"
RETRIEVE_DATA="retrieve_data"
SELECT_UNIQUE_VALUES="select_unique_values"
SORT_DATA="sort_data"
TRANSFORM_DATA="transform_data"

PEEK_FCN="peek_fcn"


def add_read_from_file_decorator(apis: dict, temp_cache_location: str) -> dict:
    for api_name, api in apis.items():
        if api_name == INITIALIZE_ACTIVE_DATA:
            apis[api_name] = save_as_csv(deepcopy(api), tempfile_location=temp_cache_location)
        elif api_name.startswith("get_"):
            signature = inspect.signature(api)
            param_names = [p.name for p in signature.parameters.values()]
            assert 'data' in param_names
            apis[api_name] = load_from_csv(deepcopy(api))
        elif api_name in [AGGREGATE_DATA, RETRIEVE_DATA] or api_name.startswith("compute_"):
            apis[api_name] = load_from_csv(deepcopy(api))
        elif api_name == CONCATENATE_DATA:
            apis[api_name] = load_multiple_csvs_as_dataframes(deepcopy(api), tempfile_location=temp_cache_location)
        else:
            signature = inspect.signature(api)
            for param in signature.parameters.values():
                if param.name == 'data':
                    apis[api_name] = load_csv_as_dataframe(deepcopy(api), tempfile_location=temp_cache_location)
                    break
    return apis

SLOT_FILLING_TOOLS = [
    {
        "name": AGGREGATE_DATA,
        "tool": aggregate_data,
        "docstring": "Apply a specified aggregation operation to the values in a single key of the input data. ",
        "input_model": AggregateDataInput,
        "output_model": AggregateDataResult,
    },
    {
        "name": CALCULATOR,
        "tool": Calculator,
        "docstring": "Perform arithmetic operations on scalar or list inputs. Supports element-wise operations on lists and scalar-scalar operations.",
        "input_model": CalculatorInput,
        "output_model": CalculatorResult,
    },
    {
        "name": CONCATENATE_DATA,
        "tool": concatenate_data,
        "docstring": "Concatenates two data tables along axis 0 (rows). ",
        "input_model": ConcatenateDataInput,
        "output_model": ConcatenateDataResult,
    },
    {
        "name": FILTER_DATA,
        "tool": filter_data,
        "docstring": "This function applies a filter on the given key of the input data, based on the provided condition and value. ",
        "input_model": FilterDataInput,
        "output_model": FilterDataResult,
    },
    {
        "name": INITIALIZE_ACTIVE_DATA,
        "tool": initialize_active_data,
        "docstring": "fakestring",
        "input_model": FilterDataInput,
        "output_model": FilterDataResult,
    },
    {
        "name": RETRIEVE_DATA,
        "tool": retrieve_data,
        "docstring": "This function selects the values corresponding to the given key or keys from the input data. ",
        "input_model": RetrieveDataInput,
        "output_model": RetrieveDataResult,
    },
    {
        "name": SELECT_UNIQUE_VALUES,
        "tool": select_unique_values,
        "docstring": "Returns only the unique values from the input array. ",
        "input_model": SelectUniqueDataInput,
        "output_model": SelectUniqueDataResult,
    },
    {
        "name": SORT_DATA,
        "tool": sort_data,
        "docstring": "Sort the input data in either descending (default) or ascending order. ",
        "input_model": SortDataInput,
        "output_model": SortDataResult,
    },
    {
        "name": TRANSFORM_DATA,
        "tool": transform_data,
        "docstring": "This function applies the specified transformation to the given key of the input data. ",
        "input_model": TransformDataInput,
        "output_model": TransformDataResult,
    },
    # {
    #     "name": PEEK_FCN,
    #     "tool": peek_function,
    #     "docstring": "This function returns a summary of the given data. ",
    #     "input_model": PeekInput,
    #     "output_model": PeekResult,
    # },
]


class SlotFillingTools():
    def __init__(self, io_cache: str = ".", use_io_wrappers: bool = False, use_pydantic_signatures: bool = True):
        self.io_cache = io_cache
        self.use_io_wrappers = use_io_wrappers
        self.use_pydantic_wrappers = use_pydantic_signatures
        self.tools = {}

        if use_io_wrappers:
            if not os.path.isdir(io_cache):
                logger.info(f"Creating cache dir: {io_cache}")
                os.makedirs(io_cache, exist_ok=True)

        for spec in SLOT_FILLING_TOOLS:
            tool = spec['tool']
            if use_io_wrappers:
                tool = list(add_read_from_file_decorator({spec['name']: tool}, self.io_cache).values())[0]
            if use_pydantic_signatures and spec['name'] != INITIALIZE_ACTIVE_DATA: # Don't use initialize_active_data
                tool = wrap_with_models(
                    tool, 
                    docstring=spec['docstring'], 
                    input_model=spec['input_model'],
                    output_model=spec['output_model'],
                )
            self.tools[spec['name']] = tool


    def get_toolbox_with_schema(self, key_names_and_descriptions: list[dict]) -> dict[str, Callable]:
        return self.tools

TRANSFORM_DATA_TO_SUBSTRING="transform_data_to_substring"
TRANSFORM_DATA_TO_ABSOLUTE_VALUE="transform_data_to_absolute_value"
TRANSFORM_DATA_TO_DATETIME_PART="transform_data_to_datetime_part"
TRUNCATE="truncate"

# Base tools dict used by SelectionTools class
SELECTION_BASE_TOOLS = {
    CALCULATOR : Calculator,
    CONCATENATE_DATA : concatenate_data,
    # GROUP_DATA_BY : group_data_by,
    INITIALIZE_ACTIVE_DATA : initialize_active_data,
    SELECT_UNIQUE_VALUES : select_unique_values,
    TRANSFORM_DATA_TO_SUBSTRING : transform_data_to_substring,
    TRANSFORM_DATA_TO_ABSOLUTE_VALUE : transform_data_to_absolute_value,
    TRANSFORM_DATA_TO_DATETIME_PART : transform_data_to_datetime_part,
    TRUNCATE : truncate
}

# Tool specifications for specialized selection tools (with Pydantic models)
SELECTION_TOOLS = [
    # Filter tools (8)
    {"name": "select_data_equal_to", "input_model": SelectDataEqualToInput, "output_model": FilterDataResult},
    {"name": "select_data_not_equal_to", "input_model": SelectDataNotEqualToInput, "output_model": FilterDataResult},
    {"name": "select_data_greater_than", "input_model": SelectDataGreaterThanInput, "output_model": FilterDataResult},
    {"name": "select_data_less_than", "input_model": SelectDataLessThanInput, "output_model": FilterDataResult},
    {"name": "select_data_greater_than_equal_to", "input_model": SelectDataGreaterThanEqualToInput, "output_model": FilterDataResult},
    {"name": "select_data_less_than_equal_to", "input_model": SelectDataLessThanEqualToInput, "output_model": FilterDataResult},
    {"name": "select_data_contains", "input_model": SelectDataContainsInput, "output_model": FilterDataResult},
    {"name": "select_data_like", "input_model": SelectDataLikeInput, "output_model": FilterDataResult},
    # Sort tools (2)
    {"name": "sort_data_ascending", "input_model": SortDataAscendingInput, "output_model": SortDataResult},
    {"name": "sort_data_descending", "input_model": SortDataDescendingInput, "output_model": SortDataResult},
    # Aggregate tools (8)
    {"name": "compute_data_min", "input_model": ComputeDataMinInput, "output_model": AggregateDataResult},
    {"name": "compute_data_max", "input_model": ComputeDataMaxInput, "output_model": AggregateDataResult},
    {"name": "compute_data_sum", "input_model": ComputeDataSumInput, "output_model": AggregateDataResult},
    {"name": "compute_data_mean", "input_model": ComputeDataMeanInput, "output_model": AggregateDataResult},
    {"name": "compute_data_count", "input_model": ComputeDataCountInput, "output_model": AggregateDataResult},
    {"name": "compute_data_std", "input_model": ComputeDataStdInput, "output_model": AggregateDataResult},
    {"name": "compute_data_argmin", "input_model": ComputeDataArgminInput, "output_model": AggregateDataResult},
    {"name": "compute_data_argmax", "input_model": ComputeDataArgmaxInput, "output_model": AggregateDataResult},
    # Utility tools (3)
    {"name": CALCULATOR, "input_model": CalculatorInput, "output_model": CalculatorResult},
    {"name": CONCATENATE_DATA, "input_model": ConcatenateDataInput, "output_model": ConcatenateDataResult},
    {"name": SELECT_UNIQUE_VALUES, "input_model": SelectUniqueDataInput, "output_model": SelectUniqueDataResult},
    # Transform tools (4)
    {"name": TRUNCATE, "input_model": TruncateInput, "output_model": TruncateResult},
    {"name": TRANSFORM_DATA_TO_SUBSTRING, "input_model": TransformDataToSubstringInput, "output_model": TransformDataToSubstringResult},
    {"name": TRANSFORM_DATA_TO_ABSOLUTE_VALUE, "input_model": TransformDataToAbsoluteValueInput, "output_model": TransformDataToAbsoluteValueResult},
    {"name": TRANSFORM_DATA_TO_DATETIME_PART, "input_model": TransformDataToDatetimePartInput, "output_model": TransformDataToDatetimePartResult},
]

class SelectionTools():
    def __init__(self, io_cache: str = ".", use_io_wrappers: bool = True, use_pydantic_signatures: bool = False):
        self.io_cache = io_cache
        self.use_io_wrappers = use_io_wrappers
        self.use_pydantic_signatures = use_pydantic_signatures
        self.tools = None
        self.base_tools = deepcopy(SELECTION_BASE_TOOLS)

        # Bind function args to finish base tools
        self.base_tools.update(fill_slots_filter_data())
        self.base_tools.update(fill_slots_sort_data())
        self.base_tools.update(fill_slots_aggregate_data())

        # Apply Pydantic wrappers if requested (before IO wrappers)
        if use_pydantic_signatures:
            # Create lookup dict for model specs
            selection_model_specs = {spec['name']: spec for spec in SELECTION_TOOLS}

            # Wrap specialized functions with Pydantic models
            for tool_name, tool_func in self.base_tools.items():
                if tool_name in selection_model_specs:
                    spec = selection_model_specs[tool_name]
                    self.base_tools[tool_name] = wrap_with_models(
                        tool_func,
                        docstring=tool_func.__doc__,
                        input_model=spec['input_model'],
                        output_model=spec['output_model'],
                    )

        if use_io_wrappers:
            if not os.path.isdir(io_cache):
                logger.info(f"Creating cache dir: {io_cache}")
                os.makedirs(io_cache, exist_ok=True)
            wrapped_fcns = add_read_from_file_decorator(self.base_tools, self.io_cache)
            self.tools = wrapped_fcns
        else:
            self.tools = self.base_tools

    def get_toolbox_with_schema(self, key_names_and_descriptions: list[dict]) -> dict[str, Callable]:
        template_api_calls = {}
        for row in key_names_and_descriptions:
            getter = create_getter(row['key_name'], row['description'], row['dtype'])

            # Apply Pydantic wrapper if requested (before IO wrapper)
            if self.use_pydantic_signatures:
                getter = wrap_with_models(
                    getter,
                    docstring=getter.__doc__ or "",
                    input_model=GetterInput,
                    output_model=GetterResult,
                )

            template_api_calls[getter.__name__] = getter
        if self.use_io_wrappers:
            template_api_calls = add_read_from_file_decorator(template_api_calls, self.io_cache)

        tools_with_getters = dict(template_api_calls, **self.tools)
        return tools_with_getters
