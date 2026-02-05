
import inspect
import functools
from typing import Type, Callable, Mapping, Union, List, Any, cast, Optional
from pydantic import BaseModel, RootModel, Field, ConfigDict
from enum import StrEnum

def make_enum(name: str, values: list[str]) -> StrEnum:
    new_enum = StrEnum(
        name,
        {v.upper(): v for v in values},
        type=str,
    )
    return cast(StrEnum, new_enum)


class KeySummary(BaseModel):
    name: str = Field(..., description="Name of key or attribute value")
    dtype: str = Field(..., description="data type of key value")
    first_3_values: List[Any] = Field(..., description="Values associated with this key for the first three records")


class PeekInput(BaseModel):
    model_config = ConfigDict(
        title="PeekInput",
        extra="forbid", 
        json_schema_extra={
            "description": "Peek at the contents of data. ",
        },
    )
    # Fields
    data: Mapping[str, List[Any]] = Field(..., description="Data object to peek at. ")
    key_name: List[StrEnum] = Field([], description="The key specifying which values to inspect.")

def get_peek_model(keys: List[str]):

    KeyEnum = make_enum(
        "KeyEnum",
        keys
    )

    class PeekInput(BaseModel):
        model_config = ConfigDict(
            title="PeekInput",
            extra="forbid", 
            json_schema_extra={
                "description": "Peek at the contents of data. ",
            },
        )
        # Fields
        data: Mapping[str, List[Any]] = Field(..., description="Data object to peek at. ")
        key_name: Optional[List[KeyEnum]] = Field(None, description="The key specifying which values to inspect. If not passed, all keys in the input data will be summarized. ")



class PeekResult(BaseModel):
    model_config = ConfigDict(
        title="PeekResult",
        extra="forbid", 
        json_schema_extra={
            "description": "Summary of data for given key_name (or all keys). ",
        },
    )
    # Fields
    num_records: int = Field(..., description="Number of individual records in data")
    key_details: List[KeySummary] = Field(..., description="Summary info about values for key or keys in data. ")


def peek_function(input: PeekInput) -> PeekResult:
    "Summary info about values for key or keys in data. "
    keys = input.key_name
    if keys is None:
        keys = list(input.data.keys())
    
    key_details = []
    num_records = max([len(v) for v in input.data.values()])
    for k in keys:
        vals = input.data[k]
        if len(vals) > 3:
            vals = vals[:3]
        summ = KeySummary(name=k, dtype=str(type(vals[0])), first_3_values=vals)
        key_details.append(summ)
    
    return PeekResult(num_records=num_records, key_details=key_details)


def wrap_with_models(
    func: Callable,
    *,
    docstring: str,
    input_model: Type[BaseModel],
    output_model: Union[Type[BaseModel], Type[RootModel]],
) -> Callable:
    """
    Wrap a function so that:
    - Inputs are validated via input_model
    - Output is validated via output_model
    - Docstring and function signature are replaced
    """

    @functools.wraps(func)
    def wrapper(**kwargs):

        # Validate input directly
        input_obj = input_model(**kwargs)

        # Build function arguments from validated input
        func_kwargs = {}
        for key in input_model.model_fields.keys():
            func_kwargs[key] = getattr(input_obj, key)

        result = func(**func_kwargs)


        # --- Validate output ---
        # Convert tool output (plain dict with optional _dtypes inside) to BaseModel for validation,
        # then back to dict for storage

        # 1. Already the correct model
        if isinstance(result, output_model):
            validated_result = result
        # 2. Dict → convert to BaseModel format
        elif isinstance(result, Mapping):
            # Check if output model has 'data' field (new BaseModel format)
            if 'data' in output_model.model_fields:
                # New BaseModel format: wrap entire result (including _dtypes) in data field
                # _dtypes stays inside the data dict where it belongs
                validated_result = output_model.model_validate({
                    'data': result  # Keep _dtypes inside if present
                })
            else:
                # Old format or scalar output - validate directly
                validated_result = output_model.model_validate(result)
        # 3. Primitive → single-field BaseModel
        elif len(output_model.model_fields) == 1:
            field_name = next(iter(output_model.model_fields))
            validated_result = output_model.model_validate({field_name: result})
        # 4. Otherwise, this is a contract error
        else:
            raise TypeError(
                f"{func.__name__} returned {type(result).__name__}, "
                f"which cannot be validated against {output_model.__name__}"
            )

        # Convert BaseModel back to plain dict for storage
        if hasattr(validated_result, 'model_dump'):
            # Convert BaseModel to dict
            result_dict = validated_result.model_dump()

            # If new format with 'data' field, extract the data dict (which includes _dtypes)
            if 'data' in result_dict and isinstance(result_dict['data'], dict):
                return result_dict['data']  # Return the data dict with _dtypes inside
            else:
                return result_dict
        else:
            return validated_result

    # Replace docstring
    wrapper.__doc__ = docstring

    # Replace annotations
    wrapper.__annotations__ = {
        **{k: v.annotation for k, v in input_model.model_fields.items()},
        "return": output_model,
    }

    # Replace signature
    parameters = [
        inspect.Parameter(
            name=field_name,
            kind=inspect.Parameter.KEYWORD_ONLY,
            annotation=field.annotation,
            default=inspect.Parameter.empty,
        )
        for field_name, field in input_model.model_fields.items()
    ]

    setattr(
        wrapper, 
        "__signature__", 
        inspect.Signature(
        parameters=parameters,
        return_annotation=output_model,
    )
    )

    return wrapper
