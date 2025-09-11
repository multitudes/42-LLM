from pydantic import BaseModel
from typing import List, Dict


class FunctionDef(BaseModel):
    """
    Schema for defining a function signature. This is what we get from
    input/functions_definition.json. This will be passed to the model
    in the prompt to let it know what functions are available to call.

    Example input:
    {
        "fn_name": "fn_add_numbers",
        "args_names": ["a", "b"],
        "args_types": {
            "a": "float",
            "b": "float"
        },
        "return_type": "float"
    }

    Attributes:
        fn_name: Name of the function.
        args_names: Ordered list of argument names.
        args_types: Mapping of argument names to their types.
        return_type: The return type of the function.
    """
    fn_name: str
    args_names: List[str]
    args_types: Dict[str, str]
    return_type: str
