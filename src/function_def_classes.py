from pydantic import BaseModel
from typing import List, Dict

# Pydantic classes to define the input function schema
# Ex of input an array of :
#   {
#     "fn_name": "fn_add_numbers",
#     "args_names": [
#       "a",
#       "b"
#     ],
#     "args_types": {
#       "a": "float",
#       "b": "float"
#     },
#     "return_type": "float"
#   },
class FunctionDef(BaseModel):
    fn_name: str
    args_names: List[str]
    args_types: Dict[str, str]
    return_type: str
