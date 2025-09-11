from pydantic import BaseModel
from typing import Dict, Any


# For each prompt in input/function_calling_tests.json,
# your program must produce a single JSON file:
# output/function_calling_name.json.
# Each object in the array must contain
# exactly the following keys:
# • str : The original natural-language request.
# • str : The name of the function to call.
# • object : all required arguments with the correct types.
class SelectedFunction(BaseModel):
    prompt: str
    fn_name: str | None
    args: Dict[str, Any]
