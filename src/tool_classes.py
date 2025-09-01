from pydantic import BaseModel
from typing import List, Dict

# Pydantic classes to define the different parts of the tool schema:
# "parameters": {
#     "type": "object",
#     "properties": {
#         "city": {
#             "type": "string",
#             "description": "The city to get the weather for"
#         }
#     },
#     "required": ["city"]
# }


class ToolParameter(BaseModel):
    type: str = "object"
    properties: Dict[str, Dict[str, str]]
    required: List[str] = []


# "function": {
#     "name": "get_weather",
#     "description": "Get the weather in a given city",
#     "parameters": {
#             "type": "object",
#             "properties": {
#                 "city": {
#                     "type": "string",
#                     "description": "The city to get the weather for"
#                 }
#             },
#         "required": ["city"]
#     }
# }
class ToolFunction(BaseModel):
    name: str
    description: str
    parameters: ToolParameter


# {
#     "type": "function",
#     "function": {
#         "name": "get_weather",
#         "description": "Get the weather in a given city",
#         "parameters": {
#             "type": "object",
#             "properties": {
#                 "city": {
#                     "type": "string",
#                     "description": "The city to get the weather for"
#                 }
#             },
#             "required": ["city"]
#         }
#     }
# }
class Tool(BaseModel):
    type: str = "function"
    function: ToolFunction
