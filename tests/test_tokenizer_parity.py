import json
import pytest
import torch

from llm_sdk import Small_LLM_Model
from src.bpe_tokenizer import bpe_tokenize, custom_decode
