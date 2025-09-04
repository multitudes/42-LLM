# src/__main__.py
import json
import string
from time import sleep
from .utils import get_prompts, get_functions
from llm_sdk import Small_LLM_Model


# OLLAMA_URL_API = "http://localhost:11434/api/chat"


# The tokens that I need are
# "[": 58
# " (1) as a delimiter before and after each function name.
# ",": 11,
# "]": 60.
# "<unk" and "<unk>": Unknown token, used when a word or character is not in the vocabulary.
# "<s>": Start-of-sequence token, marks the beginning of a sentence or input.
# "</s" and "</s>": End-of-sequence token, marks the end of a sentence or input.
# "<unk": 128243,
# "<unk>": 128244,
# "<s>": 128245,
# "</s": 128246,
# "</s>": 128247,
# The token "Ċ": 198 in your vocab.json typically represents a newline character (\n) or a line break


def preprocess_word(word):
    # Remove punctuation
    print(word)
    word = word.strip(string.punctuation)
    # Add Ġ to indicate space as seen in the vocab file
    if word and not word.isnumeric():
        return "Ġ" + word
    return word


def main():
    input_ids = []
    llm = Small_LLM_Model(model_name="Qwen/Qwen3-0.6B")
    vocab_path = llm.get_path_to_vocabulary_json()
    with open(vocab_path, "r") as f:
        vocabs = json.load(f)
    prompts = get_prompts()
    for prompt in prompts:
        # llm = Small_LLM_Model(model_name="Qwen/Qwen3-0.6B")
        logits = []
        usertext = f"'{prompt}'"
        system_msg = "You are a tool selector. Reply ONLY with the correct index as a digit 0–6."

        user_msg = f"""{usertext}.

        Here are the functions:
        0: add numbers
        1: get square root
        2: greet
        3: is even
        4: multiply numbers
        5: reverse string
        6: substitute string with regex

        Reply ONLY with the integer index.
        """

        prompt = (
            "<|im_start|>system\n" + system_msg + "<|im_end|>\n"
            "<|im_start|>user\n" + user_msg + "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        input_ids = llm._encode(prompt)

        # prompt = llm._tokenizer.apply_chat_template(
        #     [
        #         {"role": "system", "content": "You are a tool selector. You must only reply with the correct index as an integer."},
        #         {"role": "user", "content": f"{usertext}.\n\nHere are the functions:\n0: add numbers\n1: get square root\n2: greet\n3: is even\n4: multiply numbers\n5: reverse string\n6: substitute string with regex\n\nReply ONLY with the integer index."}
        #     ],
        #     tokenize=False,
        #     add_generation_prompt=True
        # )
        input_ids = llm._encode(prompt).tolist()[0]
        # input_ids += llm._encode(' Reply ONLY with a single integer (the INDEX of the correct function in the list below, starting from 0). DO NOT WRITE ANYTHING ELSE JUST THE INDEX. ').tolist()[0]
        # input_ids = [20841, 26687, 448, 264, 3175, 7546, 320, 1782, 1922, 315, 279, 4396, 729, 304, 279, 1140, 3685, 11, 5916, 504, 220, 15, 568, 3155, 537, 3270, 4113, 770, 13]
        # input_ids += llm._encode('["function index 0: add numbers", "function index 1: get square root", "function index 2: greet", "function index 3: is even", "function index 4: multiply numbers", "function index 5: reverse string", "function index 6: substitute string with regex "]').tolist()[0]
        # print(input_ids)
        input = llm._decode(input_ids)
        print(input)
    
    # words = prompt.split()
    # print(words)
    # tokens = [preprocess_word(word) for word in words]
    # print(tokens)
    
    # input_ids += [vocabs.get(token, vocabs.get("<unk>", 0)) for token in tokens]
    # append the functions to the input_ids
    # input_ids.append(256)  # space before the array
    # input_ids.append(58)
    # print("[")
    # for fn in get_functions():
    #     input_ids.append(1)  # '"' delimiter
    #     print("\"")
    #     # Remove 'fn' prefix and underscores, then preprocess each word
    #     fn_name = fn.fn_name
    #     fn_words = fn_name.split('_')
    #     if fn_words[0] == 'fn':
    #         print("removed fn")
    #         fn_words = fn_words[1:]
    #     fn_tokens = [preprocess_word(word) for word in fn_words]
    #     for token in fn_tokens:
    #         input_ids.append(vocabs.get(token, vocabs.get("<unk>", 0)))
    #     input_ids.append(1)  # '"' delimiter
    #     input_ids.append(11)  # ',' separator
    # input_ids.append(60)
    # input_ids.append(128247)  # end of sequence token

    # Reverse the vocab dict for ID to token lookup
    # id_to_token = {v: k for k, v in vocabs.items()}
    # for id in input_ids:
    #     # print("Token ID:", id)
    #     print(id_to_token.get(id, "<unk>"), end="")

        # output = llm._decode(input_ids)
        # print("\nDecoded output:", output)
        answer_ids = []
        for _ in range(140):  # Generate 20 tokens
            # print(input_ids)
            logits = llm.get_logits_from_input_ids(input_ids)
            next_token_id = logits.index(max(logits))
            # print("Next token ID:", next_token_id)
            input_ids.append(next_token_id)
            answer_ids.append(next_token_id)
        # print(f"from the vocab: {id_to_token.get(next_token_id, '<unk>')}")

        output = llm._decode(answer_ids)
        print("\nDecoded output:", output, end="\n")
        
    # outputs = []
    # prompts = get_prompts()
    # for prompt in prompts:
    #     print(f"Prompt: {prompt}")
    #     messages = [
    #         Message(
    #             role="user",
    #             content=f"{prompt}.  "
    #             "Reply ONLY in JSON with this exact format: "
    #             '{"name": <function_name or null>, "arguments": <dict of arguments>}. '
    #             "If no function is called, set 'name' to null and 'arguments' to {}. Do not include any other fields or text."
    #         )
    #     ]
    #     tools = get_tools()  # Should return a list of dicts
    #     data = OllamaRequest(
    #         model="qwen3:0.6b",
    #         messages=messages,
    #         tools=tools,
    #         stream=False,
    #         think=False
    #     )

    #     result = call_ollama_api(data.dict())

    #     print(result)
    #     tool_calls = result["message"].get("tool_calls")
    #     if tool_calls and len(tool_calls) > 0:
    #         tool_call = tool_calls[0]
    #         fn_name = tool_call["function"]["name"]
    #         args = tool_call["function"]["arguments"]
    #         print(fn_name)
    #         print(args)
    #     else:
    #         content = result["message"].get("content")
    #         if content:
    #             try:
    #                 parsed = json.loads(content)
    #                 fn_name = parsed["name"]
    #                 args = parsed["arguments"]
    #                 print(fn_name)
    #                 print(args)
    #             except (TypeError, json.JSONDecodeError):
    #                 print("Plain text reply from LLM:")
    #                 print(content)
    #                 fn_name = None
    #                 args = {}

    #     outputs.append(NameFunctionCall(
    #         prompt=prompt,
    #         fn_name=fn_name,
    #         args=args
    #     ))

    # os.makedirs("output", exist_ok=True)
    # with open("output/name_function_calls.json", "w") as f:
    #     json.dump([o.dict() for o in outputs], f)


if __name__ == "__main__":
    main()
