# src/__main__.py
import sys

from llm_sdk import Small_LLM_Model

from .bpe_tokenizer import (
    bpe_tokenize,
    create_prompt,
    custom_decode,
    get_answer_ids,
    initialize_tokenizer,
)
from .utils import (
    INPUT_FILE,
    extract_json_from_response,
    get_input_prompts,
    get_tool_list,
    write_output_to_file,
)


def main(input_file: str = INPUT_FILE) -> None:
    """
    Main entry point for function-calling LLM pipeline.

    Args:
        input_file: Path to the prompts JSON file. Defaults to
            "exercise_input/function_calling_tests.json".

    Returns:
        None

    """
    try:
        llm = Small_LLM_Model()
        vocab_path = llm.get_path_to_tokenizer_file()
        print("Vocabulary file path:", vocab_path)
        vocab, merge_ranks = initialize_tokenizer(vocab_path)
        outputs = []
        tools = get_tool_list()

        # Reverse the vocab dict for ID to token lookup
        id_to_token = {v: k for k, v in vocab.items()}

        for user_prompt in get_input_prompts(input_file):
            print(f"\n\nProcessing prompt: {user_prompt}")
            prompt = create_prompt(user_prompt, tools)
            # input_ids = llm._encode(final_prompt).tolist()[0]
            input_ids = bpe_tokenize(
                prompt, vocab=vocab, merge_ranks=merge_ranks,
            )
            answer_ids = get_answer_ids(llm, input_ids)
            # llm_output = llm._decode(answer_ids)
            llm_output = custom_decode(answer_ids, id_to_token)
            result = extract_json_from_response(
                user_prompt, llm_output,
            )
            outputs.append(result)

        write_output_to_file(outputs)

    except RuntimeError as e:
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
