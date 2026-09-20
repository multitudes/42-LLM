# src/__main__.py
import argparse
import sys
from pathlib import Path

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
    OUTPUT_FILE,
    TOOLS_DEFINITION_FILE,
    extract_json_from_response,
    get_input_prompts,
    get_tool_list,
    write_output_to_file,
)


def parse_args() -> argparse.Namespace:
    """Parses command line arguments for the function calling pipeline."""
    parser = argparse.ArgumentParser(
        description="Run local tool-calling LLM pipeline."
    )
    parser.add_argument(
        "--functions_definition",
        type=str,
        default=TOOLS_DEFINITION_FILE,
        help="Path to function definitions JSON file",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=INPUT_FILE,
        help="Path to input prompts JSON file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=OUTPUT_FILE,
        help="Path to output JSON file",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point for function-calling LLM pipeline."""
    args = parse_args()
    print("Tool file path:", args.functions_definition)
    print("Input file path:", args.input)
    print("Output file path:", args.output)

    llm = Small_LLM_Model()
    tokenizer_path = Path(llm.get_path_to_tokenizer_file())
    print("Tokenizer file path:", tokenizer_path)
    merges_path = Path(llm.get_path_to_merges_file())
    print("Merge file path:", merges_path)
    # vocab_path = Path(llm.get_path_to_vocab_file())
    # print(f"Vocab file path: {vocab_path}")

    try:
        vocab, merge_ranks = initialize_tokenizer(tokenizer_path, merges_path)
        outputs = []
        tools = get_tool_list(tools_file=args.functions_definition)

        # Reverse the vocab dict for ID to token lookup
        id_to_token = {v: k for k, v in vocab.items()}

        for user_prompt in get_input_prompts(file=args.input):
            print(f"\n---\nProcessing prompt: {user_prompt}")
            prompt = create_prompt(user_prompt, tools)
            input_ids = bpe_tokenize(
                prompt,
                vocab=vocab,
                merge_ranks=merge_ranks,
            )
            answer_ids = get_answer_ids(llm, input_ids)
            llm_output = custom_decode(answer_ids, id_to_token)
            result = extract_json_from_response(
                user_prompt,
                llm_output,
                tools_file=args.functions_definition,
            )
            print(result)
            outputs.append(result)

        write_output_to_file(outputs, output_file=args.output)

    except (RuntimeError, TypeError, ValueError, FileNotFoundError) as e:
        print(f"Pipeline Execution Error: {e}", file=sys.stderr)
        sys.exit(1)

    except Exception as e:
        # Catch-all for unexpected model/CUDA/system errors
        print(f"Unexpected Fatal Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
