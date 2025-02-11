import tiktoken


input_filename = "/Users/daniel/dev/crowd/simple-trader/logs/2025-02-11T01-12-52/prompt.txt"
model_name = "gpt-4o"

def main():
    # Read the file contents.
    try:
        with open(input_filename, "r", encoding="utf-8") as file:
            text = file.read()
    except Exception as e:
        print(f"Error reading file '{input_filename}': {e}")
        return

    # Get the encoding for the specified model.
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except Exception:
        print(f"Model '{model_name}' not recognized. Falling back to 'gpt2' encoding.")
        encoding = tiktoken.get_encoding("gpt2")

    # Tokenize the text.
    tokens = encoding.encode(text)
    token_count = len(tokens)
    print(f"Token count for '{input_filename}': {token_count}")

if __name__ == '__main__':
    main()
