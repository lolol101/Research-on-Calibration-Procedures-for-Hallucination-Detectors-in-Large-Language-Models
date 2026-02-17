def retrieve_answer_token_index(tokens):
    for i in range(len(tokens) - 1, 0, -1):
        if tokens[i]["token"].isdigit():
            return i


# field1_text = '"short_chain_of_thoughts": "'
# field2_text = '"answer": "'
def retrieve_reasoning_tokens_range(tokens, start="<think>", end="</think>"):
    start_index, end_index = -1, -1
    tmp_text = ""
    for i in range(len(tokens)):
        if start in tmp_text:
            start_index = i
            break
        tmp_text += tokens[i]["token"]

    tmp_text = ""

    for i in range(len(tokens) - 1, 0, -1):
        tmp_text = tokens[i]["token"] + tmp_text
        if end in tmp_text:
            end_index = i
            break

    return (start_index, end_index)