# added top_k and top_p parameters to generation functions
import torch
import tiktoken

def generate_with_stop(
    model,
    input_indices: torch.Tensor,
    max_new_tokens: int,
    stop_ids: set[int],
    temperature: float,
    use_cache: bool = True,
    reset_cache: bool = False,
    top_k=None,          # ### NEW ###
    top_p=None,          # ### NEW ###
):
    """
    Python側で stop を行う generator
    """
    for token_id in model.generate(
        input_indices=input_indices,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        use_cache=use_cache,
        reset_cache=reset_cache,
        top_k=top_k,      # ### NEW ###
        top_p=top_p,      # ### NEW ###
    ):
        if token_id in stop_ids:
            break
        yield token_id


def infer_stream(
    model,
    prompt: str,
    max_new_tokens=256,
    stop_ids={50256},
    temperature=0.7,
    use_cache=True,
    reset_cache=True,
    top_k=None,          # ### NEW ###
    top_p=None,          # ### NEW ###
):
    tokenizer = tiktoken.get_encoding("gpt2")
    input_ids_list = [tokenizer.encode(prompt, allowed_special="all")]
    current_device = "cuda" if torch.cuda.is_available() else "cpu"
    input_ids = torch.tensor(input_ids_list, dtype=torch.long, device=current_device)

    acc = []
    last = ""

    for tid in generate_with_stop(
        model,
        input_indices=input_ids,
        max_new_tokens=max_new_tokens,
        stop_ids=stop_ids,
        temperature=temperature,
        use_cache=use_cache,
        reset_cache=reset_cache,
        top_k=top_k,      # ### NEW ###
        top_p=top_p,      # ### NEW ###
    ):
        acc.append(tid)
        text = tokenizer.decode(acc)
        if not text.endswith("�"):
            new = text[len(last):]
            if new:
                yield new
                last = text
