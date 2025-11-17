from PIL import Image
import torch
import fire 
from processing import PaliGemmaProcessor
from paligemma import PaliGemmaForConditionalGeneration
from gemma import KVCache
from utils import load_hf_model

def get_model_inputs(
        processor: PaliGemmaProcessor,
        prompt: str,
        image_file_path: str,
        device: str,
):
    image = Image.open(image_file_path)
    image = [image]
    prompt = [prompt]
    model_inputs = processor(images=image, texts=prompt)
    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
    return model_inputs

def sample_top_p(probs: torch.Tensor, p: float):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort = probs_sort / probs_sort.sum(dim=-1, keepdim=True)
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token

def test_inference(
        model: PaliGemmaForConditionalGeneration,
        processor: PaliGemmaProcessor,
        device: str,
        prompt: str,
        image_file_path: str,
        max_token_to_generate: int,
        temperature: float,
        top_p: float,
        do_sample: bool,
):
    model_inputs = get_model_inputs(processor, prompt, image_file_path, device)
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"]
    pixel_values = model_inputs["pixel_values"]

    kv_cache = KVCache()

    stop_token = processor.tokenizer.eos_token_id
    generated_tokens = []

    for _ in range(max_token_to_generate):
        outputs = model(
            input_ids=input_ids,
            pixel_value=pixel_values,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
        )
        kv_cache = outputs['kv_cache']
        next_token_logits = outputs['logits'][:, -1, :]

        if do_sample:
            prob_next_token = torch.softmax(next_token_logits / temperature, dim=-1)
            next_token = sample_top_p(prob_next_token, top_p)
        else:
            next_token = torch.argmax(next_token_logits, dim=-1)

        assert next_token.size() == (1, 1), f"Next token size should be (1, 1), got {next_token.size()}"
        next_token = next_token.squeeze(0)
        generated_tokens.append(next_token)
        if next_token.item() == stop_token:
            break   

        input_ids = next_token.unsqueeze(-1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones((1, 1)).to(device)],
            dim=-1,
        )
    generated_tokens = torch.cat(generated_tokens, dim=-1)
    generated_text = processor.tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )
    print("Generated text:\n")
    print(prompt + generated_text)

def main(
        model_path: str = None,
        prompt: str = None,
        image_file_path: str = None,
        max_token_to_generate: int = 100,
        temperature: float = 0.8,
        top_p: float = 0.9,
        do_sample: bool = True,
        only_cpu: bool =False,
):
    device = "cpu"

    if not only_cpu:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"

    print("Device in use:", device)

    print("Loading model...")
    model, tokenizer = load_hf_model(model_path, device)
    model = model.to(device).eval()

    num_image_tokens = model.config.vision_config.num_image_tokens
    image_size = model.config.vision_config.image_size
    processor = PaliGemmaProcessor(tokenizer, num_image_tokens, image_size)

    print("running inference...")
    with torch.no_grad():
        test_inference(
            model,
            processor,
            device,
            prompt,
            image_file_path,
            max_token_to_generate,
            temperature,
            top_p,
            do_sample,
        )

if __name__ == "__main__":
    fire.Fire(main)