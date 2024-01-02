#!/usr/bin/env python

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


def get_args():
    from argparse import ArgumentParser
    parser = ArgumentParser()

    parser.add_argument('--model_name_or_path', type=str, required=True)

    return parser.parse_args()

def main():
    args = get_args()

    print(f"Loading model from {args.model_name_or_path} ...")

    model_path = f"/opt/local/llm_models/huggingface.co/{args.model_name_or_path}"
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, config=config, trust_remote_code=True)

    print(f"Loading tokenizer from {args.model_name_or_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    print("Converting model to fp16 ...")
    model = model.half()

    fp16_model_path = f"{model_path}-fp16"

    print(f"Saving model to {fp16_model_path} ...")
    model.save_pretrained(fp16_model_path)
    print(f"Saving tokenizer to {fp16_model_path} ...")
    tokenizer.save_pretrained(fp16_model_path)

    print("Done.")

if __name__ == '__main__':  
    main()