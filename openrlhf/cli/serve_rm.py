import argparse
import re
import json
import jsonlines
from datasets import load_from_disk
import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from math_verify import parse, verify

from openrlhf.models import get_llm_for_sequence_regression
from openrlhf.utils import get_tokenizer
from openrlhf.utils.logging_utils import init_logger
from openrlhf.utils.check.qwen_equal import math_equal
from multiprocessing import Pool
from transformers import AutoTokenizer
logger = init_logger(__name__)
import signal
from contextlib import contextmanager

class TimeoutException(Exception):
    pass

@contextmanager
def timeout(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

def math_equal2(gold, answer):
    try:
        gold=parse(gold)
        answer=parse(answer)
        return verify(gold, answer)
    except:
        return False
    
def strip_sequence(text, pad_token, eos_token):
    pad_token_escaped = re.escape(pad_token)
    eos_token_escaped = re.escape(eos_token)

    pattern = f"^({eos_token_escaped}|{pad_token_escaped})+"
    text = re.sub(pattern, "", text)

    pattern = f"({eos_token_escaped}|{pad_token_escaped})+$"
    text = re.sub(pattern, "", text)
    return text

                
class RewardModelProxy:
    def __init__(self, args):
        self.reward_model = get_llm_for_sequence_regression(
            args.reward_pretrain,
            "reward",
            normalize_reward=args.normalize_reward,
            use_flash_attention_2=args.flash_attn,
            bf16=args.bf16,
            load_in_4bit=args.load_in_4bit,
            value_head_prefix=args.value_head_prefix,
            device_map="auto",
        )
        self.reward_model.eval()

        self.tokenizer = get_tokenizer(
            args.reward_pretrain, self.reward_model, "left", None, use_fast=not args.disable_fast_tokenizer
        )
        self.max_length = args.max_len
        self.batch_size = args.batch_size

    def get_reward(self, queries):
        if self.batch_size is None:
            batch_size = len(queries)
        else:
            batch_size = self.batch_size

        # remove pad_token
        for i in range(len(queries)):
            queries[i] = (
                strip_sequence(queries[i], self.tokenizer.pad_token, self.tokenizer.eos_token)
                + self.tokenizer.eos_token
            )
        logger.info(f"queries[0]: {queries[0]}")

        scores = []
        # batch
        with torch.no_grad():
            for i in range(0, len(queries), batch_size):
                inputs = self.tokenize_fn(
                    queries[i : min(len(queries), i + batch_size)], device=self.reward_model.device
                )
                r = self.reward_model(inputs["input_ids"], inputs["attention_mask"])
                r = r.tolist()
                scores.extend(r)
        return scores

    def tokenize_fn(self, texts, device):
        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            add_special_tokens=False,
            max_length=self.max_length,
            padding=True,
            truncation=True,
        )
        return {k: v.to(device) for k, v in batch.items()}


class RuleBasedRMProxy:
    def __init__(self, args):
        self.args = args
        self.prompt2answer = {}
        
        dataset = load_from_disk(args.data_path)
        train_list = list(dataset["train"])
        validation_list = list(dataset["test"])
        
        for line in train_list:
            self.prompt2answer[line['context'].strip()] = line['answer']
        for line in validation_list:
            self.prompt2answer[line['context'].strip()] = line['answer']
        
        self.timeout_seconds=2
        self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
        
        self.chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
        self.english_pattern = re.compile(r'[a-zA-Z]')
        self.boxed_pattern = re.compile(r"\\boxed\{((?:[^{}]|\\{|\\}|(?:\{(?:[^{}]|\\{|\\}|(?:\{(?:[^{}]|\\{|\\}|(?:\{[^{}]*\}))*\}))*\}))*\})")
        self.valid_char_pattern = re.compile(r'[a-zA-Z0-9\s\.,!?"\'\(\)\{\}\[\]_\-+=<>/@#$%^&*\\|:;~`\u2200-\u22FF]')
        self.repeat_pattern = re.compile(r'(.{5,}?)\1{4,}')

    def check_mixed_languages(self, text):
        chinese_chars = len(self.chinese_pattern.findall(text))
        english_chars = len(self.english_pattern.findall(text))
        return chinese_chars >= 20 and english_chars >= 20
    
    def check_garbled_characters(self, text):
        valid_chars = self.valid_char_pattern.sub('', text)
        if not text: 
            return False
        invalid_ratio = len(valid_chars) / len(text)
        return invalid_ratio > 0.3
    
    def has_repeated_patterns(self, text):
        return bool(self.repeat_pattern.search(text))
    
    def correctness_score(self, prompt, response):
        matches = self.boxed_pattern.findall(response)
        if not matches:
            return -1.0
        
        pred = matches[-1][:-1]
        if prompt not in self.prompt2answer:
            return -1.0
        
        return 1.0 if math_equal2(self.prompt2answer[prompt], pred) else -0.5
    
    def split_and_score(self, query):
        try:
            with timeout(self.timeout_seconds):
                if args.template_type=="qwen":
                    prompt=query.split("<|im_end|>\n<|im_start|>user\n")[-1].split("<|im_end|>\n<|im_start|>assistant\n")[0].strip()
                    response=query.split("<|im_end|>\n<|im_start|>assistant\n")[-1]
                    if "<|im_end|>" not in response and "<|endoftext|>" not in response:
                        return -1.0
                    response=query.split("<|im_end|>\n<|im_start|>assistant\n")[-1].split("<|im_end|>")[0].split("<|endoftext|>")[0].strip()
                elif args.template_type=="deepseek":
                    prompt = query.split("<｜User｜>")[-1].split("<｜Assistant｜>")[0].strip()
                    prompt = prompt.replace("Please reason step by step, and put your final answer within \\boxed{}", "").strip()
                    response = query.split("<｜Assistant｜>")[-1].strip()
                    
                if "\\boxed" not in response or response.count("\\boxed")>=5:
                    return -1.0
                
                # if self.check_mixed_languages(response):
                #     return -1.0
                    
                if self.check_garbled_characters(response):
                    return -1.0
                
                if self.has_repeated_patterns(response):
                    return -1.0
                
                return self.correctness_score(prompt, response)
                
        except TimeoutException:
            logger.warning("Processing timed out")
            return -1.0
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return -1.0
    
    def get_reward(self, queries):
        scores = []
        for query in queries:
            score = self.split_and_score(query)
            scores.append(score)
        return scores
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="rule")
    # RuleBasedRM Parameters
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--max_gen_len", type=int)
    parser.add_argument("--template_type", type=str, default="qwen", choices=["qwen", "deepseek"])
    # Reward Model
    parser.add_argument("--data_path", type=str, default=None)    # for 
    parser.add_argument("--reward_pretrain", type=str, default=None, help="HF model name or path")
    parser.add_argument("--normalize_reward", action="store_true", default=False, help="Enable Reward Normazation")
    parser.add_argument("--value_head_prefix", type=str, default="score")
    parser.add_argument("--max_len", type=int, default="2048")

    parser.add_argument("--port", type=int, default=5000, help="Port number for the server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="IP for the server")

    # Performance
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--bf16", action="store_true", default=False, help="Enable bfloat16")
    parser.add_argument("--flash_attn", action="store_true", default=False, help="Enable FlashAttention2")
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)
    parser.add_argument("--batch_size", type=int, default=None)

    args = parser.parse_args()

    # server
    if args.mode=="model":
        reward_model = RewardModelProxy(args)
    else:
        reward_model = RuleBasedRMProxy(args)
    
    # test_case="<im_start>\nsystem\nnihao<|im_end|>\n<|im_start|>user\n1+1<|im_end|>\n<|im_start|>assistant\n1+1=\\boxed{2}<im_end>"
    # reward=reward_model.get_reward([test_case for _ in range(4)])
    # print(reward)
    # exit()
    app = FastAPI()

    @app.post("/get_reward")
    async def get_reward(request: Request):
        client_host = request.client.host
        logger.info(f"client_ip: {client_host}")
        data = await request.json()
        queries = data.get("query")
        rewards = reward_model.get_reward(queries)
        result = {"rewards": rewards}
        logger.info(f"Sent JSON: {result}")
        return JSONResponse(result)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
