from vllm import LLM, SamplingParams
import os
import re
import json
import jsonlines
import argparse
from tqdm import tqdm
import sys
import warnings
import pdb
import ray
warnings.filterwarnings("ignore")
import torch
from transformers import AutoTokenizer
# from math_verify import parse, verify
from utils.equal import math_equal, normalize_final_answer

def get_prompts(dev_set, apply_chat_template):
    prompt2answer={}
    processed_prompts=[]
    with open(f"./data/{dev_set}.jsonl", 'r', encoding='utf-8') as f:
        for line in jsonlines.Reader(f):
            chat=[{"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."}, {"role": "user", "content": line['question']}]
            prompt=apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            processed_prompts.append(prompt)
            prompt2answer[prompt]=line['answer']
    print(processed_prompts[-1])
    return processed_prompts, prompt2answer
    


def math_eval(response, gt):
    matches = re.findall(r"\\boxed\{((?:[^{}]|\\{|\\}|(?:\{(?:[^{}]|\\{|\\}|(?:\{(?:[^{}]|\\{|\\}|(?:\{[^{}]*\}))*\}))*\}))*\})", response)
    if len(matches)==0: pred=""
    else: pred=matches[-1][:-1]
    pred, gt=normalize_final_answer(pred), normalize_final_answer(gt)
    return math_equal(pred, gt)


def eval_ckpt(model_path, dev_sets):
    if not dev_sets: 
        dev_sets=["aime24", "amc23", "math500"]
    with open("./config.json", 'r', encoding='utf-8') as f: config=json.load(f)
    
    num_gpus = torch.cuda.device_count()
    another_args = {'max_num_batched_tokens': 32768}
    apply_chat_template=AutoTokenizer.from_pretrained(model_path).apply_chat_template
    llm = LLM(model=model_path, tensor_parallel_size=num_gpus, **another_args, trust_remote_code=True)
    
    eval_acc={}
    eval_responses={}
    for dev_set in dev_sets:
        processed_prompts, prompt2answer = get_prompts(dev_set, apply_chat_template)
        n, temperature=config[dev_set]['n'], config[dev_set]['temperature']
        sampling_params = SamplingParams(n=n, temperature=temperature, top_p=0.95, max_tokens=3072)
        outputs = llm.generate(processed_prompts, sampling_params)
        eval_results=[]
        eval_response=[]
        for output in outputs:
            prompt=output.prompt
            responses=[output.outputs[i].text for i in range(n)]
            answer=prompt2answer[prompt]
            eval_result=[math_eval(response, answer) for response in responses]
            eval_results.extend(eval_result)
            eval_response.append({"question": prompt, "responses": responses, "results": eval_result, "answer": answer})
            
        acc=sum(1 for result in eval_results if result is True)/len(eval_results)
        eval_acc[dev_set]=acc
        eval_responses[dev_set]=eval_response
    
    return eval_acc, eval_responses


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_path", type=str, default=None)
    parser.add_argument("--step", type=str, default=None)
    parser.add_argument("--dev_sets", type=list, default=None)
    parser.add_argument('--output_name', type=str, default='./output')
    args = parser.parse_args()

    run_name=args.run_path.split("/")[-1]
    model_path=os.path.join(args.run_path, "global_step{}_hf".format(args.step))
    eval_acc, outputs = eval_ckpt(model_path, args.dev_sets)
    print(eval_acc)
    print("eval done")
    os.makedirs("./output/eval_outputs/{}".format(run_name), exist_ok=True)
    os.makedirs("./output/results/{}".format(run_name), exist_ok=True)
    with open("./output/eval_outputs/{}/{}.json".format(run_name, args.step), 'w', encoding='utf-8') as f: 
        json.dump(outputs, f ,ensure_ascii=False, indent=4)
    with open("./output/results/{}/{}.json".format(run_name, args.step), 'w', encoding='utf-8') as f: 
        json.dump(eval_acc, f,ensure_ascii=False, indent=4)
    