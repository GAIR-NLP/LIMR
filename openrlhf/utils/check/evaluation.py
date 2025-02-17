import os
import json
import jsonlines
import copy
from tqdm import tqdm 
import pandas as pd
from multiprocessing import Pool
from functools import partial
from datetime import datetime
from .qwen_equal import math_equal
from .extraction import extract
from .math_normalization import normalize_final_answer

def process_answer_list(answer_list):
    answer_list = list(set(answer_list))
    if "" in answer_list: answer_list.remove("")
    return answer_list



def exact_match_eval(pred, gt):
    gt=normalize_final_answer(gt)

    answer_list=extract(pred)
    normalized_answer_list=[]
    for answer in copy.deepcopy(answer_list):
        normalized_answer_list.append(normalize_final_answer(answer))
    normalized_answer_list=process_answer_list(normalized_answer_list)
    
    for answer in normalized_answer_list:
        if math_equal(gt, answer):
            return normalized_answer_list, True
    return normalized_answer_list, False


def process_chunk(chunk_w_id):
    process_index, chunk = chunk_w_id
    tqdm.pandas(desc=f"Chunk {process_index}", position=process_index)
    for idx, line in tqdm(enumerate(chunk)):
        answer=line['response']['answer']
        content=line['response']['raw_content'][-300:]
        answer_list, res = exact_match_eval(content, answer)
        chunk[idx]['extract_answer']=answer_list
        chunk[idx]['eval_result']=res
        chunk[idx]['gt']=answer
    return (process_index, chunk)

def eval_in_parallel(input_data, output_file, num_processes=3, chunk_size=100):
    data=[]
    with open(input_data, 'r', encoding='utf-8') as f:
        for line in jsonlines.Reader(f):
            data.append(line)
    # Split the DataFrame into chunks
    chunk_size = min(chunk_size, len(data) // num_processes) # int(len(df) / num_processes) + 1
    chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
    
    chunks_to_process = []
    for i, chunk in enumerate(chunks):
        chunks_to_process.append((i, chunk))

    res=[]
    if chunks_to_process:
        process_func = partial(process_chunk)
        with Pool(processes=num_processes) as pool:
            # Use imap_unordered to process results as they become available
            for result in tqdm(pool.imap_unordered(process_func, chunks_to_process),
                               total=len(chunks_to_process),
                               desc="Processing Chunks"):
                i, result_chunk = result
                res+=result_chunk
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in res:
            json.dump(line, f)
            f.write('\n')

if __name__ == "__main__":
    exact_match_eval("Answer**\n\n\\[\\boxed{1.2}\\]", "\\frac{6}{5}")
    pass