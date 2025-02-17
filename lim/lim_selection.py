import json
import numpy as np
from pathlib import Path
import argparse

def load_epochs(train_samples_path, steps_per_epoch, max_epochs):
    epochs = []
    prompts = set()
    current_epoch = []
    
    for step in range(1, steps_per_epoch * max_epochs + 1):
        with open(train_samples_path / f'step_{step}.json') as f:
            step_data = json.load(f)
            current_epoch.extend(step_data)
            
        if step % steps_per_epoch == 0:
            prompts.update(sample['prompt'] for sample in current_epoch)
            epochs.append(current_epoch)
            current_epoch = []
            
    return epochs, prompts

def calculate_accuracy(epoch_data, prompt_set):
    accuracies = {prompt: [0, 0] for prompt in prompt_set}  # [correct_count, total_count]
    
    for sample in epoch_data:
        prompt = sample['prompt'].strip()
        accuracies[prompt][1] += 1
        if sample['reward'] == 1:
            accuracies[prompt][0] += 1
            
    return {prompt: correct/total if total else -1 
            for prompt, (correct, total) in accuracies.items()}

def process_accuracy_sequences(prompt_accuracies, max_epochs):
    # Forward fill missing values
    for accuracy_sequence in prompt_accuracies.values():
        for i in range(len(accuracy_sequence)-1):
            if accuracy_sequence[i] == -1 and accuracy_sequence[i+1] != -1:
                accuracy_sequence[i] = accuracy_sequence[i+1]
    
    # Filter valid sequences
    valid_sequences = [(prompt, sequence) 
                      for prompt, sequence in prompt_accuracies.items() 
                      if -1 not in sequence[:max_epochs]]
    
    if not valid_sequences:
        return [], [], []
    
    prompts, sequences = zip(*valid_sequences)
    sequences = [seq[:max_epochs] for seq in sequences]
    mean_sequence = np.mean(sequences, axis=0)
    
    return prompts, sequences, mean_sequence

def calculate_similarity_score(sequence, baseline_sequence):
    squared_diff_sum = sum((acc - baseline)**2 for acc, baseline in zip(sequence, baseline_sequence))
    max_diff_sum = sum((1 - baseline)**2 for baseline in baseline_sequence)
    return 1 - squared_diff_sum/max_diff_sum

def parse_args():
    parser = argparse.ArgumentParser(description='Process training data and filter prompts')
    parser.add_argument('--train_samples_path', type=str, required=True,
                      help='Path to the training samples directory')
    parser.add_argument('--original_prompts_path', type=str, required=True,
                      help='Path to the original prompts json file')
    parser.add_argument('--output_path', type=str, default='math.sub.average_filtered.json',
                      help='Path for output filtered data')
    parser.add_argument('--steps_per_epoch', type=int, default=8,
                      help='Number of steps that constitute one epoch')
    parser.add_argument('--max_epochs', type=int, default=21,
                      help='Maximum number of epochs to consider')
    parser.add_argument('--similarity_threshold', type=float, default=0.2,
                      help='Minimum similarity score threshold for selecting prompts')
    return parser.parse_args()

def main():
    args = parse_args()
    train_samples_path = Path(args.train_samples_path)
    
    epochs, prompts = load_epochs(
        train_samples_path, 
        args.steps_per_epoch,
        args.max_epochs
    )
    
    epoch_accuracies = [calculate_accuracy(epoch, prompts) for epoch in epochs]
    
    # Collect accuracy sequences for each prompt
    prompt_accuracies = {prompt: [epoch[prompt] for epoch in epoch_accuracies] 
                        for prompt in prompts}
    
    valid_prompts, accuracy_sequences, baseline_sequence = process_accuracy_sequences(
        prompt_accuracies, args.max_epochs)
    
    # Calculate similarity scores
    prompt_scores = {
        prompt: calculate_similarity_score(sequence, baseline_sequence)
        for prompt, sequence in zip(valid_prompts, accuracy_sequences)
    }
    
    selected_prompts = {prompt for prompt, score in prompt_scores.items() 
                       if score >= args.similarity_threshold}
    
    # Save filtered data
    with open(args.original_prompts_path) as f:
        original_data = json.load(f)
    
    filtered_data = [sample for sample in original_data if sample['prompt'] in selected_prompts]
    with open(args.output_path, 'w') as f:
        json.dump(filtered_data, f)
    
    print(f"Selected {len(filtered_data)} prompts")

if __name__ == "__main__":
    main()