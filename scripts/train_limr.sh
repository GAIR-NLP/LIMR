set -x
# start rm
if [ "$PET_NODE_RANK" -eq 0 ]; then
    python -m openrlhf.cli.serve_rm \
        --mode rule \
        --tokenizer_path Qwen/Qwen2.5-Math-7B \
        --max_gen_len 3072 \
        --data_path data/train/limr \
        --port 5000 \
        --host $MASTER_ADDR &
fi

sleep 10s
RAY_MASTER_PORT=6379
RAY_DASHBOARD_PORT=8265
if [ "$PET_NODE_RANK" -eq 0 ]; then
    echo "Starting Ray head node on $(hostname)"
    ray start --head  --port=$RAY_MASTER_PORT --dashboard-host=127.0.0.1 --dashboard-port=$RAY_DASHBOARD_PORT --num-gpus 8
else
    echo "Starting Ray worker node on $(hostname)"
    ray start --address="$MASTER_ADDR:$RAY_MASTER_PORT" --num-gpus 8 --block
fi


sleep 10s
# start rl
if [ "$PET_NODE_RANK" -eq 0 ]; then
RAY_ADDRESS="http://127.0.0.1:$RAY_DASHBOARD_PORT" ray job submit --address="http://$MASTER_ADDR:$MASTER_PORT" \
    --runtime-env-json='{"working_dir": "path/to/limr"}' \
    -- python3 -m openrlhf.cli.train_ppo_ray \
    --ref_num_nodes 1 \
    --ref_num_gpus_per_node 4 \
    --critic_num_nodes 1 \
    --critic_num_gpus_per_node 4 \
    --actor_num_nodes 1 \
    --actor_num_gpus_per_node 4 \
    --vllm_num_engines 4 \
    --vllm_tensor_parallel_size 1 \
    --eval_steps 1 \
    --save_steps 1 \
    --pretrain Qwen/Qwen2.5-Math-7B \
    --ref_pretrain Qwen/Qwen2.5-Math-7B \
    --remote_rm_url http://$MASTER_ADDR:5000/get_reward \
    --save_path path/to/save \
    --ckpt_path path/to/save \
    --samples_save_path data/output/limr/ \
    --micro_train_batch_size 4 \
    --train_batch_size 256 \
    --micro_rollout_batch_size 4 \
    --rollout_batch_size 1024 \
    --n_samples_per_prompt 8 \
    --max_epochs 1 \
    --num_episodes 10000 \
    --prompt_max_len 1024 \
    --generate_max_len 3072 \
    --advantage_estimator gae \
    --actor_learning_rate 5e-7 \
    --critic_learning_rate 9e-6 \
    --init_kl_coef 0.01 \
    --eps_clip 0.2 \
    --value_clip 0.2 \
    --lambd 0.95 \
    --zero_stage 3 \
    --bf16 \
    --lr_warmup_steps 10 \
    --prompt_data data/train/limr \
    --test_path data/evaluation/test.json \
    --input_key context_messages \
    --apply_chat_template \
    --max_samples 100000 \
    --packing_samples \
    --normalize_reward \
    --adam_offload \
    --flash_attn \
    --vllm_sync_backend nccl \
    --gradient_checkpointing \
    --temperature 1.2 
fi
