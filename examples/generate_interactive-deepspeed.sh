# Set visible devices
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

# Set MP_SIZE to the number of devices
MP_SIZE=8

# Provide path to vocab file and model
VOCAB_PATH="/mnt/data/Yanex-YaML-100b/vocab/voc_100b.sp"
MODEL_PATH="/mnt/data/Yanex-YaML-100b/weights"
LOAD_ARGS="\
    --vocab-file ${VOCAB_PATH} \
    --load ${MODEL_PATH}"

# Set generation parameters
GEN_ARGS="
    --temperature 1.0 \
    --top_p 0.9 \
    --seed 1234 \
    --seq-length 256 \
    --out-seq-length 128"

HPARAM_ARGS="\
    --pos-encoding-type rotary \
    --num-layers 80 \
    --embedding-size 2048 \
    --hidden-size 10240 \
    --intermediate-size 27308 \
    --activation-type geglu \
    --num-attention-heads 128 \
    --max-position-embeddings 1024 \
    --tokenizer-type SentencePiece \
    --fp16"

#DISTRIBUTED_ARGS="--nproc_per_node $MP_SIZE \
#                  --nnodes 1 \
#                  --node_rank 0 \
#                  --master_addr localhost \
#                  --master_port=1234"

COMMON_ARGS="\
    --num-samples 0 \
    --load-release-checkpoint \
    --batch-size 1 \
    --model-parallel-size $MP_SIZE \
    --make-vocab-size-divisible-by 1"

deepspeed --hostfile /domino/mpi/hosts /mnt/imported/code/YaML-100B/megatron_lm/tools/generate_samples_gpt2.py \
    $LOAD_ARGS \
    $HPARAM_ARGS \
    $COMMON_ARGS \
    $GEN_ARGS
