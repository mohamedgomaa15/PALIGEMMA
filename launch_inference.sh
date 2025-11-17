MODEL_PATH="$HOME/projects/paligemma-weights/paligemma-3b-pt-224"
PROMPT="this animal is"
IMAGE_FILE_PATH="test_images/image1.jpg"
MAX_TOKEN_TO_GENERATE=100
TEMPERATURE=0.8
TOP_P=0.9
DO_SAMPLE="False"
ONLY_CPU="False"

python inference.py \
    --model_path $MODEL_PATH \
    --prompt "$PROMPT" \
    --image_file_path $IMAGE_FILE_PATH \
    --max_tokens_to_generate $MAX_TOKEN_TO_GENERATE \
    --temperature $TEMPERATURE \
    --top_p $TOP_P \
    --do_sample $DO_SAMPLE \
    --only_cpu $ONLY_CPU