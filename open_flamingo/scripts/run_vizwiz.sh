DEVICE=0 # gpu num

RANDOM_ID="VizWiz_Result_file_name"
RESULTS_FILE="results_${RANDOM_ID}.json"

export MASTER_ADDR='localhost'
export MASTER_PORT='10000'

python open_flamingo/eval/evaluate_vizwiz.py \
    --model "open_flamingo" \
    --lm_path "Path for mpt-7b" \
    --lm_tokenizer_path "Path for mpt-7b" \
    --checkpoint_path "Path for OpenFlamingo-9B-vitl-mpt7b checkpoint.pt" \
    --vision_encoder_path "ViT-L-14" \
    --vision_encoder_pretrained 'openai' \
    --device $DEVICE \
    --vizwiz_train_image_dir_path "vizwiz/image/train/"  \
    --vizwiz_train_questions_json_path "vizwiz/train_questions_vqa_format.json" \
    --vizwiz_train_annotations_json_path  "vizwiz/train_annotations_vqa_format.json" \
    --vizwiz_test_image_dir_path "vizwiz/image/val/" \
    --vizwiz_test_questions_json_path "vizwiz/val_questions_vqa_format.json" \
    --vizwiz_test_annotations_json_path "vizwiz/val_annotations_vqa_format.json" \
    --results_file $RESULTS_FILE \
    --num_samples 5000 \
    --shots 4 8 16 32 \
    --num_trials 1 \
    --seed 5 \
    --batch_size 1 \
    --cross_attn_every_n_layers 4 \
    --precision fp16 \
    --eval_vizwiz

echo "evaluation complete! results written to ${RESULTS_FILE}"
