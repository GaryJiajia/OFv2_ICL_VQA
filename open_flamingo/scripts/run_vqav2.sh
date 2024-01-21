DEVICE=0 # gpu number

RANDOM_ID="VQAv2_Result_file_name"
RESULTS_FILE="results_${RANDOM_ID}.json"

export MASTER_ADDR='localhost'
export MASTER_PORT='10000' 

python open_flamingo/eval/evaluate_vqa.py \
    --retrieval_name $RANDOM_ID \
    --lm_path "Path for mpt-7b" \
    --lm_tokenizer_path "Path for mpt-7b" \
    --checkpoint_path "Path for OpenFlamingo-9B-vitl-mpt7b checkpoint.pt" \
    --vision_encoder_path "ViT-L-14" \
    --vision_encoder_pretrained 'openai' \
    --device $DEVICE \
    --vqav2_train_image_dir_path "mscoco2014/train2014/"  \
    --vqav2_train_questions_json_path "vqav2/v2_OpenEnded_mscoco_train2014_questions.json" \
    --vqav2_train_annotations_json_path  "vqav2/v2_mscoco_train2014_annotations.json" \
    --vqav2_test_image_dir_path "mscoco2014/val2014/" \
    --vqav2_test_questions_json_path "vqav2/v2_OpenEnded_mscoco_val2014_questions" \
    --vqav2_test_annotations_json_path "vqav2/v2_mscoco_val2014_annotations.json" \
    --results_file $RESULTS_FILE \
    --num_samples 5000\
    --shots 4 8 16 32\
    --num_trials 1 \
    --seed 5 \
    --batch_size 1 \
    --cross_attn_every_n_layers 4 \
    --precision fp16 \
    --dataset_name vqav2 \
    --eval_vqav2 \
    
echo "evaluation complete! results written to $RESULTS_FILE"
