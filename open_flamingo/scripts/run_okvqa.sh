DEVICE=0 # gpu num

RANDOM_ID="OKVQA_Result_file_name"
RESULTS_FILE="results_${RANDOM_ID}.json"

export MASTER_ADDR='localhost'
export MASTER_PORT='10000'

python open_flamingo/eval/evaluate_okvqa.py \
    --retrieval_name $RANDOM_ID \
    --lm_path "Path for mpt-7b" \
    --lm_tokenizer_path "Path for mpt-7b" \
    --checkpoint_path "Path for OpenFlamingo-9B-vitl-mpt7b checkpoint.pt" \
    --vision_encoder_path "ViT-L-14" \
    --vision_encoder_pretrained 'openai' \
    --device $DEVICE \
    --ok_vqa_train_image_dir_path "mscoco2014/train2014/"  \
    --ok_vqa_train_questions_json_path "okvqa/OpenEnded_mscoco_train2014_questions.json" \
    --ok_vqa_train_annotations_json_path  "okvqa/mscoco_train2014_annotations.json" \
    --ok_vqa_test_image_dir_path "mscoco2014/val2014/" \
    --ok_vqa_test_questions_json_path "okvqa/OpenEnded_mscoco_val2014_questions.json" \
    --ok_vqa_test_annotations_json_path "okvqa/mscoco_val2014_annotations.json" \
    --results_file $RESULTS_FILE \
    --num_samples 5000\
    --shots 4 8 16 32 \
    --num_trials 1 \
    --seed 5 \
    --batch_size 1 \
    --cross_attn_every_n_layers 4 \
    --precision fp16 \
    --dataset_name ok_vqa \
    --eval_ok_vqa \
    
echo "evaluation complete! results written to $RESULTS_FILE"
