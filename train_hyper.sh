export SQUAD_DIR=/home/ps/文档/SQUAD

python -m torch.distributed.launch --nproc_per_node=4 run_squad_hyper.py \
  --model_type albert \
  --model_name_or_path albert-xlarge-v2\
  --do_train \
  --do_eval \
  --evaluate_during_training \
  --do_lower_case \
  --train_file $SQUAD_DIR/train-v2.0.json \
  --predict_file $SQUAD_DIR/dev-v2.0.json \
  --per_gpu_train_batch_size 24 \
  --learning_rate 3e-5 \
  --num_train_epochs 1.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --fp16 \
  --overwrite_output_dir\
  --save_steps 15000 \
  --output_dir /tmp/debug_hyper/
  