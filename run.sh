CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 oscar/run_oscarplus_pretrain.py \
    --use_b 1 \
    --max_grad_norm 10.0 --gradient_accumulation_steps 1 \
    --use_img_layernorm 1 \
    --output_dir output \
    --bert_model bert --model_name_or_path bert-base-uncased \
    --do_lower_case --learning_rate 5e-05 \
    --warmup_steps 0 --do_train --max_seq_length 35 --on_memory \
    --max_img_seq_length 50 --img_feature_dim 2054 \
    --drop_out 0.1 --train_batch_size 8 \
    --ckpt_period 10000 --max_iters 2000000 --log_period 100 \
    --data_dir /fsx/harman/Oscar/yaml_files --dataset_file coco_x152c4big2exp168.yaml \
    --textb_sample_mode 1 --texta_false_prob 0.25 

#Test Wandb
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 oscar/debug_run_oscarplus_pretrain.py \
    --use_b 1 \
    --max_grad_norm 10.0 --gradient_accumulation_steps 1 \
    --use_img_layernorm 1 \
    --output_dir /checkpoints/harman/oscar/output_bsize1024_8gpu \
    --bert_model bert --model_name_or_path bert-base-uncased \
    --do_lower_case --learning_rate 5e-05 \
    --warmup_steps 0 --do_train --max_seq_length 35 --on_memory \
    --max_img_seq_length 50 --img_feature_dim 2054 \
    --drop_out 0.1 --train_batch_size 32 \
    --ckpt_period 10000 --max_iters 2000000 --log_period 100 \
    --data_dir /fsx/harman/Oscar/yaml_files --dataset_file coco_x152c4big2exp168.yaml \
    --textb_sample_mode 1 --texta_false_prob 0.25


# debug code
python -m torch.distributed.launch --nproc_per_node=1 oscar/run_oscarplus_pretrain.py \
    --use_b 1 \
    --max_grad_norm 10.0 --gradient_accumulation_steps 1 \
    --use_img_layernorm 1 \
    --output_dir /checkpoints/harman/oscar/output_temp \
    --bert_model bert --model_name_or_path bert-base-uncased \
    --do_lower_case --learning_rate 5e-05 \
    --warmup_steps 0 --do_train --max_seq_length 35 --on_memory \
    --max_img_seq_length 50 --img_feature_dim 2054 \
    --drop_out 0.1 --train_batch_size 1024 \
    --ckpt_period 10000 --max_iters 2000000 --log_period 100 \
    --data_dir /fsx/harman/Oscar/yaml_files --dataset_file coco_x152c4big2exp168.yaml \
    --textb_sample_mode 1 --texta_false_prob 0.25 --num_workers 0

# debug code - see if use_sg works and what does it return
python -m torch.distributed.launch --nproc_per_node=1 oscar/run_oscarplus_pretrain.py \
    --use_b 1 \
    --max_grad_norm 10.0 --gradient_accumulation_steps 1 \
    --use_img_layernorm 1 \
    --output_dir /checkpoints/harman/oscar/output_temp \
    --bert_model bert --model_name_or_path bert-base-uncased \
    --do_lower_case --learning_rate 5e-05 \
    --warmup_steps 0 --do_train --max_seq_length 35 --on_memory \
    --max_img_seq_length 50 --img_feature_dim 2054 \
    --drop_out 0.1 --train_batch_size 1024 \
    --ckpt_period 10000 --max_iters 2000000 --log_period 100 \
    --data_dir /fsx/harman/Oscar/yaml_files --dataset_file coco_sg.yaml \
    --textb_sample_mode 1 --texta_false_prob 0.25 --num_workers 0 \
    --use_sg

# debug code - see if use_sg works and what does it return ,with max datapoints
python -m torch.distributed.launch --nproc_per_node=1 oscar/run_oscarplus_pretrain.py \
    --use_b 1 \
    --max_grad_norm 10.0 --gradient_accumulation_steps 1 \
    --use_img_layernorm 1 \
    --output_dir /checkpoints/harman/oscar/output_temp \
    --bert_model bert --model_name_or_path bert-base-uncased \
    --do_lower_case --learning_rate 5e-05 \
    --warmup_steps 0 --do_train --max_seq_length 35 --on_memory \
    --max_img_seq_length 50 --img_feature_dim 2054 \
    --drop_out 0.1 --train_batch_size 32 \
    --ckpt_period 10000 --max_iters 2000000 --log_period 100 \
    --data_dir /fsx/harman/Oscar/yaml_files --dataset_file coco_sg.yaml \
    --textb_sample_mode 1 --texta_false_prob 0.25 --num_workers 0 \
    --use_sg --max_datapoints 10000

# debug code - see if use_sg works and what does it return ,with max datapoints, output hiddne satte
python -m torch.distributed.launch --nproc_per_node=1 oscar/run_oscarplus_pretrain.py \
    --use_b 1 \
    --max_grad_norm 10.0 --gradient_accumulation_steps 1 \
    --use_img_layernorm 1 \
    --output_dir /checkpoints/harman/oscar/output_temp \
    --bert_model bert --model_name_or_path bert-base-uncased \
    --do_lower_case --learning_rate 5e-05 \
    --warmup_steps 0 --do_train --max_seq_length 35 --on_memory \
    --max_img_seq_length 50 --img_feature_dim 2054 \
    --drop_out 0.1 --train_batch_size 32 \
    --ckpt_period 10000 --max_iters 2000000 --log_period 100 \
    --data_dir /fsx/harman/Oscar/yaml_files --dataset_file coco_sg.yaml \
    --textb_sample_mode 1 --texta_false_prob 0.25 --num_workers 0 \
    --use_sg --max_datapoints 10000 --output_hidden_states --obj_relation_vocab_size 51