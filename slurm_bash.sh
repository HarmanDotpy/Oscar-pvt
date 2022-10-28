#!/bin/bash
#SBATCH --job-name=train_vlt_models
#SBATCH --nodes=1
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --exclude=a100-st-p4d24xlarge-6,a100-st-p4d24xlarge-25,a100-st-p4d24xlarge-198,a100-st-p4d24xlarge-18,a100-st-p4d24xlarge-240,a100-st-p4d24xlarge-254,a100-st-p4d24xlarge-27,a100-st-p4d24xlarge-136,a100-st-p4d24xlarge-235
#SBATCH --cpus-per-task=10
#SBATCH --output=/fsx/harman/Oscar/log_test/slurm-%j.out
#SBATCH --error=/fsx/harman/Oscar/log_test/slurm-%j.err
#SBATCH --partition=hipri


export MASTER_PORT=12340
export WORLD_SIZE=8
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

module unload /fsx/harman/KEPTLMs/module_files/cuda/11.3
module load /fsx/harman/KEPTLMs/module_files/cuda/11.3
module load /fsx/harman/KEPTLMs/module_files/nccl/2.12.7-cuda.11.3
module load /fsx/harman/KEPTLMs/module_files/nccl_efa/1.2.0-nccl.2.12.7-cuda.11.3

source ~/miniconda/etc/profile.d/conda.sh
conda activate meter_efa_dinoclone


## finetune 2mill iter oscar model on gqa
python -m torch.distributed.launch --nproc_per_node=8 oscar/run_gqa_tsv.py --img_feature_dim 2054 --max_img_seq_length 50 --data_label_type all --train_data_type all --eval_data_type bal --img_feature_type faster_r-cnn --data_dir /fsx/harman/data/VinVL_img_features/gqa --model_type bert --model_name_or_path /fsx/harman/Oscar/pretrained_models/vqa/base/checkpoint-2000000 --task_name gqa --do_lower_case --max_seq_length 165 --per_gpu_eval_batch_size 256 --per_gpu_train_batch_size 48 --learning_rate 5e-05 --num_train_epochs 5 --output_dir /checkpoints/harman/oscar/gqa --label_file /fsx/harman/data/VinVL_img_features/gqa/trainval_testdev_all_ans2label.pkl --save_epoch 1 --seed 88 --evaluate_during_training --logging_steps 4000 --drop_out 0.3 --weight_decay 0.05 --warmup_steps 0 --loss_type xe --img_feat_format tsv --classifier linear --cls_hidden_scale 2 --txt_data_dir /fsx/harman/data/VinVL_img_features/gqa --do_train --evaluate_during_training --num_workers 50 --logging_steps 200


# Without SG
# python -m torch.distributed.launch --nproc_per_node=8 oscar/run_oscarplus_pretrain.py \
#     --use_b 1 \
#     --max_grad_norm 10.0 --gradient_accumulation_steps 1 \
#     --use_img_layernorm 1 \
#     --output_dir /checkpoints/harman/oscar/ \
#     --bert_model bert --model_name_or_path bert-base-uncased \
#     --do_lower_case --learning_rate 5e-05 \
#     --warmup_steps 0 --do_train --max_seq_length 35 --on_memory \
#     --max_img_seq_length 50 --img_feature_dim 2054 \
#     --drop_out 0.1 --train_batch_size 1024 \
#     --ckpt_period 10000 --max_iters 2000000 --log_period 100 \
#     --data_dir /fsx/harman/Oscar/yaml_files --dataset_file coco_sg.yaml \
#     --textb_sample_mode 1 --texta_false_prob 0.25 --num_workers 10 \
#     --output_hidden_states --obj_relation_vocab_size 51 \
#     --max_datapoints -1

# #With SG
# python -m torch.distributed.launch --nproc_per_node=8 oscar/run_oscarplus_pretrain.py \
#     --use_b 1 \
#     --max_grad_norm 10.0 --gradient_accumulation_steps 1 \
#     --use_img_layernorm 1 \
#     --output_dir /checkpoints/harman/oscar/ \
#     --bert_model bert --model_name_or_path bert-base-uncased \
#     --do_lower_case --learning_rate 5e-05 \
#     --warmup_steps 0 --do_train --max_seq_length 35 --on_memory \
#     --max_img_seq_length 50 --img_feature_dim 2054 \
#     --drop_out 0.1 --train_batch_size 1024 \
#     --ckpt_period 10000 --max_iters 2000000 --log_period 100 \
#     --data_dir /fsx/harman/Oscar/yaml_files --dataset_file coco_sg.yaml \
#     --textb_sample_mode 1 --texta_false_prob 0.25 --num_workers 10 \
#     --use_sg --output_hidden_states --obj_relation_vocab_size 51



# srun python -m torch.distributed.launch --nproc_per_node=1 oscar/run_oscarplus_pretrain.py \
#     --use_b 1 \
#     --max_grad_norm 10.0 --gradient_accumulation_steps 1 \
#     --use_img_layernorm 1 \
#     --output_dir /checkpoints/harman/oscar/output_bsize1024_8gpu \
#     --bert_model bert --model_name_or_path bert-base-uncased \
#     --do_lower_case --learning_rate 5e-05 \
#     --warmup_steps 0 --do_train --max_seq_length 35 --on_memory \
#     --max_img_seq_length 50 --img_feature_dim 2054 \
#     --drop_out 0.1 --train_batch_size 1024 \
#     --ckpt_period 10000 --max_iters 2000000 --log_period 100 \
#     --data_dir /fsx/harman/Oscar/yaml_files --dataset_file coco_x152c4big2exp168.yaml \
#     --textb_sample_mode 1 --texta_false_prob 0.25 


# srun python oscar/run_oscarplus_pretrain.py \
#     --use_b 1 \
#     --max_grad_norm 10.0 --gradient_accumulation_steps 1 \
#     --use_img_layernorm 1 \
#     --output_dir output \
#     --bert_model bert --model_name_or_path bert-base-uncased \
#     --do_lower_case --learning_rate 5e-05 \
#     --warmup_steps 0 --do_train --max_seq_length 35 --on_memory \
#     --max_img_seq_length 50 --img_feature_dim 2054 \
#     --drop_out 0.1 --train_batch_size 64 \
#     --ckpt_period 10000 --max_iters 2000000 --log_period 100 \
#     --data_dir /fsx/harman/Oscar/yaml_files --dataset_file coco_x152c4big2exp168.yaml \
#     --textb_sample_mode 1 --texta_false_prob 0.25 



    # --nnodes=2 --node_rank=0 --master_addr="192.168.1.1" --master_port=1234

# srun --time=00:05:00 --nodes=2 --gres=gpu:1 --cpus-per-task=10 python -m torch.distributed.launch --nproc_per_node=1 oscar/run_oscarplus_pretrain.py \
#     --use_b 1 \
#     --max_grad_norm 10.0 --gradient_accumulation_steps 1 \
#     --use_img_layernorm 1 \
#     --output_dir output \
#     --bert_model bert --model_name_or_path bert-base-uncased \
#     --do_lower_case --learning_rate 5e-05 \
#     --warmup_steps 0 --do_train --max_seq_length 35 --on_memory \
#     --max_img_seq_length 50 --img_feature_dim 2054 \
#     --drop_out 0.1 --train_batch_size 1024 \
#     --ckpt_period 10000 --max_iters 2000000 --log_period 100 \
#     --data_dir /fsx/harman/Oscar/yaml_files --dataset_file coco_x152c4big2exp168.yaml \
#     --textb_sample_mode 1 --texta_false_prob 0.25 