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
    --ckpt_period 100 --max_iters 2000000 --log_period 100 \
    --data_dir /fsx/harman/Oscar/yaml_files --dataset_file coco_sg.yaml \
    --textb_sample_mode 1 --texta_false_prob 0.25 --num_workers 0 \
    --use_sg --max_datapoints 10000 --output_hidden_states --obj_relation_vocab_size 51

# run same without use_sg
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
    --ckpt_period 100 --max_iters 2000000 --log_period 100 \
    --data_dir /fsx/harman/Oscar/yaml_files --dataset_file coco_sg.yaml \
    --textb_sample_mode 1 --texta_false_prob 0.25 --num_workers 0 \
    --max_datapoints 10000 --output_hidden_states --obj_relation_vocab_size 51


python -m torch.distributed.launch --nproc_per_node=1 oscar/run_oscarplus_pretrain.py \
    --use_b 1 \
    --max_grad_norm 10.0 --gradient_accumulation_steps 1 \
    --use_img_layernorm 1 \
    --output_dir /checkpoints/harman/oscar/ \
    --bert_model bert --model_name_or_path bert-base-uncased \
    --do_lower_case --learning_rate 5e-05 \
    --warmup_steps 0 --do_train --max_seq_length 35 --on_memory \
    --max_img_seq_length 50 --img_feature_dim 2054 \
    --drop_out 0.1 --train_batch_size 32 \
    --ckpt_period 10000 --max_iters 2000000 --log_period 100 \
    --data_dir /fsx/harman/Oscar/yaml_files --dataset_file coco_sg.yaml \
    --textb_sample_mode 1 --texta_false_prob 0.25 --num_workers 10 \
    --use_sg --output_hidden_states --obj_relation_vocab_size 51

###### after changing from srun to normal python run

# Without SG - 88
python -m torch.distributed.launch --nproc_per_node=8 oscar/run_oscarplus_pretrain.py \
    --use_b 1 \
    --max_grad_norm 10.0 --gradient_accumulation_steps 1 \
    --use_img_layernorm 1 \
    --output_dir /checkpoints/harman/oscar/ \
    --bert_model bert --model_name_or_path bert-base-uncased \
    --do_lower_case --learning_rate 5e-05 \
    --warmup_steps 0 --do_train --max_seq_length 35 --on_memory \
    --max_img_seq_length 50 --img_feature_dim 2054 \
    --drop_out 0.1 --train_batch_size 1024 \
    --ckpt_period 10000 --max_iters 2000000 --log_period 100 \
    --data_dir /fsx/harman/Oscar/yaml_files --dataset_file coco_sg.yaml \
    --textb_sample_mode 1 --texta_false_prob 0.25 --num_workers 10 \
    --output_hidden_states --obj_relation_vocab_size 51 \
    --max_datapoints -1

# #With SG - 87
python -m torch.distributed.launch --nproc_per_node=8 oscar/run_oscarplus_pretrain.py \
    --use_b 1 \
    --max_grad_norm 10.0 --gradient_accumulation_steps 1 \
    --use_img_layernorm 1 \
    --output_dir /checkpoints/harman/oscar/ \
    --bert_model bert --model_name_or_path bert-base-uncased \
    --do_lower_case --learning_rate 5e-05 \
    --warmup_steps 0 --do_train --max_seq_length 35 --on_memory \
    --max_img_seq_length 50 --img_feature_dim 2054 \
    --drop_out 0.1 --train_batch_size 1024 \
    --ckpt_period 10000 --max_iters 2000000 --log_period 100 \
    --data_dir /fsx/harman/Oscar/yaml_files --dataset_file coco_sg.yaml \
    --textb_sample_mode 1 --texta_false_prob 0.25 --num_workers 10 \
    --use_sg --output_hidden_states --obj_relation_vocab_size 51


# debugging first before incoroporating obj tag rel classification loss
python -m torch.distributed.launch --nproc_per_node=1 oscar/run_oscarplus_pretrain.py \
    --use_b 1 \
    --max_grad_norm 10.0 --gradient_accumulation_steps 1 \
    --use_img_layernorm 1 \
    --output_dir /checkpoints/harman/oscar/ \
    --bert_model bert --model_name_or_path bert-base-uncased \
    --do_lower_case --learning_rate 5e-05 \
    --warmup_steps 0 --do_train --max_seq_length 35 --on_memory \
    --max_img_seq_length 50 --img_feature_dim 2054 \
    --drop_out 0.1 --train_batch_size 32 \
    --ckpt_period 10000 --max_iters 2000000 --log_period 100 \
    --data_dir /fsx/harman/Oscar/yaml_files --dataset_file coco_sg.yaml \
    --textb_sample_mode 1 --texta_false_prob 0.25 --num_workers 0 \
    --max_datapoints 10000 --use_sg --output_hidden_states --obj_relation_vocab_size 51 --debug



# run gqa eval
python oscar/run_gqa.py -j 4 --img_feature_dim 2054 --max_img_seq_length
    45 --data_dir vinvl/datasets/gqa --model_type bert --model_name_or_path /fsx/harman/Oscar/pretrained_models/vqa/base/checkpoint-2000000
    --task_name gqa --do_lower_case --max_seq_length 165 --per_gpu_eval_batch_size
    256 --per_gpu_train_batch_size 48 --learning_rate 5e-05 --num_train_epochs 5 --output_dir
    results --label_file vinvl/datasets/gqa/trainval_testdev_all_ans2label.pkl
    --img_feature_type faster_r-cnn --data_label_type all --train_data_type all --eval_data_type
    bal --label2ans_file vinvl/datasets/gqa/trainval_testdev_all_label2ans.pkl
    --loss_type xe --save_epoch 2 --seed 88 --evaluate_during_training --logging_steps
    4000 --drop_out 0.3 --weight_decay 0.05 --warmup_steps 0 --do_eval

# copied the vqa comand but changed it for gqa -eval only
python oscar/run_gqa_tsv.py --img_feature_dim 2054 --max_img_seq_length 50 
--data_label_type all --train_data_type all --eval_data_type bal --img_feature_type faster_r-cnn --data_dir /fsx/harman/data/VinVL_img_features/gqa --model_type bert --model_name_or_path /fsx/harman/Oscar/pretrained_models/vqa/base/checkpoint-2000000 --task_name gqa --do_lower_case --max_seq_length 165 --per_gpu_eval_batch_size 256 --per_gpu_train_batch_size 48 --learning_rate 5e-05 --num_train_epochs 5 --output_dir oscar_results/gqa --label_file /fsx/harman/data/VinVL_img_features/gqa/trainval_testdev_all_ans2label.pkl --save_epoch 1 --seed 88 --evaluate_during_training --drop_out 0.3 --weight_decay 0.05 --warmup_steps 0 --loss_type xe --img_feat_format tsv --classifier linear --cls_hidden_scale 2 --txt_data_dir /fsx/harman/data/VinVL_img_features/gqa --do_eval --num_workers 0

#- gqa finetune
python -m torch.distributed.launch --nproc_per_node=1 oscar/run_gqa_tsv.py --img_feature_dim 2054 --max_img_seq_length 50 --data_label_type all --train_data_type all --eval_data_type bal --img_feature_type faster_r-cnn --data_dir /fsx/harman/data/VinVL_img_features/gqa --model_type bert --model_name_or_path /fsx/harman/Oscar/pretrained_models/vqa/base/checkpoint-2000000 --task_name gqa --do_lower_case --max_seq_length 165 --per_gpu_eval_batch_size 256 --per_gpu_train_batch_size 48 --learning_rate 5e-05 --num_train_epochs 5 --output_dir /checkpoints/harman/oscar/gqa --label_file /fsx/harman/data/VinVL_img_features/gqa/trainval_testdev_all_ans2label.pkl --save_epoch 1 --seed 88 --evaluate_during_training --drop_out 0.3 --weight_decay 0.05 --warmup_steps 0 --loss_type xe --img_feat_format tsv --classifier linear --cls_hidden_scale 2 --txt_data_dir /fsx/harman/data/VinVL_img_features/gqa --do_train --evaluate_during_training --num_workers 0 --logging_steps 200


## finetune oscar our pretrained model on coco +vg + vqa dataset, -- 88 model
python -m torch.distributed.launch --nproc_per_node=8 oscar/run_gqa_tsv.py --img_feature_dim 2054 --max_img_seq_length 50 --data_label_type all --train_data_type all --eval_data_type bal --img_feature_type faster_r-cnn --data_dir /fsx/harman/data/VinVL_img_features/gqa --model_type bert --model_name_or_path /checkpoints/harman/oscar/88/checkpoint-0240000 --task_name gqa --do_lower_case --max_seq_length 165 --per_gpu_eval_batch_size 256 --per_gpu_train_batch_size 48 --learning_rate 5e-05 --num_train_epochs 5 --output_dir /checkpoints/harman/oscar/gqa --label_file /fsx/harman/data/VinVL_img_features/gqa/trainval_testdev_all_ans2label.pkl --save_epoch 1 --seed 88 --evaluate_during_training --drop_out 0.3 --weight_decay 0.05 --warmup_steps 0 --loss_type xe --img_feat_format tsv --classifier linear --cls_hidden_scale 2 --txt_data_dir /fsx/harman/data/VinVL_img_features/gqa --do_train --evaluate_during_training --num_workers 50 --logging_steps 200


## finetune oscar + rel predict our pretrained model on coco +vg + vqa dataset,  -- 87 model
python -m torch.distributed.launch --nproc_per_node=8 oscar/run_gqa_tsv.py --img_feature_dim 2054 --max_img_seq_length 50 --data_label_type all --train_data_type all --eval_data_type bal --img_feature_type faster_r-cnn --data_dir /fsx/harman/data/VinVL_img_features/gqa --model_type bert --model_name_or_path /checkpoints/harman/oscar/87/checkpoint-0240000 --task_name gqa --do_lower_case --max_seq_length 165 --per_gpu_eval_batch_size 256 --per_gpu_train_batch_size 48 --learning_rate 5e-05 --num_train_epochs 5 --output_dir /checkpoints/harman/oscar/gqa --label_file /fsx/harman/data/VinVL_img_features/gqa/trainval_testdev_all_ans2label.pkl --save_epoch 1 --seed 88 --evaluate_during_training --drop_out 0.3 --weight_decay 0.05 --warmup_steps 0 --loss_type xe --img_feat_format tsv --classifier linear --cls_hidden_scale 2 --txt_data_dir /fsx/harman/data/VinVL_img_features/gqa --do_train --evaluate_during_training --num_workers 50 --logging_steps 200


# 112
## gqa fintune using 88 model
python -m torch.distributed.launch --nproc_per_node=8 oscar/run_gqa_tsv.py --img_feature_dim 2054 --max_img_seq_length 50 --data_label_type all --train_data_type all --eval_data_type bal --img_feature_type faster_r-cnn --data_dir /fsx/harman/data/VinVL_img_features/gqa --model_type bert --model_name_or_path /checkpoints/harman/oscar/88/checkpoint-0240000 --task_name gqa --do_lower_case --max_seq_length 165 --per_gpu_eval_batch_size 256 --per_gpu_train_batch_size 48 --learning_rate 5e-05 --num_train_epochs 5 --output_dir /checkpoints/harman/oscar/gqa --label_file /fsx/harman/data/VinVL_img_features/gqa/trainval_testdev_all_ans2label.pkl --save_epoch 1 --seed 88 --evaluate_during_training --drop_out 0.3 --weight_decay 0.05 --warmup_steps 0 --loss_type xe --img_feat_format tsv --classifier linear --cls_hidden_scale 2 --txt_data_dir /fsx/harman/data/VinVL_img_features/gqa --do_train --evaluate_during_training --num_workers 50 --logging_steps 200

## gqa fintune using 87 model, continue form 109
python -m torch.distributed.launch --nproc_per_node=8 oscar/run_gqa_tsv.py --img_feature_dim 2054 --max_img_seq_length 50 --data_label_type all --train_data_type all --eval_data_type bal --img_feature_type faster_r-cnn --data_dir /fsx/harman/data/VinVL_img_features/gqa --model_type bert --model_name_or_path /checkpoints/harman/oscar/gqa/109/checkpoint-2 --task_name gqa --do_lower_case --max_seq_length 165 --per_gpu_eval_batch_size 256 --per_gpu_train_batch_size 48 --learning_rate 5e-05 --num_train_epochs 5 --output_dir /checkpoints/harman/oscar/gqa --label_file /fsx/harman/data/VinVL_img_features/gqa/trainval_testdev_all_ans2label.pkl --save_epoch 1 --seed 88 --evaluate_during_training --drop_out 0.3 --weight_decay 0.05 --warmup_steps 0 --loss_type xe --img_feat_format tsv --classifier linear --cls_hidden_scale 2 --txt_data_dir /fsx/harman/data/VinVL_img_features/gqa --do_train --evaluate_during_training --num_workers 50 --logging_steps 200


## finetune on gaa oscar + rel predict (our pretrained model trianed on coco +vg + vqa dataset),  -- 87 model on gqa BAL
python -m torch.distributed.launch --nproc_per_node=8 oscar/run_gqa_tsv.py --img_feature_dim 2054 --max_img_seq_length 50 --data_label_type all --train_data_type bal --eval_data_type bal --img_feature_type faster_r-cnn --data_dir /fsx/harman/data/VinVL_img_features/gqa --model_type bert --model_name_or_path /checkpoints/harman/oscar/87/checkpoint-0240000 --task_name gqa --do_lower_case --max_seq_length 165 --per_gpu_eval_batch_size 256 --per_gpu_train_batch_size 48 --learning_rate 5e-05 --num_train_epochs 5 --output_dir /checkpoints/harman/oscar/gqa --label_file /fsx/harman/data/VinVL_img_features/gqa/trainval_testdev_all_ans2label.pkl --save_epoch 1 --seed 88 --evaluate_during_training --drop_out 0.3 --weight_decay 0.05 --warmup_steps 0 --loss_type xe --img_feat_format tsv --classifier linear --cls_hidden_scale 2 --txt_data_dir /fsx/harman/data/VinVL_img_features/gqa --do_train --evaluate_during_training --num_workers 50 --logging_steps 200

## finetune on gaa oscar + rel predict (our pretrained model trianed on coco +vg + vqa dataset),  -- 88 model BAL
python -m torch.distributed.launch --nproc_per_node=8 oscar/run_gqa_tsv.py --img_feature_dim 2054 --max_img_seq_length 50 --data_label_type all --train_data_type bal --eval_data_type bal --img_feature_type faster_r-cnn --data_dir /fsx/harman/data/VinVL_img_features/gqa --model_type bert --model_name_or_path /checkpoints/harman/oscar/88/checkpoint-0240000 --task_name gqa --do_lower_case --max_seq_length 165 --per_gpu_eval_batch_size 256 --per_gpu_train_batch_size 48 --learning_rate 5e-05 --num_train_epochs 5 --output_dir /checkpoints/harman/oscar/gqa --label_file /fsx/harman/data/VinVL_img_features/gqa/trainval_testdev_all_ans2label.pkl --save_epoch 1 --seed 88 --evaluate_during_training --drop_out 0.3 --weight_decay 0.05 --warmup_steps 0 --loss_type xe --img_feat_format tsv --classifier linear --cls_hidden_scale 2 --txt_data_dir /fsx/harman/data/VinVL_img_features/gqa --do_train --evaluate_during_training --num_workers 50 --logging_steps 200


## finetune 2mill iter oscar model on gqa - BAL
python -m torch.distributed.launch --nproc_per_node=8 oscar/run_gqa_tsv.py --img_feature_dim 2054 --max_img_seq_length 50 --data_label_type all --train_data_type bal --eval_data_type bal --img_feature_type faster_r-cnn --data_dir /fsx/harman/data/VinVL_img_features/gqa --model_type bert --model_name_or_path /fsx/harman/Oscar/pretrained_models/vqa/base/checkpoint-2000000 --task_name gqa --do_lower_case --max_seq_length 165 --per_gpu_eval_batch_size 256 --per_gpu_train_batch_size 48 --learning_rate 5e-05 --num_train_epochs 5 --output_dir /checkpoints/harman/oscar/gqa --label_file /fsx/harman/data/VinVL_img_features/gqa/trainval_testdev_all_ans2label.pkl --save_epoch 1 --seed 88 --evaluate_during_training --logging_steps 4000 --drop_out 0.3 --weight_decay 0.05 --warmup_steps 0 --loss_type xe --img_feat_format tsv --classifier linear --cls_hidden_scale 2 --txt_data_dir /fsx/harman/data/VinVL_img_features/gqa --do_train --evaluate_during_training --num_workers 50 --logging_steps 200

# ## finetune 2mill iter oscar model on gqa  -ALL
python -m torch.distributed.launch --nproc_per_node=8 oscar/run_gqa_tsv.py --img_feature_dim 2054 --max_img_seq_length 50 --data_label_type all --train_data_type all --eval_data_type bal --img_feature_type faster_r-cnn --data_dir /fsx/harman/data/VinVL_img_features/gqa --model_type bert --model_name_or_path /fsx/harman/Oscar/pretrained_models/vqa/base/checkpoint-2000000 --task_name gqa --do_lower_case --max_seq_length 165 --per_gpu_eval_batch_size 256 --per_gpu_train_batch_size 48 --learning_rate 5e-05 --num_train_epochs 5 --output_dir /checkpoints/harman/oscar/gqa --label_file /fsx/harman/data/VinVL_img_features/gqa/trainval_testdev_all_ans2label.pkl --save_epoch 1 --seed 88 --evaluate_during_training --logging_steps 4000 --drop_out 0.3 --weight_decay 0.05 --warmup_steps 0 --loss_type xe --img_feat_format tsv --classifier linear --cls_hidden_scale 2 --txt_data_dir /fsx/harman/data/VinVL_img_features/gqa --do_train --evaluate_during_training --num_workers 50 --logging_steps 200


# run evaluation for gqa finetuned models
# oscar+ 2mill eg trained --> finetuned on gqa bal
python -m torch.distributed.launch --nproc_per_node=1 oscar/run_gqa_tsv.py --img_feature_dim 2054 --max_img_seq_length 50 --data_label_type all --train_data_type bal --eval_data_type bal --img_feature_type faster_r-cnn --data_dir /fsx/harman/data/VinVL_img_features/gqa --model_type bert --model_name_or_path /fsx/harman/Oscar/pretrained_models/vqa/base/checkpoint-2000000 --task_name gqa --do_lower_case --max_seq_length 165 --per_gpu_eval_batch_size 256 --per_gpu_train_batch_size 48 --learning_rate 5e-05 --num_train_epochs 5 --output_dir /checkpoints/harman/oscar/gqa --label_file /fsx/harman/data/VinVL_img_features/gqa/trainval_testdev_all_ans2label.pkl --save_epoch 1 --seed 88 --evaluate_during_training --logging_steps 4000 --drop_out 0.3 --weight_decay 0.05 --warmup_steps 0 --loss_type xe --img_feat_format tsv --classifier linear --cls_hidden_scale 2 --txt_data_dir /fsx/harman/data/VinVL_img_features/gqa --do_eval --evaluate_during_training --num_workers 50 --logging_steps 200 

### todo
# oscar+ 2mill eg trained --> finetuned on gqa bal
python -m torch.distributed.launch --nproc_per_node=1 oscar/run_gqa_tsv.py --img_feature_dim 2054 --max_img_seq_length 50 --data_label_type all --train_data_type bal --eval_data_type bal --img_feature_type faster_r-cnn --data_dir /fsx/harman/data/VinVL_img_features/gqa --model_type bert --model_name_or_path /checkpoints/harman/oscar/gqa/134/best-4 --task_name gqa --do_lower_case --max_seq_length 165 --per_gpu_eval_batch_size 256 --per_gpu_train_batch_size 48 --learning_rate 5e-05 --num_train_epochs 5 --output_dir /checkpoints/harman/oscar/gqa --label_file /fsx/harman/data/VinVL_img_features/gqa/trainval_testdev_all_ans2label.pkl --save_epoch 1 --seed 88 --evaluate_during_training --logging_steps 4000 --drop_out 0.3 --weight_decay 0.05 --warmup_steps 0 --loss_type xe --img_feat_format tsv --classifier linear --cls_hidden_scale 2 --txt_data_dir /fsx/harman/data/VinVL_img_features/gqa --do_eval --evaluate_during_training --num_workers 50 --logging_steps 200 

### todo
# with relation, coco-vg-vqa 240k iter finetuned gt tags --> finetuned on gqa bal
python -m torch.distributed.launch --nproc_per_node=1 oscar/run_gqa_tsv.py --img_feature_dim 2054 --max_img_seq_length 50 --data_label_type all --train_data_type bal --eval_data_type bal --img_feature_type faster_r-cnn --data_dir /fsx/harman/data/VinVL_img_features/gqa --model_type bert --model_name_or_path /checkpoints/harman/oscar/gqa/114/best-4 --task_name gqa --do_lower_case --max_seq_length 165 --per_gpu_eval_batch_size 256 --per_gpu_train_batch_size 48 --learning_rate 5e-05 --num_train_epochs 5 --output_dir /checkpoints/harman/oscar/gqa --label_file /fsx/harman/data/VinVL_img_features/gqa/trainval_testdev_all_ans2label.pkl --save_epoch 1 --seed 88 --evaluate_during_training --logging_steps 4000 --drop_out 0.3 --weight_decay 0.05 --warmup_steps 0 --loss_type xe --img_feat_format tsv --classifier linear --cls_hidden_scale 2 --txt_data_dir /fsx/harman/data/VinVL_img_features/gqa --do_eval --evaluate_during_training --num_workers 50 --logging_steps 200 

### todo
# without relation, coco-vg-vqa 240k iter finetuned gt tags --> finetuned on gqa bal
python -m torch.distributed.launch --nproc_per_node=1 oscar/run_gqa_tsv.py --img_feature_dim 2054 --max_img_seq_length 50 --data_label_type all --train_data_type bal --eval_data_type bal --img_feature_type faster_r-cnn --data_dir /fsx/harman/data/VinVL_img_features/gqa --model_type bert --model_name_or_path /checkpoints/harman/oscar/gqa/115/best-4 --task_name gqa --do_lower_case --max_seq_length 165 --per_gpu_eval_batch_size 256 --per_gpu_train_batch_size 48 --learning_rate 5e-05 --num_train_epochs 5 --output_dir /checkpoints/harman/oscar/gqa --label_file /fsx/harman/data/VinVL_img_features/gqa/trainval_testdev_all_ans2label.pkl --save_epoch 1 --seed 88 --evaluate_during_training --logging_steps 4000 --drop_out 0.3 --weight_decay 0.05 --warmup_steps 0 --loss_type xe --img_feat_format tsv --classifier linear --cls_hidden_scale 2 --txt_data_dir /fsx/harman/data/VinVL_img_features/gqa --do_eval --evaluate_during_training --num_workers 50 --logging_steps 200 




#### TESTING OUT WANDB RUN CONTINUATION
# --> RUN FROM SCARTCH GQA
CUDA_LAUNCH_BLOCKING=1 python -m torch.distributed.launch --nproc_per_node=8 oscar/run_gqa_tsv.py --img_feature_dim 2054 --max_img_seq_length 50 --data_label_type all --train_data_type all --eval_data_type bal --img_feature_type faster_r-cnn --data_dir /fsx/harman/data/VinVL_img_features/gqa --model_type bert --model_name_or_path /checkpoints/harman/oscar/88/checkpoint-0240000 --task_name gqa --do_lower_case --max_seq_length 165 --per_gpu_eval_batch_size 256 --per_gpu_train_batch_size 48 --learning_rate 5e-05 --num_train_epochs 5 --output_dir /checkpoints/harman/oscar/gqa --label_file /fsx/harman/data/VinVL_img_features/gqa/trainval_testdev_all_ans2label.pkl --save_epoch 1 --seed 88 --evaluate_during_training --drop_out 0.3 --weight_decay 0.05 --warmup_steps 0 --loss_type xe --img_feat_format tsv --classifier linear --cls_hidden_scale 2 --txt_data_dir /fsx/harman/data/VinVL_img_features/gqa --do_train --evaluate_during_training --num_workers 50 --logging_steps 200

# --> RERUN TO CONTINUE
CUDA_LAUNCH_BLOCKING=1 python -m torch.distributed.launch --nproc_per_node=8 oscar/run_gqa_tsv.py --img_feature_dim 2054 --max_img_seq_length 50 --data_label_type all --train_data_type all --eval_data_type bal --img_feature_type faster_r-cnn --data_dir /fsx/harman/data/VinVL_img_features/gqa --model_type bert --model_name_or_path /checkpoints/harman/oscar/88/checkpoint-0240000 --task_name gqa --do_lower_case --max_seq_length 165 --per_gpu_eval_batch_size 256 --per_gpu_train_batch_size 48 --learning_rate 5e-05 --num_train_epochs 5 --output_dir /checkpoints/harman/oscar/gqa --label_file /fsx/harman/data/VinVL_img_features/gqa/trainval_testdev_all_ans2label.pkl --save_epoch 1 --seed 88 --evaluate_during_training --drop_out 0.3 --weight_decay 0.05 --warmup_steps 0 --loss_type xe --img_feat_format tsv --classifier linear --cls_hidden_scale 2 --txt_data_dir /fsx/harman/data/VinVL_img_features/gqa --do_train --evaluate_during_training --num_workers 50 --logging_steps 200 --resume_ckpt 


### runing on GQA without torch.distributed

python oscar/run_gqa_tsv.py --img_feature_dim 2054 --max_img_seq_length 50 --data_label_type all --train_data_type all --eval_data_type bal --img_feature_type faster_r-cnn --data_dir /fsx/harman/data/VinVL_img_features/gqa --model_type bert --model_name_or_path /fsx/harman/Oscar/pretrained_models/vqa/base/checkpoint-2000000 --task_name gqa --do_lower_case --max_seq_length 165 --per_gpu_eval_batch_size 256 --per_gpu_train_batch_size 48 --learning_rate 5e-05 --num_train_epochs 5 --output_dir /checkpoints/harman/oscar/gqa --label_file /fsx/harman/data/VinVL_img_features/gqa/trainval_testdev_all_ans2label.pkl --save_epoch 1 --seed 88 --evaluate_during_training --logging_steps 4000 --drop_out 0.3 --weight_decay 0.05 --warmup_steps 0 --loss_type xe --img_feat_format tsv --classifier linear --cls_hidden_scale 2 --txt_data_dir /fsx/harman/data/VinVL_img_features/gqa --do_train --evaluate_during_training --num_workers 50 --logging_steps 200


# finetune on gqa oscar + rel predict (our pretrained model trianed on coco +vg + vqa dataset),  -- 87 model
python oscar/run_gqa_tsv.py --img_feature_dim 2054 --max_img_seq_length 50 --data_label_type all --train_data_type all --eval_data_type bal --img_feature_type faster_r-cnn --data_dir /fsx/harman/data/VinVL_img_features/gqa --model_type bert --model_name_or_path /checkpoints/harman/oscar/87/checkpoint-0240000 --task_name gqa --do_lower_case --max_seq_length 165 --per_gpu_eval_batch_size 256 --per_gpu_train_batch_size 48 --learning_rate 5e-05 --num_train_epochs 5 --output_dir /checkpoints/harman/oscar/gqa --label_file /fsx/harman/data/VinVL_img_features/gqa/trainval_testdev_all_ans2label.pkl --save_epoch 1 --seed 88 --evaluate_during_training --drop_out 0.3 --weight_decay 0.05 --warmup_steps 0 --loss_type xe --img_feat_format tsv --classifier linear --cls_hidden_scale 2 --txt_data_dir /fsx/harman/data/VinVL_img_features/gqa --do_train --evaluate_during_training --num_workers 50 --logging_steps 200

# finetune on gaa oscar + rel predict (our pretrained model trianed on coco +vg + vqa dataset),  -- 88 model
python oscar/run_gqa_tsv.py --img_feature_dim 2054 --max_img_seq_length 50 --data_label_type all --train_data_type all --eval_data_type bal --img_feature_type faster_r-cnn --data_dir /fsx/harman/data/VinVL_img_features/gqa --model_type bert --model_name_or_path /checkpoints/harman/oscar/88/checkpoint-0240000 --task_name gqa --do_lower_case --max_seq_length 165 --per_gpu_eval_batch_size 256 --per_gpu_train_batch_size 48 --learning_rate 5e-05 --num_train_epochs 5 --output_dir /checkpoints/harman/oscar/gqa --label_file /fsx/harman/data/VinVL_img_features/gqa/trainval_testdev_all_ans2label.pkl --save_epoch 1 --seed 88 --evaluate_during_training --drop_out 0.3 --weight_decay 0.05 --warmup_steps 0 --loss_type xe --img_feat_format tsv --classifier linear --cls_hidden_scale 2 --txt_data_dir /fsx/harman/data/VinVL_img_features/gqa --do_train --evaluate_during_training --num_workers 50 --logging_steps 200






# eval run 150 for gqa
python -m torch.distributed.launch --nproc_per_node=1 oscar/run_gqa_tsv.py --img_feature_dim 2054 --max_img_seq_length 50 --data_label_type all --train_data_type bal --eval_data_type bal --img_feature_type faster_r-cnn --data_dir /fsx/harman/data/VinVL_img_features/gqa --model_type bert --model_name_or_path /checkpoints/harman/oscar/gqa/150/best-3 --task_name gqa --do_lower_case --max_seq_length 165 --per_gpu_eval_batch_size 256 --per_gpu_train_batch_size 48 --learning_rate 5e-05 --num_train_epochs 5 --output_dir /checkpoints/harman/oscar/gqa --label_file /fsx/harman/data/VinVL_img_features/gqa/trainval_testdev_all_ans2label.pkl --save_epoch 1 --seed 88 --evaluate_during_training --logging_steps 4000 --drop_out 0.3 --weight_decay 0.05 --warmup_steps 0 --loss_type xe --img_feat_format tsv --classifier linear --cls_hidden_scale 2 --txt_data_dir /fsx/harman/data/VinVL_img_features/gqa --do_eval --evaluate_during_training --num_workers 50 --logging_steps 200 



# finetune 87--> different models on gqa bal
    python oscar/run_gqa_tsv.py --img_feature_dim 2054 --max_img_seq_length 50 --data_label_type all --train_data_type bal --eval_data_type bal --img_feature_type faster_r-cnn --data_dir /fsx/harman/data/VinVL_img_features/gqa --model_type bert --model_name_or_path /checkpoints/harman/oscar/87/checkpoint-0050000 --task_name gqa --do_lower_case --max_seq_length 165 --per_gpu_eval_batch_size 256 --per_gpu_train_batch_size 48 --learning_rate 5e-05 --num_train_epochs 5 --output_dir /checkpoints/harman/oscar/gqa --label_file /fsx/harman/data/VinVL_img_features/gqa/trainval_testdev_all_ans2label.pkl --save_epoch 1 --seed 88 --evaluate_during_training --drop_out 0.3 --weight_decay 0.05 --warmup_steps 0 --loss_type xe --img_feat_format tsv --classifier linear --cls_hidden_scale 2 --txt_data_dir /fsx/harman/data/VinVL_img_features/gqa --do_train --evaluate_during_training --num_workers 50 --logging_steps 200

    python oscar/run_gqa_tsv.py --img_feature_dim 2054 --max_img_seq_length 50 --data_label_type all --train_data_type bal --eval_data_type bal --img_feature_type faster_r-cnn --data_dir /fsx/harman/data/VinVL_img_features/gqa --model_type bert --model_name_or_path /checkpoints/harman/oscar/87/checkpoint-00100000 --task_name gqa --do_lower_case --max_seq_length 165 --per_gpu_eval_batch_size 256 --per_gpu_train_batch_size 48 --learning_rate 5e-05 --num_train_epochs 5 --output_dir /checkpoints/harman/oscar/gqa --label_file /fsx/harman/data/VinVL_img_features/gqa/trainval_testdev_all_ans2label.pkl --save_epoch 1 --seed 88 --evaluate_during_training --drop_out 0.3 --weight_decay 0.05 --warmup_steps 0 --loss_type xe --img_feat_format tsv --classifier linear --cls_hidden_scale 2 --txt_data_dir /fsx/harman/data/VinVL_img_features/gqa --do_train --evaluate_during_training --num_workers 50 --logging_steps 200

    python oscar/run_gqa_tsv.py --img_feature_dim 2054 --max_img_seq_length 50 --data_label_type all --train_data_type bal --eval_data_type bal --img_feature_type faster_r-cnn --data_dir /fsx/harman/data/VinVL_img_features/gqa --model_type bert --model_name_or_path /checkpoints/harman/oscar/87/checkpoint-0150000 --task_name gqa --do_lower_case --max_seq_length 165 --per_gpu_eval_batch_size 256 --per_gpu_train_batch_size 48 --learning_rate 5e-05 --num_train_epochs 5 --output_dir /checkpoints/harman/oscar/gqa --label_file /fsx/harman/data/VinVL_img_features/gqa/trainval_testdev_all_ans2label.pkl --save_epoch 1 --seed 88 --evaluate_during_training --drop_out 0.3 --weight_decay 0.05 --warmup_steps 0 --loss_type xe --img_feat_format tsv --classifier linear --cls_hidden_scale 2 --txt_data_dir /fsx/harman/data/VinVL_img_features/gqa --do_train --evaluate_during_training --num_workers 50 --logging_steps 200

    python oscar/run_gqa_tsv.py --img_feature_dim 2054 --max_img_seq_length 50 --data_label_type all --train_data_type bal --eval_data_type bal --img_feature_type faster_r-cnn --data_dir /fsx/harman/data/VinVL_img_features/gqa --model_type bert --model_name_or_path /checkpoints/harman/oscar/87/checkpoint-0200000 --task_name gqa --do_lower_case --max_seq_length 165 --per_gpu_eval_batch_size 256 --per_gpu_train_batch_size 48 --learning_rate 5e-05 --num_train_epochs 5 --output_dir /checkpoints/harman/oscar/gqa --label_file /fsx/harman/data/VinVL_img_features/gqa/trainval_testdev_all_ans2label.pkl --save_epoch 1 --seed 88 --evaluate_during_training --drop_out 0.3 --weight_decay 0.05 --warmup_steps 0 --loss_type xe --img_feat_format tsv --classifier linear --cls_hidden_scale 2 --txt_data_dir /fsx/harman/data/VinVL_img_features/gqa --do_train --evaluate_during_training --num_workers 50 --logging_steps 200

# finetune 87--> different models on gqa bal
    python oscar/run_gqa_tsv.py --img_feature_dim 2054 --max_img_seq_length 50 --data_label_type all --train_data_type bal --eval_data_type bal --img_feature_type faster_r-cnn --data_dir /fsx/harman/data/VinVL_img_features/gqa --model_type bert --model_name_or_path /checkpoints/harman/oscar/88/checkpoint-0050000 --task_name gqa --do_lower_case --max_seq_length 165 --per_gpu_eval_batch_size 256 --per_gpu_train_batch_size 48 --learning_rate 5e-05 --num_train_epochs 5 --output_dir /checkpoints/harman/oscar/gqa --label_file /fsx/harman/data/VinVL_img_features/gqa/trainval_testdev_all_ans2label.pkl --save_epoch 1 --seed 88 --evaluate_during_training --drop_out 0.3 --weight_decay 0.05 --warmup_steps 0 --loss_type xe --img_feat_format tsv --classifier linear --cls_hidden_scale 2 --txt_data_dir /fsx/harman/data/VinVL_img_features/gqa --do_train --evaluate_during_training --num_workers 50 --logging_steps 200

    python oscar/run_gqa_tsv.py --img_feature_dim 2054 --max_img_seq_length 50 --data_label_type all --train_data_type bal --eval_data_type bal --img_feature_type faster_r-cnn --data_dir /fsx/harman/data/VinVL_img_features/gqa --model_type bert --model_name_or_path /checkpoints/harman/oscar/88/checkpoint-00100000 --task_name gqa --do_lower_case --max_seq_length 165 --per_gpu_eval_batch_size 256 --per_gpu_train_batch_size 48 --learning_rate 5e-05 --num_train_epochs 5 --output_dir /checkpoints/harman/oscar/gqa --label_file /fsx/harman/data/VinVL_img_features/gqa/trainval_testdev_all_ans2label.pkl --save_epoch 1 --seed 88 --evaluate_during_training --drop_out 0.3 --weight_decay 0.05 --warmup_steps 0 --loss_type xe --img_feat_format tsv --classifier linear --cls_hidden_scale 2 --txt_data_dir /fsx/harman/data/VinVL_img_features/gqa --do_train --evaluate_during_training --num_workers 50 --logging_steps 200

    python oscar/run_gqa_tsv.py --img_feature_dim 2054 --max_img_seq_length 50 --data_label_type all --train_data_type bal --eval_data_type bal --img_feature_type faster_r-cnn --data_dir /fsx/harman/data/VinVL_img_features/gqa --model_type bert --model_name_or_path /checkpoints/harman/oscar/88/checkpoint-0150000 --task_name gqa --do_lower_case --max_seq_length 165 --per_gpu_eval_batch_size 256 --per_gpu_train_batch_size 48 --learning_rate 5e-05 --num_train_epochs 5 --output_dir /checkpoints/harman/oscar/gqa --label_file /fsx/harman/data/VinVL_img_features/gqa/trainval_testdev_all_ans2label.pkl --save_epoch 1 --seed 88 --evaluate_during_training --drop_out 0.3 --weight_decay 0.05 --warmup_steps 0 --loss_type xe --img_feat_format tsv --classifier linear --cls_hidden_scale 2 --txt_data_dir /fsx/harman/data/VinVL_img_features/gqa --do_train --evaluate_during_training --num_workers 50 --logging_steps 200

    python oscar/run_gqa_tsv.py --img_feature_dim 2054 --max_img_seq_length 50 --data_label_type all --train_data_type bal --eval_data_type bal --img_feature_type faster_r-cnn --data_dir /fsx/harman/data/VinVL_img_features/gqa --model_type bert --model_name_or_path /checkpoints/harman/oscar/88/checkpoint-0200000 --task_name gqa --do_lower_case --max_seq_length 165 --per_gpu_eval_batch_size 256 --per_gpu_train_batch_size 48 --learning_rate 5e-05 --num_train_epochs 5 --output_dir /checkpoints/harman/oscar/gqa --label_file /fsx/harman/data/VinVL_img_features/gqa/trainval_testdev_all_ans2label.pkl --save_epoch 1 --seed 88 --evaluate_during_training --drop_out 0.3 --weight_decay 0.05 --warmup_steps 0 --loss_type xe --img_feat_format tsv --classifier linear --cls_hidden_scale 2 --txt_data_dir /fsx/harman/data/VinVL_img_features/gqa --do_train --evaluate_during_training --num_workers 50 --logging_steps 200



    # run gqa eval 

python -m torch.distributed.launch --nproc_per_node=1 oscar/run_gqa_tsv.py --img_feature_dim 2054 --max_img_seq_length 50 --data_label_type all --train_data_type bal --eval_data_type bal --img_feature_type faster_r-cnn --data_dir /fsx/harman/data/VinVL_img_features/gqa --model_type bert --model_name_or_path /checkpoints/harman/oscar/gqa/167/best-4 --task_name gqa --do_lower_case --max_seq_length 165 --per_gpu_eval_batch_size 256 --per_gpu_train_batch_size 48 --learning_rate 5e-05 --num_train_epochs 5 --output_dir /checkpoints/harman/oscar/gqa --label_file /fsx/harman/data/VinVL_img_features/gqa/trainval_testdev_all_ans2label.pkl --save_epoch 1 --seed 88 --evaluate_during_training --logging_steps 4000 --drop_out 0.3 --weight_decay 0.05 --warmup_steps 0 --loss_type xe --img_feat_format tsv --classifier linear --cls_hidden_scale 2 --txt_data_dir /fsx/harman/data/VinVL_img_features/gqa --do_eval --evaluate_during_training --num_workers 50 --logging_steps 2000 

python -m torch.distributed.launch --nproc_per_node=1 oscar/run_gqa_tsv.py --img_feature_dim 2054 --max_img_seq_length 50 --data_label_type all --train_data_type bal --eval_data_type bal --img_feature_type faster_r-cnn --data_dir /fsx/harman/data/VinVL_img_features/gqa --model_type bert --model_name_or_path /checkpoints/harman/oscar/gqa/166/best-4 --task_name gqa --do_lower_case --max_seq_length 165 --per_gpu_eval_batch_size 256 --per_gpu_train_batch_size 48 --learning_rate 5e-05 --num_train_epochs 5 --output_dir /checkpoints/harman/oscar/gqa --label_file /fsx/harman/data/VinVL_img_features/gqa/trainval_testdev_all_ans2label.pkl --save_epoch 1 --seed 88 --evaluate_during_training --logging_steps 4000 --drop_out 0.3 --weight_decay 0.05 --warmup_steps 0 --loss_type xe --img_feat_format tsv --classifier linear --cls_hidden_scale 2 --txt_data_dir /fsx/harman/data/VinVL_img_features/gqa --do_eval --evaluate_during_training --num_workers 50 --logging_steps 2000 

python -m torch.distributed.launch --nproc_per_node=1 oscar/run_gqa_tsv.py --img_feature_dim 2054 --max_img_seq_length 50 --data_label_type all --train_data_type bal --eval_data_type bal --img_feature_type faster_r-cnn --data_dir /fsx/harman/data/VinVL_img_features/gqa --model_type bert --model_name_or_path /checkpoints/harman/oscar/gqa/163/best-3 --task_name gqa --do_lower_case --max_seq_length 165 --per_gpu_eval_batch_size 256 --per_gpu_train_batch_size 48 --learning_rate 5e-05 --num_train_epochs 5 --output_dir /checkpoints/harman/oscar/gqa --label_file /fsx/harman/data/VinVL_img_features/gqa/trainval_testdev_all_ans2label.pkl --save_epoch 1 --seed 88 --evaluate_during_training --logging_steps 4000 --drop_out 0.3 --weight_decay 0.05 --warmup_steps 0 --loss_type xe --img_feat_format tsv --classifier linear --cls_hidden_scale 2 --txt_data_dir /fsx/harman/data/VinVL_img_features/gqa --do_eval --evaluate_during_training --num_workers 50 --logging_steps 2000 


python -m torch.distributed.launch --nproc_per_node=1 oscar/run_gqa_tsv.py --img_feature_dim 2054 --max_img_seq_length 50 --data_label_type all --train_data_type bal --eval_data_type bal --img_feature_type faster_r-cnn --data_dir /fsx/harman/data/VinVL_img_features/gqa --model_type bert --model_name_or_path /checkpoints/harman/oscar/gqa/177/best-3 --task_name gqa --do_lower_case --max_seq_length 165 --per_gpu_eval_batch_size 256 --per_gpu_train_batch_size 48 --learning_rate 5e-05 --num_train_epochs 5 --output_dir /checkpoints/harman/oscar/gqa --label_file /fsx/harman/data/VinVL_img_features/gqa/trainval_testdev_all_ans2label.pkl --save_epoch 1 --seed 88 --evaluate_during_training --logging_steps 4000 --drop_out 0.3 --weight_decay 0.05 --warmup_steps 0 --loss_type xe --img_feat_format tsv --classifier linear --cls_hidden_scale 2 --txt_data_dir /fsx/harman/data/VinVL_img_features/gqa --do_eval --evaluate_during_training --num_workers 50 --logging_steps 2000 



## Pretraining with More relation losses. relation losses
## debug
python -m torch.distributed.launch --nproc_per_node=1 oscar/run_oscarplus_pretrain.py \
    --use_b 1 \
    --max_grad_norm 10.0 --gradient_accumulation_steps 1 \
    --use_img_layernorm 1 \
    --output_dir /checkpoints/harman/oscar/ \
    --bert_model bert --model_name_or_path bert-base-uncased \
    --do_lower_case --learning_rate 5e-05 \
    --warmup_steps 0 --do_train --max_seq_length 35 --on_memory \
    --max_img_seq_length 50 --img_feature_dim 2054 \
    --drop_out 0.1 --train_batch_size 32 \
    --ckpt_period 10000 --max_iters 2000000 --log_period 100 \
    --data_dir /fsx/harman/Oscar/yaml_files --dataset_file coco_sg.yaml \
    --textb_sample_mode 1 --texta_false_prob 0.25 --num_workers 10 \
    --use_sg --output_hidden_states --obj_relation_vocab_size 51 --max_datapoints 10000


python -m torch.distributed.launch --nproc_per_node=1 oscar/run_oscarplus_pretrain.py \
    --use_b 1 \
    --max_grad_norm 10.0 --gradient_accumulation_steps 1 \
    --use_img_layernorm 1 \
    --output_dir /checkpoints/harman/oscar/ \
    --bert_model bert --model_name_or_path bert-base-uncased \
    --do_lower_case --learning_rate 5e-05 \
    --warmup_steps 0 --do_train --max_seq_length 50 --on_memory \
    --max_img_seq_length 50 --img_feature_dim 2054 \
    --drop_out 0.1 --train_batch_size 32 \
    --ckpt_period 10000 --max_iters 2000000 --log_period 100 \
    --data_dir /fsx/harman/Oscar/yaml_files --dataset_file coco_sg.yaml \
    --textb_sample_mode 1 --texta_false_prob 0.25 --num_workers 10 \
    --use_sg --output_hidden_states --obj_relation_vocab_size 51 --max_datapoints 1000


# debug all lm labels -1 error
python -m torch.distributed.launch --nproc_per_node=1 oscar/run_oscarplus_pretrain.py \
    --use_b 1 \
    --max_grad_norm 10.0 --gradient_accumulation_steps 1 \
    --use_img_layernorm 1 \
    --output_dir /checkpoints/harman/oscar/ \
    --bert_model bert --model_name_or_path bert-base-uncased \
    --do_lower_case --learning_rate 5e-05 \
    --warmup_steps 0 --do_train --max_seq_length 50 --on_memory \
    --max_img_seq_length 50 --img_feature_dim 2054 \
    --drop_out 0.1 --train_batch_size 32 \
    --ckpt_period 10000 --max_iters 2000000 --log_period 100 \
    --data_dir /fsx/harman/Oscar/yaml_files --dataset_file coco_sg.yaml \
    --textb_sample_mode 1 --texta_false_prob 0.25 --num_workers 0 \
    --output_hidden_states --obj_relation_vocab_size 51 --max_datapoints 1000