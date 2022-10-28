from __future__ import absolute_import, division, print_function

import argparse
import datetime
import json
import logging
import os
import random
import sys
import time
import math
import shutil

sys.path.insert(0, '.')

import numpy as np
import torch

import transformers
from oscar.modeling.modeling_bert import BertImgForPreTraining
# from transformers.pytorch_transformers import (WEIGHTS_NAME, BertConfig,
#                                   BertTokenizer)
# from transformers import (WEIGHTS_NAME, BertConfig,
#                                   BertTokenizer)
from transformers import WEIGHTS_NAME
from transformers import AutoConfig, AutoTokenizer

from oscar.datasets.build import make_data_loader

# from transformers.pytorch_transformers import AdamW, WarmupLinearSchedule
from transformers.optimization import AdamW #, WarmupLinearSchedule
from transformers import get_linear_schedule_with_warmup
from oscar.utils.misc import mkdir, get_rank
from oscar.utils.metric_logger import TensorboardLogger

logger = logging.getLogger(__name__)

# from transformers.models.bert.modeling_bert import BERT_PRETRAINED_MODEL_ARCHIVE_LIST
# BertConfig.pretrained_config_archive_map = BERT_PRETRAINED_MODEL_ARCHIVE_LIST # This is a hack again, BERT_PRETRAINED_MODEL_ARCHIVE_LIST was discarded after transformers version 2.0.0
# import pdb; pdb.set_trace()
#The folliwng should not be required
# ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig,)), ())
# ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map) for conf in (BertConfig,)), ())


# MODEL_CLASSES = {
#     'bert': (BertConfig, BertImgForPreTraining, BertTokenizer),
# }

""" ****** Pretraining ****** """

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=False,
                        help="The input data dir. "
                             "Should contain the .yaml files for the task.")
    parser.add_argument("--dataset_file", default=None, type=str, required=True,
                        help="The training dataset yaml file.")
    parser.add_argument("--extra_dataset_file", default=None, type=str, required=False,
                        help="The extra training dataset yaml file.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints will be written.")

    # image chunks
    parser.add_argument("--chunk_start_id", default=-1, type=int,
                        help="Image Chunk Start ID")
    parser.add_argument("--chunk_end_id", default=-1, type=int,
                        help="Image Chunk End ID")

    ## Image parameters
    parser.add_argument("--max_img_seq_length", default=50, type=int,
                        help="The maximum total input image sequence length.")
    parser.add_argument("--img_feature_dim", default=2054, type=int,
                        help="The Image Feature Dimension.")
    parser.add_argument("--img_feature_type", default='faster_r-cnn', type=str,
                        help="faster_r-cnn or mask_r-cnn")
    parser.add_argument("--use_layernorm", action='store_true',
                        help="use_layernorm")

    parser.add_argument("--drop_out", default=0.1, type=float,
                        help="Drop out for BERT.")

    parser.add_argument("--use_b", type=int, default=1, help="use_b")
    parser.add_argument("--textb_sample_mode", type=int, default=0,
                        help="0: sample from both texta&textb, "
                             "1: sample from textb, "
                             "2: sample from QA answers")
    parser.add_argument("--extra_textb_sample_mode", type=int, default=1)
    parser.add_argument("--texta_false_prob", type=float, default=0.0,
                        help="the probality that we sample wrong texta, should in [0.0, 0.5]")

    parser.add_argument("--model_name_or_path", default=None, type=str,
                        required=True,
                        help="Path to pre-trained model or shortcut name eg 'bert-base-uncased' ")
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")

    parser.add_argument("--max_seq_length", default=35, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--max_iters", default=2000000, type=int,
                        help="Maximal number of training iterations.")
    parser.add_argument("--train_batch_size", default=1024, type=int,
                        help="Batch size for training.")
    parser.add_argument("--num_workers", default=6, type=int,
                        help="Number of workers for dataset.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--optim", default='adamw', type=str,
                        help="The optimizer used for Bert, [adamw, lamb], default: adamw")
    parser.add_argument("--max_grad_norm", default=-1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--on_memory", action='store_true',
                        help="Whether to load train samples into memory or use disk")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumualte before performing a backward/update pass.")

    parser.add_argument("--from_scratch", action='store_true',
                        help="train from scratch")
    parser.add_argument("--use_img_layernorm", type=int, default=0,
                        help="Normalize image features with bertlayernorm")
    parser.add_argument("--img_layer_norm_eps", default=1e-12, type=float,
                        help="The eps in image feature laynorm layer")


    ## scene graph related
    parser.add_argument("--use_sg", action='store_true',
                        help="Whether to use scene graph information while training")
    parser.add_argument("--use_objfeat_rel_loss", action='store_true',
                        help="Whether to use objfeat_rel_loss while training")
    parser.add_argument("--use_objtag_rel_loss", action='store_true',
                        help="Whether to use objtag_rel_loss while training")
    parser.add_argument("--max_rel_length", default=100, type=int,
                        help="The maximum total input number of relations \n"
                             "Sequences longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--obj_relation_vocab_size",  type=int,
                        help="The maximum total input number of relations in the vocabulary")
    parser.add_argument("--loss_weight_rel_classif", type=float, default=1.0,
                        help="the loss weight for the relation classification")
                             

    ##Model extra stuff
    parser.add_argument("--output_hidden_states", action='store_true',
                        help="output the hidden states from the bert model or not")
    
    ## debug related
    parser.add_argument("--max_datapoints", type=int, default=-1,
                        help="max datapoints to load")
    parser.add_argument("--debug", action='store_true',
                        help="debug mode")
                        
    # distributed
    parser.add_argument('--gpu_ids', type=str, default='-1')
    parser.add_argument("--mask_loss_for_unmatched", type=int, default=1,
                        help="masked language model loss for unmatched triplets")
    parser.add_argument("--extra_loss_weight", type=float, default=0.0,
                        help="the loss weight for the extra train data batch (should be in [0,1])")
    parser.add_argument(
        "--use_gtlabels",
        type=int, default=1,
        help="use groundtruth labels for text b or not"
    )
    # logging
    parser.add_argument('--ckpt_period', type=int, default=10000,
                        help="Period for saving checkpoint")
    parser.add_argument('--log_period', type=int, default=100,
                        help="Period for saving logging info")
    args = parser.parse_args()

    import wandb

    if args.local_rank == 0:  # only on main process
        # Initialize wandb run
        wandb.init(entity="harman", project = 'SGVL', config=args)
        wandb.run.name = wandb.run.name.split('-')[-1] ## a number is sufficient for the run name
        args.output_dir = os.path.join(args.output_dir, wandb.run.name)



    MODEL_CLASSES = {
        'bert': (AutoConfig.from_pretrained(args.bert_model), BertImgForPreTraining, AutoTokenizer.from_pretrained(args.bert_model)),
    }


    if args.gpu_ids != '-1':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    print(f'localrank = {args.local_rank}')

    args.local_rank = int(os.environ["LOCAL_RANK"])
    print(f'localrank = {args.local_rank}')
################
    args.num_gpus = int(os.environ["WORLD_SIZE"])
################
    args.distributed = args.num_gpus > 1

    if args.gpu_ids != '-1':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        logger.info("Output Directory Exists.")
    
    if not os.path.exists(args.output_dir) and args.do_train:
        logger.info("Output Directory doesnot exist, creating it")
        os.makedirs(args.output_dir, exist_ok=True)

    print(f'numgpu = {args.num_gpus}, worldsize = {os.environ["WORLD_SIZE"]}, gpu ids = {args.gpu_ids}')

    # Setup CUDA, GPU & distributed training
    #####################
    if "WORLD_SIZE" in os.environ:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1
    ngpus_per_node = torch.cuda.device_count()

    if args.distributed:
        if args.local_rank != -1: # for torch.distributed.launch
            args.rank = args.local_rank
            args.gpu = args.local_rank
        elif 'SLURM_PROCID' in os.environ: # for slurm scheduler
            args.rank = int(os.environ['SLURM_PROCID'])
            args.gpu = args.rank % torch.cuda.device_count()
    #####################

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(
            backend='nccl', init_method="env://",
                                world_size=args.world_size, rank=args.local_rank
        )
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
        args.local_rank, device, args.n_gpu, bool(args.local_rank != -1)
    )

    if args.gradient_accumulation_steps < 1:
        raise ValueError(
            "Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                args.gradient_accumulation_steps))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train:
        raise ValueError(
            "Training is currently the only implemented execution option. Please set `do_train`.")

    if not os.path.exists(args.output_dir):
        mkdir(args.output_dir)

    last_checkpoint_dir = None
    arguments = {"iteration": 0}
    if os.path.exists(args.output_dir):
        save_file = os.path.join(args.output_dir, "last_checkpoint")
        try:
            with open(save_file, "r") as f:
                last_saved = f.read()
                last_saved = last_saved.strip()
        except IOError:
            # if file doesn't exist, maybe because it has just been
            # deleted by a separate process
            last_saved = ""
        if last_saved:
            folder_name = os.path.splitext(last_saved.split('/')[0])[0] # in the form of checkpoint-00001 or checkpoint-00001/pytorch_model.bin
            last_checkpoint_dir = os.path.join(args.output_dir, folder_name)
            arguments["iteration"] = int(folder_name.split('-')[-1])
            assert os.path.isfile(os.path.join(last_checkpoint_dir, WEIGHTS_NAME)), "Last_checkpoint detected, but file not found!"

    # model first
    if get_rank() != 0:
        torch.distributed.barrier()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.bert_model]
    if last_checkpoint_dir is not None:  # recovery
        args.model_name_or_path = last_checkpoint_dir
        logger.info(" -> Recovering model from {}".format(last_checkpoint_dir))

    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
    )
    # the folliwng will be used in Imgbert class, since bert by default has no support for images, we are making support
    config.img_layer_norm_eps = args.img_layer_norm_eps
    config.use_img_layernorm = args.use_img_layernorm

    # TODO: Not Clear
        # - what is num contrast classes
        # - what is use_b
        # - what is texta, textb
    # discrete code
    config.img_feature_dim = args.img_feature_dim
    config.img_feature_type = args.img_feature_type
    config.hidden_dropout_prob = args.drop_out
    if args.texta_false_prob < 0.5 and (args.texta_false_prob > 0 or not args.use_b):
        args.num_contrast_classes = 3
    else:
        args.num_contrast_classes = 2
    config.num_contrast_classes = args.num_contrast_classes
    config.output_hidden_states = args.output_hidden_states

    config.obj_relation_vocab_size = args.obj_relation_vocab_size
    config.use_sg = args.use_sg

    # Prepare model
    model = BertImgForPreTraining.from_pretrained(
        args.model_name_or_path,
        config=config, ignore_mismatched_sizes=True)

    

    # train from scratch
    if args.from_scratch:
        if last_checkpoint_dir is None:
            logger.info("Training from scratch ... ")
            model.apply(model.init_weights)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        'Total Parameters: {}'.format(total_params))

    for key, val in vars(config).items():
        setattr(args, key, val)

    if get_rank() == 0 and args.local_rank != -1:
        torch.distributed.barrier()

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    tb_log_dir = os.path.join(args.output_dir, 'train_logs')
    meters = TensorboardLogger(
        log_dir=tb_log_dir,
        delimiter="  ",
    )

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if
                    not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if
                    any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters,
                              lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                     num_warmup_steps=args.warmup_steps,
                                     num_training_steps=args.max_iters)

    if arguments['iteration'] > 0 and os.path.isfile(os.path.join(last_checkpoint_dir, 'optimizer.pth')):  # recovery
        logger.info(
            "Load BERT optimizer from {}".format(last_checkpoint_dir))
        optimizer_to_load = torch.load(
            os.path.join(last_checkpoint_dir, 'optimizer.pth'),
            map_location=torch.device("cpu"))
        optimizer.load_state_dict(optimizer_to_load.pop("optimizer"))
        scheduler.load_state_dict(optimizer_to_load.pop("scheduler"))

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank,
            find_unused_parameters=False)
    elif args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # train_examples = None
    train_dataloaders = make_data_loader(
        args, is_distributed=args.distributed, arguments=arguments
    )

    if isinstance(train_dataloaders, list):
        train_dataloader = train_dataloaders[0]
    else:
        train_dataloader = train_dataloaders
    train_dataloader_extra = [None] * len(train_dataloader)
    if isinstance(train_dataloaders, list) and len(train_dataloaders) > 1:
        logger.info("Having two train dataloaders!")
        train_dataloader_extra = train_dataloaders[1]
    tokenizer = train_dataloader.dataset.tokenizer

    # torch.backends.cudnn.benchmark = True

    max_iter = len(train_dataloader)
    start_iter = arguments["iteration"]
    logger.info("***** Running training *****")
    logger.info(" Num examples = {}".format(len(train_dataloader.dataset)))
    logger.info("  Instantaneous batch size = %d",
                args.train_batch_size // args.gradient_accumulation_steps)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size)
    logger.info("  Gradient Accumulation steps = %d",
                args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d",
                max_iter // args.gradient_accumulation_steps)

    log_json = {}

    model.train()
    model.zero_grad()

    clock_started = False
    # Every args.ckpt_period, report train_score and save model
    # tr_loss = 0
    loss_dict = {'tr_loss':0, 'loss_oscar':0, 'loss_rel_classif':0}
    nb_tr_examples, nb_tr_steps = 0, 0
    for step, (batch, batch_extra) in enumerate(zip(train_dataloader, train_dataloader_extra), start_iter):
        if not clock_started:
            start_training_time = time.time()
            end = time.time()
            clock_started = True

        def data_process(mini_batch):
            images_feat_and_len, targets, sg, qa_inds = \
                mini_batch[0], mini_batch[1], mini_batch[2], mini_batch[3]
            # images, targets, qa_inds = \
            #     mini_batch[0], mini_batch[1], mini_batch[2] # each element of minibatch is just a list of elements

            images_feat_and_len_transposed = list(zip(*images_feat_and_len))
            images = torch.stack(images_feat_and_len_transposed[0]).to(args.device, non_blocking=True)
            images_feat_len = torch.tensor(list(images_feat_and_len_transposed[1])).to(args.device, non_blocking=True)

            targets_transposed = list(zip(*targets))
            input_ids = torch.stack(targets_transposed[0]).to(args.device, non_blocking=True)
            input_mask = torch.stack(targets_transposed[1]).to(args.device, non_blocking=True)
            segment_ids = torch.stack(targets_transposed[2]).to(args.device, non_blocking=True)
            lm_label_ids = torch.stack(targets_transposed[3]).to(args.device, non_blocking=True)
            is_next = torch.stack(targets_transposed[4]).to(args.device, non_blocking=True)
            is_img_match = torch.stack(targets_transposed[5]).to(args.device, non_blocking=True)


            # # sg relates
            if args.use_sg:

                sg_transposed = list(zip(*sg))
                sg_rel_idx_pairs = torch.stack(sg_transposed[0]).to(args.device, non_blocking=True)
                mask_sg_rel_idx_pairs = torch.stack(sg_transposed[1]).to(args.device, non_blocking=True)
                sg_rel_labels = torch.stack(sg_transposed[2]).to(args.device, non_blocking=True)
                mask_sg_rel_labels = torch.stack(sg_transposed[3]).to(args.device, non_blocking=True)
                sg_num_rels = sg_transposed[4]
                # import pdb; pdb.set_trace()

                sg_rel_idx_pairs = sg_rel_idx_pairs.to(args.device, non_blocking=True)
                mask_sg_rel_idx_pairs = mask_sg_rel_idx_pairs.to(args.device, non_blocking=True)
                sg_rel_labels = sg_rel_labels.to(args.device, non_blocking=True)
                mask_sg_rel_labels = mask_sg_rel_labels.to(args.device, non_blocking=True)

            else:
                sg_rel_idx_pairs, mask_sg_rel_idx_pairs, sg_rel_labels, mask_sg_rel_labels, sg_num_rels = None, None, None, None, None
            
            return images, sg_rel_idx_pairs, mask_sg_rel_idx_pairs, sg_rel_labels, mask_sg_rel_labels, sg_num_rels, input_ids, input_mask, segment_ids, lm_label_ids, is_next


        images1, sg_rel_idx_pairs1, mask_sg_rel_idx_pairs1, sg_rel_labels1, mask_sg_rel_labels1, sg_num_rels1, input_ids1, input_mask1, segment_ids1, lm_label_ids1, is_next1 \
            = data_process(batch)

        if batch_extra is not None:
            images2, sg_rel_idx_pairs2, mask_sg_rel_idx_pairs2, sg_rel_labels2, mask_sg_rel_labels2, sg_num_rels2, input_ids2, input_mask2, segment_ids2, lm_label_ids2, is_next2 \
                = data_process(batch_extra)

        data_time = time.time() - end

        def forward_backward(images, sg_rel_idx_pairs, mask_sg_rel_idx_pairs, sg_rel_labels, mask_sg_rel_labels, sg_num_rels, input_ids, input_mask, segment_ids,
                             lm_label_ids, is_next, args, loss_weight=1.0, loss_weight_rel_classif=1.0):
            # feature as input
            # image_features = torch.stack(images).to(args.device, non_blocking=True)

            outputs = model(input_ids, segment_ids, input_mask,
                            lm_label_ids, is_next, img_feats=images, 
                            sg_rel_idx_pairs = sg_rel_idx_pairs,
                            sg_rel_labels = sg_rel_labels,
                            )

            loss_oscar = loss_weight * outputs[0]
            loss_rel_classif = torch.tensor(0.)
            if args.use_sg:
                loss_rel_classif = loss_weight_rel_classif * outputs[5]

            if args.n_gpu > 1:
                loss_oscar = loss_oscar.mean()  # mean() to average on multi-gpu.
                if args.use_sg:
                    loss_rel_classif = loss_rel_classif.mean()

            if args.gradient_accumulation_steps > 1:
                loss_oscar = loss_oscar / args.gradient_accumulation_steps
                if args.use_sg:
                    loss_rel_classif = loss_rel_classif / args.gradient_accumulation_steps
            
            total_loss = loss_oscar + loss_rel_classif
            total_loss.backward()

            return {'tr_loss':total_loss.item(), 'loss_oscar': loss_oscar.item(), 'loss_rel_classif': loss_rel_classif.item()}, input_ids.size(0)

        start1 = time.time()
        loss_dict1, nb_tr_example1 = forward_backward(
            images1, sg_rel_idx_pairs1, mask_sg_rel_idx_pairs1, sg_rel_labels1, mask_sg_rel_labels1, sg_num_rels1,
            input_ids1, input_mask1,
            segment_ids1, lm_label_ids1, is_next1,
            args, loss_weight=1.0-args.extra_loss_weight, loss_weight_rel_classif = args.loss_weight_rel_classif
        )
        for k in loss_dict.keys():
            loss_dict[k] += loss_dict1[k]
        nb_tr_examples += nb_tr_example1
        compute_time1 = time.time() - start1

        loss_dict2, nb_tr_example2 = {'tr_loss':0, 'loss_oscar':0, 'loss_rel_classif':0}, 0
        compute_time2 = 0.0
        if batch_extra is not None:
            start2 = time.time()
            loss_dict2, nb_tr_example2 = forward_backward(
                images2, sg_rel_idx_pairs2, mask_sg_rel_idx_pairs2, sg_rel_labels2, mask_sg_rel_labels2, sg_num_rels2,
                input_ids2, input_mask2,
                segment_ids2, lm_label_ids2, is_next2,
                loss_weight=args.extra_loss_weight, loss_weight_rel_classif = args.loss_weight_rel_classif
            )
            for k in loss_dict.keys():
                loss_dict[k] += loss_dict2[k]
            nb_tr_examples += nb_tr_example2
            compute_time2 = time.time() - start2

        nb_tr_steps += 1
        arguments["iteration"] = step + 1

        if (step + 1) % args.gradient_accumulation_steps == 0: # gradients will flow at this point
            # do gradient clipping
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            # do the optimization steps
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            optimizer.zero_grad()

            # measure elapsed time
            batch_time = time.time() - end
            end = time.time()
            metrics_to_log = {
                'time_info': {'compute': batch_time, 'data': data_time,
                              'compute1': compute_time1,
                              'compute2': compute_time2},
                'batch_metrics': {k: loss_dict1[k]+loss_dict2[k] for k in loss_dict.keys()}
            }
            params_to_log = {'params': {'bert_lr': optimizer.param_groups[0]["lr"]}}

            if args.local_rank == 0:  # only on main process
                wandb.log(metrics_to_log)
                wandb.log(params_to_log)

            meters.update_metrics(metrics_to_log)
            meters.update_params(params_to_log)

            if args.log_period > 0 and (step + 1) % args.log_period == 0:
                avg_time = meters.meters['time_info']['compute'].global_avg
                eta_seconds = avg_time * (max_iter - step - 1)
                eta_string = str(
                    datetime.timedelta(seconds=int(eta_seconds)))
                logger.info(
                    meters.delimiter.join(
                        [
                            "eta: {eta}",
                            "iter: {iter}",
                            "max mem: {memory:.0f}",
                        ]
                    ).format(
                        eta=eta_string,
                        iter=step + 1,
                        memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                    ) + "\n    " + meters.get_logs(step + 1)
                )

        if (step + 1) == max_iter or (step + 1) % args.ckpt_period == 0:  # Save a trained model
            log_json[step+1] = loss_dict['tr_loss']            
            train_metrics_total = torch.Tensor([loss_dict['tr_loss'], nb_tr_examples, nb_tr_steps]).to(args.device)
            torch.distributed.all_reduce(train_metrics_total)
            # reset metrics
            
            loss_dict['tr_loss'] = 0
            nb_tr_examples, nb_tr_steps = 0, 0

            if get_rank() == 0:
                # report metrics
                train_score_gathered = train_metrics_total[0] / \
                                       train_metrics_total[2]
                logger.info("PROGRESS: {}%".format(
                    round(100 * (step + 1) / max_iter, 4)))
                logger.info(
                    "EVALERR: {}%".format(train_score_gathered))
                meters.update_metrics(
                    {
                        'epoch_metrics': {'ex_cnt': train_metrics_total[1],
                                          'loss': train_score_gathered}
                    }
                )
                with open(os.path.join(args.output_dir, 'loss_logs.json'),
                          'w') as fp:
                    json.dump(log_json, fp)

                # save checkpoint
                output_dir = os.path.join(args.output_dir,
                                          'checkpoint-{:07d}'.format(
                                              step + 1))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                model_to_save = model.module if hasattr(
                    model,
                    'module') else model  # Take care of distributed/parallel training
                optimizer_to_save = {
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict()}

                save_num = 0
                while save_num < 10:
                    try:
                        model_to_save.save_pretrained(output_dir)
                        torch.save(args, os.path.join(output_dir,
                                                      'training_args.bin'))
                        tokenizer.save_pretrained(output_dir)
                        torch.save(optimizer_to_save,
                                   os.path.join(output_dir,
                                                'optimizer.pth'))
                        save_file = os.path.join(args.output_dir, "last_checkpoint")
                        with open(save_file, "w") as f:
                            f.write('checkpoint-{:07d}/pytorch_model.bin'.format(step + 1))
                        break
                    except:
                        save_num += 1
                logger.info(
                    "Saving model checkpoint {0} to {1}".format(
                        step + 1, output_dir))

    if clock_started:
        total_training_time = time.time() - start_training_time
    else:
        total_training_time = 0.0
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / max_iter
        )
    )
    # close the tb logger
    meters.close()


if __name__ == "__main__":
    main()
