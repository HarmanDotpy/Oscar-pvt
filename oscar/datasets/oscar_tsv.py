import os
import time
import json
import logging
import random
import glob
import base64
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset
from oscar.utils.tsv_file import TSVFile
from oscar.utils.misc import load_from_yaml_file

import sys
sys.path.append('/fsx/harman/sgg_benchmark_pytorch17/scene_graph_benchmark')
import pickle


class OscarTSVDataset(Dataset):
    def __init__(self, yaml_file, args=None, tokenizer=None, seq_len=35,
                 encoding="utf-8", corpus_lines=None, on_memory=True,
                 **kwargs):
        self.cfg = load_from_yaml_file(yaml_file)
        self.root = os.path.dirname(yaml_file)
        self.vocab = tokenizer.vocab
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.on_memory = on_memory
        self.use_sg = args.use_sg
        self.max_datapoints = args.max_datapoints

        self.corpus_lines = corpus_lines  # number of non-empty lines in input corpus
        self.corpus_tsvfile = TSVFile(os.path.join(self.root, self.cfg['corpus_file'])) # having image id, and text
        if 'textb_sample_mode' in kwargs:
            self.textb_sample_mode = kwargs['textb_sample_mode']
        else:
            self.textb_sample_mode = args.textb_sample_mode

        self.datasets_names = self.cfg['corpus'].split('_')
        self.datasets_with_splits = ['googlecc', 'sbu', 'oi', 'objects365', 'tagoi']
        self.datasets_with_onesplit = ['coco', 'flickr30k', 'gqa']
        logging.info('Datasets: {}'.format(','.join(self.datasets_names)))
        self.image_label_path = self.cfg['image_label_path']
        for key, val in self.image_label_path.items():
            # get the absolute path
            if key in self.datasets_names:
                self.image_label_path[key] = os.path.join(self.root, val)
        self.image_feature_path = self.cfg['image_feature_path']
        self.image_file_name = 'features.tsv'

        # import pdb; pdb.set_trace() # check the lines below
        if args.data_dir is not None:
            for key, val in self.image_feature_path.items():
                # get the absolute path
                if key in self.datasets_names:
                    self.image_feature_path[key] = os.path.join(args.data_dir,val) # TODO: this logic is probably wrong, since val is the full path. note os.path.join() takes teh last full apth, and ignores the previous ones if the last path is absolute
                else:
                    logging.info("Data {} with path {} is not used in the "
                                 "training.".format(key, val))

        ########### for scene graphs ################
        if self.use_sg:
            self.image_sg_path = self.cfg['image_sg_path']
            if args.data_dir is not None:
                for key, val in self.image_sg_path.items():
                    # get the absolute path
                    if key in self.datasets_names:
                        self.image_sg_path[key] = os.path.join(args.data_dir,
                                                                    val)
                    else:
                        logging.info("SG Data {} with path {} is not used in the "
                                    "training.".format(key, val))
        #############################################

        self.encoding = encoding
        self.current_doc = 0  # to avoid random sentence from same doc
        self.current_img = '' # to avoid random sentence from same image

        self.args = args

        # for loading samples directly from file
        self.sample_counter = 0  # used to keep track of full epochs on file
        self.line_buffer = None  # keep second sentence of a pair in memory and use as first sentence in next pair

        # for loading samples in memory
        self.current_random_doc = 0
        self.num_docs = 0
        self.sample_to_doc = []  # map sample index to doc and line

        self.chunk_list = None
        if 0 <= args.chunk_start_id <= args.chunk_end_id and args.chunk_end_id >= 0:
            self.chunk_list = [str(c_i) for c_i in range(args.chunk_start_id,
                                                    args.chunk_end_id)]
            logging.info('Chunk list: {}'.format(','.join(self.chunk_list)))

        # load image tags and features
        t_start = time.time()
        self.img_label_file = None
        self.img_qa_file = None
        self.img_label_offset_map = None
        self.img_qa_offset_map = None
        self.img_feature_file = None
        self.img_feat_offset_map = None
        self.load_img_labels()
        self.load_img_tsv_features()
        t_end = time.time()
        logging.info('Info: loading img features using {} secs'
                     .format(t_end - t_start))

        # load samples into memory
        if on_memory:
            self.all_docs = []
            self.all_qa_docs = []
            self.imgid2labels = {}
            self.corpus_lines = 0
            max_tokens = 0
            max_datpnts = args.max_datapoints if args.max_datapoints > 0 else len(self.corpus_tsvfile)
            for line_no in tqdm(range(max_datpnts)):
                doc = []
                row = self.corpus_tsvfile.seek(line_no)
                img_info = row[0].split('_')
                try:
                    label_info = row[1].split('_')
                except:
                    import pdb; pdb.set_trace()
                assert img_info[0] == label_info[0], "Dataset names for image and label do not match!" # assert triggers if img_info[0] != label_info[0]
                dataset_name = label_info[0]
                if dataset_name == 'cc':
                    dataset_name = 'googlecc'

                if dataset_name not in self.datasets_names: # this line allows us to only use for eg COCO if we want to just use that for pretraining. for this, in the yaml file just change the dataset tp "coco", in place of "coco_cc_...etc"
                    continue

                if dataset_name in self.datasets_with_splits:
                    chunk_id = img_info[-2]
                    if self.chunk_list is not None and chunk_id not in self.chunk_list:
                        continue
                    else:
                        img_feat_offset_map = self.img_feat_offset_map[dataset_name][chunk_id]
                else:
                    img_feat_offset_map = self.img_feat_offset_map[dataset_name]
                assert img_info[-1] in img_feat_offset_map, "{}: Image id {} cannot be found in image feature imageid_to_index file!".format(row[0], img_info[-1])

                # append id info
                doc.append('%s|%s' % (row[0], row[1]))
                # append text_a info
                self.corpus_lines = self.corpus_lines + 1
                sample = {"doc_id": len(self.all_docs), "line": len(doc)}
                self.sample_to_doc.append(sample)
                assert len(row[2]) != 0, "Text_a is empty in {} : {}".format(dataset_name, row[0])
                doc.append(row[2])
                # append text_b info
                self.corpus_lines = self.corpus_lines + 1
                label_id = label_info[-1]
                if 'qa' in label_info:
                    assert img_info[-1] == label_info[-2], "Image ids for image and qa do not match!"
                    label_line_no = self.img_qa_offset_map[dataset_name][label_id]
                    rowb = self.img_qa_file[dataset_name].seek(label_line_no)
                else:
                    assert img_info[-1] == label_info[-1], "Image ids for image and label do not match!"
                    label_line_no = self.img_label_offset_map[dataset_name][label_id]
                    rowb = self.img_label_file[dataset_name].seek(label_line_no)
                assert label_id == rowb[0]
                results = json.loads(rowb[1])
                if 'qa' not in label_info: # more intuitively, should be if 'qa' not in label_info:
                    objects = results['objects']
                    if row[0] not in self.imgid2labels:
                        self.imgid2labels[row[0]] = {
                            "image_h": results["image_h"], "image_w": results["image_w"],
                            "boxes": None
                        }
                    else:
                        assert results["image_h"] == self.imgid2labels[row[0]]["image_h"], "Image_h does not match in image {}!".format(row[0])
                        assert results["image_w"] == self.imgid2labels[row[0]]["image_w"], "Image_w does not match in image {}!".format(row[0])
                    if args.use_gtlabels and 'gt_objects' in results:
                        # use ground-truth tags for text_b
                        textb = '[TAGSEP]'.join([cur_d['class'] for cur_d in results["gt_objects"]])
                    else:
                        textb = '[TAGSEP]'.join([cur_d['class'] for cur_d in objects])
                else:
                    tag_label_line_no = self.img_label_offset_map[dataset_name][img_info[-1]]
                    tag_rowb = self.img_label_file[dataset_name].seek(tag_label_line_no)
                    tag_results = json.loads(tag_rowb[1])
                    if row[0] not in self.imgid2labels:
                        self.imgid2labels[row[0]] = {
                            "image_h": tag_results["image_h"], "image_w": tag_results["image_w"],
                            "boxes": None
                        }
                    else:
                        assert tag_results["image_h"] == self.imgid2labels[row[0]][
                            "image_h"], "Image_h does not match in image {}!".format(row[0])
                        assert tag_results["image_w"] == self.imgid2labels[row[0]][
                            "image_w"], "Image_w does not match in image {}!".format(row[0])
                    textb = ' '.join(results['labels'])
                assert len(textb) != 0, "Text_b is empty in {} : {}".format(dataset_name, row[1])
                doc.append(textb)

                # add to all_docs
                max_tokens = max(max_tokens, len(doc[1].split(' '))
                                 + len(doc[2].split(' ')))
                if 'qa' in label_info:
                    self.all_qa_docs.append({"doc":doc, "doc_id": len(self.all_docs)})
                self.all_docs.append(doc)

            self.num_docs = len(self.all_docs)
            logging.info("Max_tokens: {}".format(max_tokens))
        # load samples later lazily from disk
        else:
            raise ValueError("on_memory = False Not supported yet!")

        logging.info(
            "Total docs - Corpus_lines: {}-{}".format(self.num_docs,
                                                      self.corpus_lines))
        logging.info(
            "Total QA docs - Corpus_lines: {}".format(len(self.all_qa_docs))
        )

    def __len__(self):
        # last line of doc won't be used, because there's no "nextSentence".
        return self.corpus_lines - self.num_docs

    def get_img_info(self, idx):
        sample = self.sample_to_doc[idx]
        # img_id = self.all_docs[sample["doc_id"]][0].strip() # original
        img_id = self.all_docs[sample["doc_id"]][0].strip().split('|')[0]
        imgid2labels = self.imgid2labels[img_id]
        return {"height": imgid2labels["image_h"], "width": imgid2labels["image_w"]}

    def __getitem__(self, item):
        # import pdb; pdb.set_trace()
        cur_id = self.sample_counter
        self.sample_counter += 1
        if not self.on_memory:
            # after one epoch we start again from beginning of file
            if cur_id != 0 and (cur_id % len(self) == 0):
                raise ValueError("on_memory = False Not supported yet!")

        img_id, t1, t2, is_next_label, is_img_match = self.random_sent(item)

        # tokenize
        tokens_a = self.tokenizer.tokenize(t1)
        tokens_b = []
        tokens_b_spans = []
        if self.args.use_b:
            span_start = 0
            # import pdb; pdb.set_trace()
            if "[TAGSEP]" not in t2:
                tokens_b_spans = [] # no spans if there are no tokens in t2 (if there are tokens then there should have been a [TAGSEP])
                tokens_b = self.tokenizer.tokenize(t2)
                is_qa = True
            else:
                is_qa = False
                t2 = t2.split('[TAGSEP]')
                t2_space_sep  = ' '.join(t2)# same as original t2, without the [TAGSEP] tokens
                for tag in t2:
                    tag_tokenized = self.tokenizer.tokenize(tag)
                    tokens_b.extend(tag_tokenized)
                    tokens_b_spans.append([span_start, span_start+len(tag_tokenized)])
                    span_start = span_start+len(tag_tokenized)
                tokens_t2_space_sep = self.tokenizer.tokenize(t2_space_sep)
                assert tokens_t2_space_sep == tokens_b # just a check that with and without the [TAGSEP], the tokenized versions are the same
                
            if self.args.debug:
                import pdb; pdb.set_trace()
        else:
            tokens_b = None

        # combine to one sample
        cur_example = InputExample(guid=cur_id, tokens_a=tokens_a,
                                   tokens_b=tokens_b, tokens_b_spans=tokens_b_spans, is_next=is_next_label,
                                   img_id=img_id, is_img_match=is_img_match, is_qa=is_qa)

        # get image feature
        img_feat = self.get_img_feature(img_id)
        if img_feat.shape[0] >= self.args.max_img_seq_length:
            img_feat = img_feat[0:self.args.max_img_seq_length, ] # truncating the img features to be 50 length max!
            img_feat_len = img_feat.shape[0]
        else:
            img_feat_len = img_feat.shape[0]
            padding_matrix = torch.zeros((self.args.max_img_seq_length - img_feat.shape[0], img_feat.shape[1]))
            img_feat = torch.cat((img_feat, padding_matrix), 0)

        # transform sample to features
        cur_features = convert_example_to_features(self.args, cur_example,
                                                   self.seq_len, self.tokenizer,
                                                   img_feat_len)
        num_obj_tags = len(cur_features.tokens_b_spans) # should turn out to be 0 if the document is a QA document    

        ##############SG################
        if self.use_sg:
            # import pdb; pdb.set_trace()
            img_sg = self.get_img_sg(img_id, img_feat.device)
            rel_idx_pairs_all, rel_labels_all = img_sg['rel_idx_pairs'], img_sg['rel_labels']

            ### for relation prediction b/w object features
            # remove the relations we cant deal with (since we have removed some object, because we need to maintain max_img_seq_length)
            keep_rel_labels = ~(torch.ge(rel_idx_pairs_all[:, 0],self.args.max_img_seq_length) + torch.ge(rel_idx_pairs_all[:, 1],self.args.max_img_seq_length))# [true, trues, false, ....] places of false means the rel_idx_pairs_all has value > max_img_seq_length
            rel_labels = rel_labels_all[keep_rel_labels==True]
            rel_idx_pairs = rel_idx_pairs_all[keep_rel_labels==True]

            rel_len = rel_idx_pairs.shape[0]
            assert rel_len == rel_labels.shape[0]

            if rel_len <= self.args.max_rel_length:
                padding_rel_idx = torch.zeros((self.args.max_rel_length - rel_len, rel_idx_pairs.shape[1]))
                padding_rel_labels = -1*torch.ones((self.args.max_rel_length - rel_len)) # making tensors having -1's, so that we can do ignore_index=-1, while calculating the loss

                rel_idx_pairs = torch.cat((rel_idx_pairs, padding_rel_idx), 0)
                rel_labels = torch.cat((rel_labels, padding_rel_labels), 0)

                mask_rel_idx_pairs = torch.cat((torch.ones((rel_len, rel_idx_pairs.shape[1])), torch.zeros((self.args.max_rel_length - rel_len, rel_idx_pairs.shape[1]))), 0)
                mask_rel_labels = torch.cat((torch.ones((rel_len)), torch.zeros((self.args.max_rel_length - rel_len))), 0)   
            else:
                raise NotImplementedError()


            ### for relation prediction b/w object tags and object tags * object features
            max_num_objs = min(num_obj_tags, self.args.max_img_seq_length)
            keep_obj_tags_rel_labels = ~(torch.ge(rel_idx_pairs_all[:, 0],max_num_objs) + torch.ge(rel_idx_pairs_all[:, 1],max_num_objs))# [true, trues, false, ....] places of false means the rel_idx_pairs_all has value > max_img_seq_length
            obj_tags_rel_labels = rel_labels_all[keep_obj_tags_rel_labels==True]
            obj_tags_rel_idx_pairs = rel_idx_pairs_all[keep_obj_tags_rel_labels==True]

            # note: if all keep_obj_tags_rel_labels = False, as will be the case when we would have a q-a input, the cross entropy loss later would for label classification 
            # would come out to be nan, since all labels will be == ignore_index = -1, hence we should keep track of this case and multpy the final cross entropy loss with 0 wherever it is nan

            obj_tags_rel_len = obj_tags_rel_idx_pairs.shape[0]
            assert obj_tags_rel_len == obj_tags_rel_labels.shape[0]

            if obj_tags_rel_len==0:
                cur_features.is_qa = True
                # obj_tags_rel_len can be 0 in 2 cases 
                # 1] the datapoint is a q-a datapoint so there are no object tags as input hence no relations
                # 2] object tags are as input but there are no relations between the first k object tags
                # is_qa hence becomes a tag for telling the model when to use the loss and when to not, and no longer signifies the data beign qa or not

            if obj_tags_rel_len <= self.args.max_rel_length:
                padding_rel_idx = torch.zeros((self.args.max_rel_length - obj_tags_rel_len, obj_tags_rel_idx_pairs.shape[1]))
                padding_rel_labels = -1*torch.ones((self.args.max_rel_length - obj_tags_rel_len)) # making tensors having -1's, so that we can do ignore_index=-1, while calculating the loss

                obj_tags_rel_idx_pairs = torch.cat((obj_tags_rel_idx_pairs, padding_rel_idx), 0)
                obj_tags_rel_labels = torch.cat((obj_tags_rel_labels, padding_rel_labels), 0)
 
            else:
                raise NotImplementedError()
            # import pdb; pdb.set_trace()

        else:
            rel_idx_pairs, mask_rel_idx_pairs, rel_labels, mask_rel_labels, rel_len, obj_tags_rel_idx_pairs, obj_tags_rel_labels = None, None, None, None, None, None, None
        ################################

    

        # number of image features should be the same as the number of object labels
        if self.use_sg:
            assert img_feat_len == img_sg['obj_labels'].shape[0] or (img_sg['obj_labels'].shape[0]>self.args.max_img_seq_length and img_feat_len==self.args.max_img_seq_length)



        return (img_feat, img_feat_len), (
            torch.tensor(cur_features.input_ids, dtype=torch.long),
            torch.tensor(cur_features.input_mask, dtype=torch.long),
            torch.tensor(cur_features.segment_ids, dtype=torch.long),
            torch.tensor(cur_features.lm_label_ids, dtype=torch.long),
            torch.tensor(cur_features.is_next),
            torch.tensor(cur_features.is_img_match),
            cur_features.tokens_b_start_pos,
            cur_features.tokens_b_spans,
            torch.tensor(cur_features.tokens_b_firsttokens),
            torch.tensor(cur_features.tokens_b_lasttokens),
            torch.tensor(cur_features.is_qa)
        ), (rel_idx_pairs, mask_rel_idx_pairs, rel_labels, mask_rel_labels, rel_len, obj_tags_rel_idx_pairs, obj_tags_rel_labels), item


    def random_sent(self, index):
        """
        Get one sample from corpus consisting of two sentences. With prob. 50% these are two subsequent sentences
        from one doc. With 50% the second sentence will be a random one from another doc.
        :param index: int, index of sample.
        :return: (str, str, int), sentence 1, sentence 2, isNextSentence Label
        """
        img_id, t1, t2 = self.get_corpus_line(index)
        rand_dice = random.random()
        if rand_dice > 0.5:
            label = 0
            random_img_id = img_id
        elif rand_dice > self.args.texta_false_prob and t2 != "":
            # wrong qa triplets
            random_img_id, t2 = self.get_random_line()
            label = 1
        else:
            # wrong retrieval triplets
            random_img_id, t1 = self.get_random_texta()
            # args.num_contrast_classes = 3 if args.texta_false_prob<0.5 and (args.texta_false_prob>0 or not args.use_b) else 2
            label = self.args.num_contrast_classes-1

        img_match_label = 0
        if img_id != random_img_id: img_match_label = 1

        assert len(t1) > 0
        assert len(t2) > 0 or not self.args.use_b
        return img_id, t1, t2, label, img_match_label

    def get_img_sg(self, image_id, device):
        """
        Get the scene graph corresponding to the image with image id = img_id
        :param img_id: image index
        :return: scene graph of the image, obtained using the object detector and a relation prediction algorithm
        """
        # import pdb; pdb.set_trace()
        img_infos = image_id.split('_')
        dataset_name = img_infos[0]
        if dataset_name=='coco':
            sg_path = os.path.join(self.image_sg_path[dataset_name], str(image_id).split('_')[-1] + '.pt')
        sg_dict = torch.load(sg_path, map_location=device)
        return sg_dict


    def get_corpus_line(self, item):
        """
        Get one sample from corpus consisting of a pair of two subsequent lines from the same doc.
        :param item: int, index of sample.
        :return: (str, str), two subsequent sentences from corpus
        """
        assert item < self.corpus_lines
        if self.on_memory:
            sample = self.sample_to_doc[item]
            # img_id = self.all_docs[sample["doc_id"]][0].strip() # original
            img_id = self.all_docs[sample["doc_id"]][0].strip().split('|')[0]
            t1 = self.all_docs[sample["doc_id"]][sample["line"]]
            t2 = self.all_docs[sample["doc_id"]][sample["line"] + 1]
            # used later to avoid random nextSentence from same doc
            self.current_doc = sample["doc_id"]
            self.current_img = img_id

            assert t1 != ""
            if self.args.use_b or 'qa' in self.all_docs[sample["doc_id"]][0].split('_'):
                assert t2 != ""
            else:
                t2 = ""
            return img_id, t1, t2
        else:
            raise ValueError("on_memory = False Not supported yet!")

    def get_random_line(self):
        """
        Get random line from another document for nextSentence task.
        :return: str, content of one line
        """
        # Similar to original tf repo: This outer loop should rarely go for more than one iteration for large
        # corpora. However, just to be careful, we try to make sure that
        # the random document is not the same as the document we're processing.
        if self.on_memory:
            if self.textb_sample_mode in [0, 1]:
                # sample from all docs
                for _ in range(10):
                    rand_doc_idx = random.randrange(0, len(self.all_docs))
                    img_id = self.all_docs[rand_doc_idx][0].split('|')[0]
                    # check if our picked random line is really from another image like we want it to be
                    if img_id != self.current_img:
                        break
                rand_doc = self.all_docs[rand_doc_idx]
            else:
                # sample from all qa docs
                for _ in range(10):
                    rand_doc_idx = random.randrange(0, len(self.all_qa_docs))
                    # check if our picked random line is really from another doc like we want it to be % no need to be different image here
                    if self.all_qa_docs[rand_doc_idx]["doc_id"] != self.current_doc:
                        break
                rand_doc = self.all_qa_docs[rand_doc_idx]["doc"]
            # img_id = rand_doc[0] # original
            img_id = rand_doc[0].split('|')[0]
            if self.textb_sample_mode == 0:
                # default oscar sample mode
                line = rand_doc[random.randrange(1, len(rand_doc))]
            else:
                # only sample text_b
                line = rand_doc[2]
            return img_id, line
        else:
            raise ValueError("on_memory = False Not supported yet!")

    def get_random_texta(self):
        """
        Get random text_a from another document for nextSentence task.
        :return: str, content of one line
        """
        # Similar to original tf repo: This outer loop should rarely go for more than one iteration for large
        # corpora. However, just to be careful, we try to make sure that
        # the random document is not the same as the document we're processing.
        if self.on_memory:
            for _ in range(10):
                rand_doc_idx = random.randrange(0, len(self.all_docs))
                img_id = self.all_docs[rand_doc_idx][0].split('|')[0]
                # check if our picked random line is really from another image like we want it to be
                if img_id != self.current_img:
                    break
            rand_doc = self.all_docs[rand_doc_idx]
            # img_id = rand_doc[0] # original
            img_id = rand_doc[0].split('|')[0]
            line = rand_doc[1] # we want the text_a
            return img_id, line
        else:
            raise ValueError("on_memory = False Not supported yet!")

    # tsv image labels
    def load_img_labels(self):
        self.check_img_label_file()
        self.check_img_label_offset_map()

    def check_img_label_file(self):
        if self.img_label_file is None:
            self.img_label_file = {}
            self.img_qa_file = {}
            for dataset_name in self.datasets_names:
                img_label_file_path = os.path.join(
                    self.image_label_path[dataset_name], 'predictions_gt.tsv')
                img_qa_file_path = os.path.join(
                    self.image_label_path[dataset_name], 'QA_fileB.tsv') # append name of dataset with QA_B, but will be only used when such a dataset actually exisits
                t_s = time.time()
                self.img_label_file[dataset_name] = TSVFile(img_label_file_path)
                if os.path.exists(img_qa_file_path):
                    self.img_qa_file[dataset_name] = TSVFile(img_qa_file_path)
                t_e = time.time()
                logging.info(
                    "Open image label file {}, time: {}".format(
                        img_label_file_path, (t_e - t_s)))

    def check_img_label_offset_map(self):
        if self.img_label_offset_map is None:
            self.img_label_offset_map = {}
            self.img_qa_offset_map = {}
            for dataset_name in self.datasets_names:
                img_label_offset_map_path = os.path.join(
                    self.image_label_path[dataset_name], 'imageid2idx.json')
                img_qa_offset_map_path = os.path.join(
                    self.image_label_path[dataset_name], 'QA_qaid2idx.json')
                t_s = time.time()
                self.img_label_offset_map[dataset_name] = json.load(
                    open(img_label_offset_map_path))
                if os.path.exists(img_qa_offset_map_path):
                    self.img_qa_offset_map[dataset_name] = json.load(
                        open(img_qa_offset_map_path))
                t_e = time.time()
                logging.info(
                    "Load img label offset map: {}, time: {}".format(
                        img_label_offset_map_path, (t_e - t_s)))

    def get_img_labels(self, image_id):
        """ decode the image labels: read the image label from the img_label.tsv """
        self.check_img_label_file()
        self.check_img_label_offset_map()

        if image_id in self.img_label_offset_map:
            img_offset = self.img_label_offset_map[image_id]

            self.img_label_file.seek(img_offset, 0)
            arr = [s.strip() for s in
                   self.img_label_file.readline().split('\t')]
            eles = json.loads(arr[1])
            labels = eles['labels']
            return labels

        return None

    # tsv feature loading
    def load_img_tsv_features(self):
        self.check_img_feature_file()
        self.check_img_feature_offset_map()

    def check_img_feature_file(self):
        if self.img_feature_file is None:
            # self.img_feature_file = [] # original
            self.img_feature_file = {}
            self.img_feat_offset_map = {}
            for dataset_name in self.datasets_names:
                logging.info("* Loading dataset {}".format(dataset_name))
                if dataset_name in self.datasets_with_splits:
                    self.img_feature_file[dataset_name] = {}
                    self.img_feat_offset_map[dataset_name] = {}
                    chunk_list = []
                    if self.chunk_list is not None:
                        chunk_list = self.chunk_list
                        chunk_file_list = []
                        for chunk_fp_id in chunk_list:
                            chunk_file_list.append(
                                os.path.join(self.image_feature_path[dataset_name], chunk_fp_id, self.image_file_name)
                            )
                        if dataset_name == 'googlecc':
                            for i, (chunk_fp_id, chunk_fp) in enumerate(zip(chunk_list, chunk_file_list)):
                                assert os.path.exists(chunk_file_list[i]), "Chunk file {} does not exists!".format(chunk_fp)
                    else:
                        chunk_file_list = glob.glob(
                            self.image_feature_path[dataset_name] + "/*/{}".format(self.image_file_name)
                        )
                        for chunk_fp in chunk_file_list:
                            chunk_fp_id = chunk_fp.split('/')[-2]
                            chunk_list.append(chunk_fp_id)
                    logging.info(
                        "* Load Image Chunks {}".format(len(chunk_list)))

                    t_s_total = time.time()
                    for chunk_fp in chunk_file_list:
                        chunk_fp_id = chunk_fp.split('/')[-2]
                        t_s = time.time()
                        self.img_feature_file[dataset_name][chunk_fp_id] = TSVFile(chunk_fp)
                        chunk_offsetmap = os.path.join(os.path.dirname(chunk_fp), 'imageid2idx.json')
                        assert os.path.isfile(chunk_offsetmap), "Imageid2idx file {} does not exists!".format(chunk_offsetmap)
                        self.img_feat_offset_map[dataset_name][
                            chunk_fp_id] = json.load(open(chunk_offsetmap, 'r'))
                        t_e = time.time()
                        logging.info(
                            "Open image chunk {}, time: {}".format(
                                chunk_fp_id, (t_e - t_s)))
                    t_e_total = time.time()
                    logging.info(
                        "Open total {} image chunks, time: {}".format(
                            len(chunk_list), (t_e_total - t_s_total)))
                    logging.info(
                        "Image chunk info: {}".format('\n'.join(chunk_file_list))
                    )
                elif dataset_name in self.datasets_with_onesplit:
                    t_s = time.time()
                    chunk_fp = os.path.join(self.image_feature_path[dataset_name], self.image_file_name)
                    self.img_feature_file[dataset_name] = TSVFile(chunk_fp)
                    chunk_offsetmap = os.path.join(os.path.dirname(chunk_fp), 'imageid2idx.json')
                    assert os.path.isfile(chunk_offsetmap), "Imageid2idx file {} does not exists!".format(chunk_offsetmap)
                    self.img_feat_offset_map[dataset_name] = json.load(open(chunk_offsetmap, 'r'))
                    t_e = time.time()
                    logging.info(
                        "Open dataset {}, time: {}".format(
                            chunk_fp, (t_e - t_s)))
                else:
                    raise ValueError("Not supported dataset: {}".format(dataset_name))

    def check_img_feature_offset_map(self):
        """ load the image feature offset map """
        if self.img_feat_offset_map is None:
            self.img_feat_offset_map = {}
            for dataset_name in self.datasets_names:
                logging.info("* Loading imageid2idx_map {}".format(dataset_name))
                if dataset_name in self.datasets_with_splits:
                    chunk_list = []
                    chunk_file_list = glob.glob(
                        self.image_feature_path[
                            dataset_name] + "/*/imageid2idx.json"
                    )
                    for chunk_fp in chunk_file_list:
                        chunk_fp_id = chunk_fp.split('/')[-2]
                        chunk_list.append(chunk_fp_id)
                    logging.info(
                        "* Load Image Chunks {}".format(len(chunk_list)))

                    t_s_total = time.time()
                    for chunk_fp in chunk_file_list:
                        chunk_fp_id = chunk_fp.split('/')[-2]
                        t_s = time.time()
                        self.img_feat_offset_map[dataset_name][
                            chunk_fp_id] = json.load(open(chunk_fp))
                        t_e = time.time()
                        logging.info(
                            "Open image chunk {}, time: {}".format(
                                chunk_fp_id, (t_e - t_s)))
                    t_e_total = time.time()
                    logging.info(
                        "Open total {} image chunks, time: {}".format(
                            len(chunk_list), (t_e_total - t_s_total)))
                elif dataset_name in self.datasets_with_onesplit:
                    t_s = time.time()
                    chunk_fp = self.image_feature_path[
                                   dataset_name] + "/imageid2idx.json"
                    self.img_feat_offset_map[dataset_name] = json.load(
                        open(chunk_fp))
                    t_e = time.time()
                    logging.info(
                        "Open dataset {}, time: {}".format(
                            chunk_fp, (t_e - t_s)))
                else:
                    raise ValueError(
                        "Not supported dataset: {}".format(dataset_name))

    def get_img_feature(self, image_id):
        """ decode the image feature: read the image feature from the right chunk id """
        self.check_img_feature_file()
        self.check_img_feature_offset_map()
        img_infos = image_id.split('_')
        dataset_name = img_infos[0]
        if dataset_name == 'cc':
            dataset_name = 'googlecc'
        img_id = img_infos[-1]
        if dataset_name in self.datasets_with_splits:
            chunk_id = img_infos[-2]
            img_feat_offset_map = self.img_feat_offset_map[dataset_name][chunk_id]
            img_feature_file = self.img_feature_file[dataset_name][chunk_id]
        else:
            img_feat_offset_map = self.img_feat_offset_map[dataset_name]
            img_feature_file = self.img_feature_file[dataset_name]
        if img_id in img_feat_offset_map:
            img_offset = img_feat_offset_map[img_id]

            arr = img_feature_file.seek(img_offset)
            num_boxes = int(arr[1])
            feat = np.frombuffer(base64.b64decode(arr[-1]),
                                 dtype=np.float32).reshape(
                (num_boxes, self.args.img_feature_dim))
            # feat = torch.from_numpy(feat)
            feat = torch.from_numpy(np.array(feat))
            return feat

        return None


class InputExample(object):
    """A single training/test example for the language model."""

    def __init__(self, guid, tokens_a, tokens_b=None, tokens_b_spans=None, is_next=None,
                 lm_labels=None, img_id=None, is_img_match=None,
                 img_label=None, is_qa=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            tokens_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            tokens_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
        """
        self.guid = guid
        self.tokens_a = tokens_a
        self.tokens_b = tokens_b

        self.tokens_b_spans = tokens_b_spans

        self.is_next = is_next  # nextSentence
        self.lm_labels = lm_labels  # masked words for language model

        self.img_id = img_id
        self.is_img_match = is_img_match
        self.img_label = img_label

        self.is_qa = is_qa


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, is_next,
                 lm_label_ids, img_feat_len, is_img_match, 
                 tokens_b_start_pos, tokens_b_spans, tokens_b_firsttokens, tokens_b_lasttokens, is_qa):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.is_next = is_next
        self.lm_label_ids = lm_label_ids

        self.img_feat_len = img_feat_len
        self.is_img_match = is_img_match

        self.tokens_b_start_pos = tokens_b_start_pos
        self.tokens_b_spans = tokens_b_spans
        self.tokens_b_firsttokens = tokens_b_firsttokens
        self.tokens_b_lasttokens = tokens_b_lasttokens
        self.is_qa = is_qa


def random_word(tokens, tokenizer):
    """
    Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
    :param tokens: list of str, tokenized sentence.
    :param tokenizer: Tokenizer, object used for tokenization (we need it's vocab here)
    :return: (list of str, list of int), masked tokens and related labels for LM prediction
    """
    output_label = []

    for i, token in enumerate(tokens):
        prob = random.random()
        # mask token with 15% probability
        if prob < 0.15:
            prob /= 0.15

            # 80% randomly change token to mask token
            if prob < 0.8:
                tokens[i] = "[MASK]"

            # 10% randomly change token to random token
            elif prob < 0.9:
                tokens[i] = random.choice(list(tokenizer.vocab.items()))[0]

            # -> rest 10% randomly keep current token

            # append current token to output (we will predict these later)
            try:
                output_label.append(tokenizer.vocab[token])
            except KeyError:
                # For unknown words (should not occur with BPE vocab)
                output_label.append(tokenizer.vocab["[UNK]"])
                logging.warning(
                    "Cannot find token '{}' in vocab. Using [UNK] insetad".format(
                        token))
        else:
            # no masking token (will be ignored by loss function later)
            output_label.append(-1)

    return tokens, output_label


def convert_example_to_features(args, example, max_seq_length, tokenizer,
                                img_feat_len):
    """
    Convert a raw sample (pair of sentences as tokenized strings) into a proper training sample with
    IDs, LM labels, input_mask, CLS and SEP tokens etc.
    :param args: parameter settings
    :param img_feat_len: lens of actual img features
    :param example: InputExample, containing sentence input as strings and is_next label
    :param max_seq_length: int, maximum length of sequence.
    :param tokenizer: Tokenizer
    :return: InputFeatures, containing all inputs and labels of one sample as IDs (as used for model training)
    """

    tokens_a = example.tokens_a
    tokens_b = None
    if example.tokens_b:
        tokens_b = example.tokens_b
        tokens_b_spans = example.tokens_b_spans
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"

        _truncate_seq_pair(tokens_a, tokens_b, tokens_b_spans, max_seq_length - 3)
    else:
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]

    is_next_type = example.is_next * example.is_img_match # is_img_match = 1 for mismatch images
    if args.num_contrast_classes == 2 and args.texta_false_prob == 0.5 and is_next_type == 1:
        is_next_type = 2 # is_next_type 0: correct pair, 1: wrong text_b, 2: wrong text_a
    # if not args.mask_loss_for_unmatched and is_next_type == 2:
    #     t1_label = [-1]*len(tokens_a)
    # else:
    tokens_a, t1_label = random_word(tokens_a, tokenizer)
    if tokens_b:
        if not args.mask_loss_for_unmatched and is_next_type == 1:
            t2_label = [-1]*len(tokens_b)
        else:
            tokens_b, t2_label = random_word(tokens_b, tokenizer)

    # concatenate lm labels and account for CLS, SEP, SEP
    if tokens_b:
        lm_label_ids = ([-1] + t1_label + [-1] + t2_label + [-1])
    else:
        lm_label_ids = ([-1] + t1_label + [-1])

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0   0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambigiously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    tokens_b_start_pos = len(tokens)
    if tokens_b:
        assert len(tokens_b) > 0
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

    tokens_b_firsttokens = [tokens_b_start_pos + span[0] for span in tokens_b_spans]
    tokens_b_lasttokens = [tokens_b_start_pos + span[1]-1 for span in tokens_b_spans] # -1 since a spans [6,8]'s last token is at 7
    ## pad tokens_b_firsttokens and tokens_b_lasttokens with 0's

    assert len(tokens_b_lasttokens) == len(tokens_b_firsttokens)
    if len(tokens_b_firsttokens) < args.max_img_seq_length:
        tokens_b_firsttokens = tokens_b_firsttokens + [0]*(args.max_img_seq_length-len(tokens_b_firsttokens))
        tokens_b_lasttokens = tokens_b_lasttokens + [0]*(args.max_img_seq_length-len(tokens_b_lasttokens))
    else:
        tokens_b_firsttokens = tokens_b_firsttokens[:args.max_img_seq_length]
        tokens_b_lasttokens = tokens_b_lasttokens[:args.max_img_seq_length]

    if args.debug:
        import pdb; pdb.set_trace()
    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        lm_label_ids.append(-1)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(lm_label_ids) == max_seq_length

    # image features
    if args.max_img_seq_length > 0:
        if img_feat_len > args.max_img_seq_length:
            input_mask = input_mask + [1] * img_feat_len
        else:
            input_mask = input_mask + [1] * img_feat_len
            pad_img_feat_len = args.max_img_seq_length - img_feat_len
            input_mask = input_mask + ([0] * pad_img_feat_len)

    lm_label_ids = lm_label_ids + [-1] * args.max_img_seq_length

    # if torch.all(torch.tensor(lm_label_ids)==-1):
    #     import pdb; pdb.set_trace()

    if example.guid < 1:
        logging.info("*** Example ***")
        logging.info("guid: %s" % example.guid)
        logging.info("tokens: %s" % " ".join([str(x) for x in tokens]))
        logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        logging.info("LM label: %s " % lm_label_ids)
        logging.info("Is next sentence label: %s " % example.is_next)
        logging.info("Is QA?: %s " % example.is_qa)
        

    features = InputFeatures(input_ids=input_ids,
                             input_mask=input_mask,
                             segment_ids=segment_ids,
                             lm_label_ids=lm_label_ids,
                             is_next=example.is_next,
                             img_feat_len=img_feat_len,
                             is_img_match=example.is_img_match,
                             tokens_b_start_pos=tokens_b_start_pos,
                             tokens_b_spans=tokens_b_spans,
                             tokens_b_firsttokens=tokens_b_firsttokens,
                             tokens_b_lasttokens=tokens_b_lasttokens,
                             is_qa=example.is_qa)
    return features


def _truncate_seq_pair(tokens_a, tokens_b, tokens_b_spans, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()
            if len(tokens_b_spans)>0:
                tokens_b_spans[-1][-1]-=1
                if tokens_b_spans[-1][0]==tokens_b_spans[-1][1]:
                    tokens_b_spans.pop()