# coding:utf-8
import os
from torch.utils.data import Dataset
from datasets import load_dataset
from dataclasses import dataclass
from transformers.file_utils import PaddingStrategy
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from typing import Optional, Union
from common.utils import shift_tokens_right

def padding_func(features, padding_side="right", pad_token_id=1, key_list="label", pad_to_multiple_of=1, max_length=None):
    for key in key_list:
        assert key in features[0].keys(), f"{key} not in {features[0].keys()}"
        max_label_length = max(len(feature[key]) for feature in features)
        if pad_to_multiple_of > 1:
            if max_length is not None:
                max_label_length = min(max_length,
                    (max_label_length + pad_to_multiple_of - 1) // pad_to_multiple_of * pad_to_multiple_of
                )
            else:
                max_label_length = (max_label_length + pad_to_multiple_of - 1) // pad_to_multiple_of * pad_to_multiple_of
                
        for feature in features:
            remainder = [pad_token_id] * (max_label_length - len(feature[key]))
            feature[key] = (
                feature[key] + remainder if padding_side == "right" else remainder + feature[key]
            )
    return


class AMRParsingDataSet(Dataset):
    def __init__(
        self, tokenizer, args, model_args
    ):
        super().__init__()
        self.train_file = args.train_file
        self.validation_file = args.validation_file
        self.test_file = args.test_file
        self.src_prefix = args.source_prefix
        self.tgt_prefix = args.target_prefix
        self.cache_dir = model_args.cache_dir
        self.use_speaker_prefix = args.use_speaker_prefix
        self.tokenizer = tokenizer
        self.unified_input = args.unified_input

        self.max_src_length = min(args.max_source_length, self.tokenizer.model_max_length)
        self.max_tgt_length = min(args.max_target_length, self.tokenizer.model_max_length)

        data_files = {}
        data_files["train"] = self.train_file
        data_files["validation"] = self.validation_file
        data_files["test"] = self.test_file

        # print("datafiles:", data_files)
        print("Dataset cache dir:", self.cache_dir)
        # exit()
        self.datasets = self.load_dataset(args.data_dir)
        column_names = self.datasets["train"].column_names
        print("datasets:", self.datasets)
        print("colums:", column_names)
    
    def load_dataset(self, data_path):
        from datasets import Dataset, DatasetDict
        import tqdm
        data_dict = DatasetDict()
        for ds in ['train', 'val', 'test']:
            data = {}
            src_list = []
            for seq in ['NLR', 'NRL_generated', 'NRL']:
                if seq == 'NRL' and ds != 'train':
                    continue
                for order in ['dfs']:
                    tgt_list = []   
                    file = open(data_path + '_' + order + '_' + seq + '/' + ds + '.jsonl', 'r', encoding='utf8')
                    
                    if seq == 'NRL_generated' and ds != 'train':
                        seq = 'NRL'
                    
                    if src_list == []:
                        for i, line in tqdm.tqdm(enumerate(file.readlines())):
                            line = eval(line)
                            src_list.append(line['sent'])
                            tgt_list.append(line['amr'])
                        data['src'] = src_list
                        data['tgt_'+order + '_'+ seq] = tgt_list
                    else:
                        for i, line in tqdm.tqdm(enumerate(file.readlines())):
                            line = eval(line)
                            assert(line['sent'] == data['src'][i])
                            tgt_list.append(line['amr'])
                        data['tgt_'+order + '_'+ seq] = tgt_list
            data_dict[ds if ds != 'val' else 'validation'] = Dataset.from_dict(data)
        return data_dict

    def tokenize_function_for_train(self, examples):
        amr_dfs_NLR = examples["tgt_dfs_NLR"]  # AMR tokens
        amr_dfs_NRL = examples["tgt_dfs_NRL"]  # AMR tokens
        amr_dfs_NRL_generated = examples["tgt_dfs_NRL_generated"]  # AMR tokens
        txt = examples["src"]  # Text tokens

        amr_ids_dfs_NLR = [self.tokenizer.tokenize_amr(itm.split())[:self.max_src_length - 2] + [self.tokenizer.amr_eos_token_id] for itm in amr_dfs_NLR]
        amr_ids_dfs_NRL = [self.tokenizer.tokenize_amr(itm.split())[:self.max_src_length - 5] + [self.tokenizer.amr_eos_token_id] for itm in amr_dfs_NRL]
        amr_ids_dfs_NRL_generated = [self.tokenizer.tokenize_amr(itm.split())[:self.max_src_length - 5] + [self.tokenizer.amr_eos_token_id] for itm in amr_dfs_NRL_generated]

        raw_txt_ids = self.tokenizer(
            txt, max_length=self.max_tgt_length, padding=False, truncation=True
        )["input_ids"]
        if self.unified_input:
            txt_ids = [itm[:self.max_tgt_length-3] + [self.tokenizer.amr_bos_token_id, self.tokenizer.mask_token_id, self.tokenizer.amr_eos_token_id] for itm in raw_txt_ids]
            
            # [<s>[mask]</s><AMR>xxx</AMR>]
            amr_ids_dfs_NRL = [[self.tokenizer.bos_token_id, self.tokenizer.mask_token_id, self.tokenizer.eos_token_id] + [self.tokenizer.amr_bos_token_id, self.tokenizer.amr_dfs_NRL_token_id] + itm for itm in amr_ids_dfs_NRL]
            amr_ids_dfs_NRL_generated = [[self.tokenizer.bos_token_id, self.tokenizer.mask_token_id, self.tokenizer.eos_token_id] + [self.tokenizer.amr_bos_token_id, self.tokenizer.amr_dfs_NRL_token_id] + itm for itm in amr_ids_dfs_NRL_generated]
        else:
            txt_ids = raw_txt_ids
        return {
            "input_ids": txt_ids,
            "input_ids_dfs_NRL": amr_ids_dfs_NRL,
            "input_ids_dfs_NRL_generated": amr_ids_dfs_NRL_generated,
            "labels": amr_ids_dfs_NLR
        }

    def tokenize_function(self, examples):
        amr_dfs_NLR = examples["tgt_dfs_NLR"]  # AMR tokens
        amr_dfs_NRL = examples["tgt_dfs_NRL"]  # AMR tokens
        txt = examples["src"]  # Text tokens

        amr_ids_dfs_NLR = [self.tokenizer.tokenize_amr(itm.split())[:self.max_src_length - 2] + [self.tokenizer.amr_eos_token_id] for itm in amr_dfs_NLR]
        amr_ids_dfs_NRL = [self.tokenizer.tokenize_amr(itm.split())[:self.max_src_length - 5] + [self.tokenizer.amr_eos_token_id] for itm in amr_dfs_NRL]

        raw_txt_ids = self.tokenizer(
            txt, max_length=self.max_tgt_length, padding=False, truncation=True
        )["input_ids"]
        if self.unified_input:
            txt_ids = [itm[:self.max_tgt_length-3] + [self.tokenizer.amr_bos_token_id, self.tokenizer.mask_token_id, self.tokenizer.amr_eos_token_id] for itm in raw_txt_ids]
            
            # [<s>[mask]</s><AMR>xxx</AMR>]
            amr_ids_dfs_NRL = [[self.tokenizer.bos_token_id, self.tokenizer.mask_token_id, self.tokenizer.eos_token_id] + [self.tokenizer.amr_bos_token_id, self.tokenizer.amr_dfs_NRL_token_id] + itm for itm in amr_ids_dfs_NRL]
        else:
            txt_ids = raw_txt_ids
        return {
            "input_ids": txt_ids,
            "input_ids_dfs_NRL": amr_ids_dfs_NRL,
            "labels": amr_ids_dfs_NLR
        }


@dataclass
class DataCollatorForAMRParsing:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        model (:class:`~transformers.PreTrainedModel`):
            The model that is being trained. If set and has the `prepare_decoder_input_ids_from_labels`, use it to
            prepare the `decoder_input_ids`

            This is useful when using `label_smoothing` to avoid calculating loss twice.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (:obj:`int`, `optional`, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
    """

    tokenizer: PreTrainedTokenizerBase
    model: Optional[PreTrainedModel] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100

    def __call__(self, features):
        
        padding_func(
            features,
            padding_side=self.tokenizer.padding_side,
            pad_token_id=self.label_pad_token_id,
            key_list = ["labels", "input_ids_dfs_NRL", "input_ids_dfs_NRL_generated"] if "input_ids_dfs_NRL_generated" in features[0].keys() else ["labels", "input_ids_dfs_NRL"],
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        
        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # prepare decoder_input_ids
        features["decoder_input_ids"] = shift_tokens_right(
            features["labels"],
            pad_token_id=self.tokenizer.pad_token_id,
            decoder_start_token_id=self.tokenizer.amr_bos_token_id,
        )

        if "input_ids_dfs_NRL_generated" in features.keys():
            return {
                "input_ids": features["input_ids"],
                "input_ids_dfs_NRL": features["input_ids_dfs_NRL"],
                "input_ids_dfs_NRL_generated": features["input_ids_dfs_NRL_generated"],
                "labels": features["labels"],
                "decoder_input_ids": features["decoder_input_ids"],
            }
        else:
            return {
                "input_ids": features["input_ids"],
                "input_ids_dfs_NRL": features["input_ids_dfs_NRL"],
                "labels": features["labels"],
                "decoder_input_ids": features["decoder_input_ids"],
            }
        

class AMR2TextDataSet(Dataset):
    def __init__(
        self, tokenizer, args, model_args
    ):
        super().__init__()
        self.train_file = args.train_file
        self.validation_file = args.validation_file
        self.test_file = args.test_file
        self.src_prefix = args.source_prefix
        self.tgt_prefix = args.target_prefix
        self.cache_dir = model_args.cache_dir
        self.use_speaker_prefix = args.use_speaker_prefix
        self.tokenizer = tokenizer
        self.unified_input = args.unified_input

        self.max_src_length = min(args.max_source_length, self.tokenizer.model_max_length)
        self.max_tgt_length = min(args.max_target_length, self.tokenizer.model_max_length)

        data_files = {}
        data_files["train"] = self.train_file
        data_files["validation"] = self.validation_file
        data_files["test"] = self.test_file

        # print("datafiles:", data_files)
        print("Dataset cache dir:", self.cache_dir)
        # exit()
        self.datasets = load_dataset(
            f"{os.path.dirname(__file__)}/data.py",
            data_files=data_files,
            keep_in_memory=False,
        )
        column_names = self.datasets["train"].column_names
        print("datasets:", self.datasets)
        print("colums:", column_names)

    def tokenize_function(self, examples):
        src = examples["src"]  # AMR tokens
        tgt = examples["tgt"]  # Text tokens
        if not self.unified_input:
            src_ids = [[self.tokenizer.amr_bos_token_id] + self.tokenizer.tokenize_amr(itm.split())[:self.max_src_length - 2] + [self.tokenizer.amr_eos_token_id] for itm in src]
        else:
            # [<s>[mask]</s><AMR>xxx</AMR>]
            src_ids = [[self.tokenizer.bos_token_id, self.tokenizer.mask_token_id, self.tokenizer.eos_token_id] + [self.tokenizer.amr_bos_token_id] + self.tokenizer.tokenize_amr(itm.split())[:self.max_src_length -5] + [self.tokenizer.amr_eos_token_id] for itm in src]
            
        with self.tokenizer.as_target_tokenizer():
            tgt_ids = self.tokenizer(
                tgt, max_length=self.max_tgt_length, padding=False, truncation=True
            )
            tgt_ids["input_ids"] = [
                label[1:] for label in tgt_ids["input_ids"]
            ]
        model_inputs = {}
        model_inputs["input_ids"] = src_ids
        model_inputs["labels"] = tgt_ids["input_ids"]
        return model_inputs


@dataclass
class DataCollatorForAMR2Text:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        model (:class:`~transformers.PreTrainedModel`):
            The model that is being trained. If set and has the `prepare_decoder_input_ids_from_labels`, use it to
            prepare the `decoder_input_ids`

            This is useful when using `label_smoothing` to avoid calculating loss twice.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (:obj:`int`, `optional`, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
    """

    tokenizer: PreTrainedTokenizerBase
    model: Optional[PreTrainedModel] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100

    def __call__(self, features):
        
        padding_func(
            features,
            padding_side=self.tokenizer.padding_side,
            pad_token_id=self.label_pad_token_id,
            key="labels",
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        
        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # prepare decoder_input_ids
        features["decoder_input_ids"] = shift_tokens_right(
            features["labels"],
            pad_token_id=self.tokenizer.pad_token_id,
            decoder_start_token_id=self.tokenizer.eos_token_id,
        )

        return {
            "input_ids": features["input_ids"],
            "labels": features["labels"],
            "decoder_input_ids": features["decoder_input_ids"],
        }
