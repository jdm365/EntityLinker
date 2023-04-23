import torch as T
from tqdm import tqdm
import numpy as np
import datasets
from torch.utils.data import Dataset, DataLoader
from transformers import ByT5Tokenizer
import gc
import re


class TorchDatasetWrapper(Dataset):
    def __init__(self, huggingface_dataset) -> None:
        super(Dataset, self).__init__()
        self.huggingface_dataset = huggingface_dataset

    def __getitem__(self, idx) -> list:
        return self.huggingface_dataset[idx]

    def __len__(self) -> int:
        return len(self.huggingface_dataset)


class CharTokenizer:
    def __init__(self, max_length=512) -> None:
        self.sep_token  = '[SEP]'
        self.mask_token = '[MASK]'
        self.pad_token  = '[PAD]'
        self.unk_token  = '[UNK]'

        self.sep_token_dummy  = chr(252)
        self.mask_token_dummy = chr(253)
        self.pad_token_dummy  = chr(254)
        self.unk_token_dummy  = chr(255)

        ## regex to replace dummy chars naturally present with unk_token.
        self.remap_regex_string = '[' + self.sep_token_dummy + self.mask_token_dummy + self.pad_token_dummy + ']'

        self.max_length = max_length
        self.attention_const_mask = [1] * self.max_length


    def __len__(self) -> int:
        ## Vocab size 256
        return 256

    def __call__(self, text: list) -> dict:
        return self.encode(text)

    def encode(self, texts: list) -> dict:
        id_list = []
        for idx, text in enumerate(texts):
            text = re.sub(self.remap_regex_string, self.unk_token_dummy, text)

            text = text.replace(self.sep_token, self.sep_token_dummy)
            text = text.replace(self.mask_token, self.mask_token_dummy)
            text = text.replace(self.pad_token, self.pad_token_dummy)
            text = text.replace(self.unk_token, self.unk_token_dummy)

            byte_list = []
            for idx, char in enumerate(text):
                ## Truncate
                if idx == self.max_length:
                    break

                ## Vocab size 256
                byte_list.append(min(255, ord(char)))

            ## Pad to max_length
            if len(byte_list) < self.max_length:
                byte_list += [254] * (self.max_length - len(byte_list))

            id_list.append(byte_list)
        
        return {
                'input_ids': id_list,
                'attention_mask': len(id_list) * [self.attention_const_mask],
                }

    def decode(self, text_ids: list) -> str:
        decoded_string = ''.join([chr(idx) for idx in text_ids])
        return decoded_string


class DataHandler:
    def __init__(
            self, 
            dataset_name='bookcorpus', 
            max_length=512, 
            subset_size=None,
            num_workers=4,
            pin_memory=True,
            dtype=T.float16,
            device=T.device('cuda:0' if T.cuda.is_available() else 'cpu'),
            eval=False
            ) -> None:
        ## Read data function
        self.text_data   = datasets.load_dataset(dataset_name, 'plain_text', split='train')['text']
        self.max_length  = max_length
        self.subset_size = subset_size
        self.num_workers = num_workers
        self.pin_memory  = pin_memory

        if not eval:
            ## Concatenate dataset to be of max_length.
            self.condense_dataset()

        self.tokenizer = CharTokenizer(max_length=max_length)

        self.vocab_size  = len(self.tokenizer)
        self.device      = device
        self.dtype       = dtype


    def condense_dataset(self):
        if self.subset_size is not None:
            self.text_data = self.text_data[:self.subset_size]

        self.text_data = ' [SEP] '.join(self.text_data)
        self.text_data = self.text_data.split(' ')

        n_batches = 1 + (len(self.text_data) // self.max_length)

        final_dataset = []
        for idx in tqdm(range(n_batches), desc='Preparing Dataset'):
            final_dataset.append(' '.join(self.text_data[idx * self.max_length:(idx + 1) * self.max_length]))
        final_dataset.append(' '.join(self.text_data[n_batches * self.max_length:]))

        self.text_data = final_dataset
        del final_dataset
        gc.collect()


    def get_dataloader(self, micro_batch_size=1024) -> DataLoader:
        collate_fn = CustomCollator(
                tokenizer=self.tokenizer,
                mlm_prob=0.15,
                max_length=self.max_length,
                dtype=self.dtype,
                batch_size=micro_batch_size
                )

        dataloader = DataLoader(
                dataset=TorchDatasetWrapper(self.text_data),
                collate_fn=collate_fn,
                shuffle=True,
                batch_size=micro_batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                drop_last=True,
                prefetch_factor=2,
                persistent_workers=True
                )
        return dataloader



class CustomCollator:
    def __init__(
            self, 
            tokenizer, 
            mlm_prob=0.15, 
            dtype=T.float16, 
            max_length=512, 
            batch_size=1024
            ) -> None:
        self.tokenizer      = tokenizer
        self.mlm_prob       = mlm_prob
        self.dtype          = dtype
        self.max_length     = max_length
        self.batch_size     = batch_size

        self.sep_token_id   = 252
        self.mask_token_id  = 253
        self.pad_token_id   = 254
        self.unk_token_id   = 255


    def __call__(self, batch) -> (T.tensor, T.tensor):
        token_ids, attention_mask = self.encode(batch)

        token_ids      = T.tensor(token_ids, dtype=T.long)
        attention_mask = T.tensor(attention_mask, dtype=T.long)

        token_ids, attention_mask, original_labels = self.mask_inputs(token_ids, attention_mask)

        return token_ids, attention_mask, original_labels


    def encode(self, batch: list) -> (list, list):
        tokenized_output = self.tokenizer(batch)
        token_ids        = tokenized_output['input_ids']
        attention_mask   = tokenized_output['attention_mask']

        return token_ids, attention_mask



    def mask_inputs(self, token_ids: T.tensor, attention_mask: T.tensor) -> (T.tensor, T.tensor, T.tensor, T.tensor):
        ## Bechmarked. Does not actually slow down execution. 
        ## GPU forward/backward pass is limiting factor w/ 4 workers and prefetch_factor=2.
        original_ids = token_ids.clone().flatten()

        special_token_ids = [self.sep_token_id, self.mask_token_id, self.pad_token_id, self.unk_token_id]

        ## mask (i.e. (1, 0, 0, 1, ...)) where mask tokens (i.e. '[MASK]') 
        ## are applied. Hence -> mask_mask
        mask_mask = T.zeros_like(attention_mask)
        for idx, batch in enumerate(token_ids):
            non_special_idxs = np.argwhere(np.isin(batch, special_token_ids) == 0).squeeze()

            mask_idxs = np.random.choice(
                    non_special_idxs,
                    size=int(non_special_idxs.shape[0] * self.mlm_prob),
                    replace=False
                    )

            mask_mask[idx, mask_idxs] = 1
            token_ids[idx, mask_idxs] = self.mask_token_id

        non_mask_idxs = T.argwhere(mask_mask.flatten() == 0).squeeze()

        ## Mask non-mask tokens to -100 for torch.CrossEntropyLoss
        original_ids[non_mask_idxs] = -100

        return token_ids, attention_mask, original_ids





if __name__ == '__main__':
    ###############################
    ####        TESTS          ####
    ###############################

    handler = DataHandler(dataset_name='bookcorpus', subset_size=100000)
    dataloader = handler.get_dataloader(micro_batch_size=32)

    for idx, (X, attention_mask, y) in enumerate(dataloader):
        #print(X[0].shape)
        #print(y[:32].shape)
        print(handler.tokenizer.decode(X[0]))
        print(handler.tokenizer.decode([x for x in y[:32] if x != -100]))
        if idx == 1:
            break
