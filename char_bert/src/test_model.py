import torch as T
import torch.nn as nn
from transformers import T5ForConditionalGeneration, AutoTokenizer
import random


class HuggingFaceByt5Wrapper(nn.Module):
    def __init__(self, device) -> None:
        super(HuggingFaceByt5Wrapper, self).__init__()
        self.model = T5ForConditionalGeneration.from_pretrained("google/byt5-base")
        self.tokenizer = AutoTokenizer.from_pretrained("google/byt5-base")

        self.device = device
        self.to(self.device)



    def forward(self, X: str, attention_mask: T.tensor = None) -> T.tensor:
        X = self.tokenizer(X, return_tensors="pt").input_ids.to(self.device)
        print(X)
        return self.model.generate(X, max_length=100)



if __name__ == "__main__":
    model = HuggingFaceByt5Wrapper(T.device("cuda:0"))
    string = f"The dog chases a <extra_id_0><extra_id_1><extra_id_2><extra_id_3> in the park."

    output = model(string)
    print(output)
    print(model.tokenizer.batch_decode(output, skip_special_tokens=True))
