import torch as T
import torch.nn as nn
from transformers import T5ForConditionalGeneration, AutoTokenizer
import random


class HuggingFaceByt5Wrapper(nn.Module):
    def __init__(self, device=None) -> None:
        super(HuggingFaceByt5Wrapper, self).__init__()
        self.model = T5ForConditionalGeneration.from_pretrained("google/byt5-small")
        self.tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")

        self.device = device
        if device is None:
            self.device = T.device("cuda" if T.cuda.is_available() else "cpu") 
        self.to(self.device)

    def forward(self, X: str, attention_mask: T.tensor = None) -> T.tensor:
        X = self.tokenizer(X, return_tensors="pt").input_ids.to(self.device)
        return self.model.generate(X, max_length=100)

    def get_embeddings(self, X: str, attention_mask: T.tensor = None) -> T.tensor:
        with T.no_grad():
            X = self.tokenizer(X, return_tensors="pt").input_ids.to(self.device)
            return self.model.get_input_embeddings()(X).mean(dim=-2).squeeze()



if __name__ == "__main__":
    model = HuggingFaceByt5Wrapper()
    string = f"The dog chases a ball in the park."

    output = model.get_embeddings(string)
    print(output.shape)
