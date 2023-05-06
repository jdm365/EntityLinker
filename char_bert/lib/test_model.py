import torch as T
import torch.nn as nn
from transformers import T5ForConditionalGeneration, AutoTokenizer


## Create enum for model size
class ModelSize:
    SMALL = "small"
    BASE  = "base"
    LARGE = "large"
    XLARGE = "xlarge"



class HuggingFaceByt5Wrapper(nn.Module):
    def __init__(self, model_size=None, device=None) -> None:
        super(HuggingFaceByt5Wrapper, self).__init__()
        self.model_size = model_size
        if model_size is None:
            self.model_size = ModelSize.BASE

        self.model = T5ForConditionalGeneration.from_pretrained(f"google/byt5-{self.model_size}")
        self.tokenizer = AutoTokenizer.from_pretrained(f"google/byt5-{self.model_size}")

        self.device = device
        if device is None:
            self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu") 
        self.to(self.device)

    def forward(self, X: str, attention_mask: T.tensor = None) -> T.tensor:
        X = self.tokenizer(X, return_tensors="pt").input_ids.to(self.device)
        return self.model.generate(X, max_length=100)

    def get_embeddings(self, X: str, attention_mask: T.tensor = None) -> T.tensor:
        with T.no_grad():
            X = self.tokenizer(X, return_tensors="pt").input_ids.to(self.device)
            return self.model.get_input_embeddings()(X).mean(dim=-2).squeeze()



if __name__ == "__main__":
    model_size = ModelSize.LARGE
    model = HuggingFaceByt5Wrapper(model_size=model_size)
    string = f"The dog chases a ball in the park."

    output = model.get_embeddings(string)
    print(output.shape)
