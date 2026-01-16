from torch.utils.data import DataLoader, Dataset
import torch
import tiktoken

'''
    creating the dataset class to create a dataset for text prediction using sliding window
    input will be 4 words and the target is the next 4 words (rolling) 
    
    ex: a b c d -> input
          b c d e > target 
          
    all dataset classesMUST implement the __getitem__ and __len__ dunders
'''

class Dtset(Dataset) :
    def __init__(self, text, tokenizer,stride,max_length) :
        self.input = []
        self.label = []
        tokens = tokenizer.encode(text)

        for i in range(0,len(tokens)-max_length,stride) :
            inp = tokens[i:i+max_length]
            tar = tokens[i+1:i+max_length+1]
            self.input.append(inp)
            self.label.append(tar)

    def __len__(self) :
        return len(self.input)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.input[idx]),
            torch.tensor(self.label[idx])
        )

def create_dataloader_v1(txt, batch_size=4, max_length=256,
    stride=128, shuffle=True, drop_last=True,
    num_workers=0):

    tokenizer = tiktoken.get_encoding('gpt2')

    dataset = Dtset(txt, tokenizer, stride, max_length)

    dataloader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers,
        drop_last=drop_last, shuffle=shuffle
    )

    return dataloader

#
# with open("the-verdict.txt") as f:
#     text = f.read()
#
# dataloader = create_dataloader_v1(text, batch_size=1, max_length=4, stride=1, shuffle=False)
#
# data_iter = iter(dataloader)    # creating an interator for the testing
#
# first_iter= next(data_iter)
#
# print(first_iter)

