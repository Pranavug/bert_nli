import torch
from bert_nli import CNNLarge

if __name__ == '__main__':
    model = CNNLarge()
    sample_input = torch.zeros((16, 50, 768))
    sample_input = sample_input.unsqueeze(1)

    print(sample_input.shape)

    output = model(sample_input)
    output = output.squeeze()

    print(output.shape)
