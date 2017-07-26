import torch.nn as nn
import torch.autograd
from CRF import CRF

class RNNLabeler(nn.Module):
    def __init__(self, hyperParams):
        super(RNNLabeler,self).__init__()

        self.hyperParams = hyperParams
        self.wordEmb = nn.Embedding(hyperParams.wordNum, hyperParams.wordEmbSize)
        self.LSTM = nn.LSTM(hyperParams.wordEmbSize, hyperParams.rnnHiddenSize // 2, bidirectional=True)
        self.LSTMHidden = self.init_hidden()
        self.linearLayer = nn.Linear(hyperParams.rnnHiddenSize, hyperParams.labelSize, bias=True)
        self.crf = CRF(hyperParams.labelSize)

    def init_hidden(self):
       return (torch.autograd.Variable(torch.randn(2, 1, self.hyperParams.rnnHiddenSize // 2)),
                torch.autograd.Variable(torch.randn(2, 1, self.hyperParams.rnnHiddenSize // 2)))


    def forward(self, feat):
        sentSize = len(feat.wordIndexs)
        wordRepresents = self.wordEmb(feat.wordIndexs)

        LSTMOutputs, self.LSTMHidden = self.LSTM(wordRepresents.view(sentSize, 1, -1), self.LSTMHidden)
        tagHiddens = self.linearLayer(LSTMOutputs.view(sentSize, -1))
        return tagHiddens










