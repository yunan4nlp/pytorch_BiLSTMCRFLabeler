import torch.nn as nn
import torch.autograd
from CRF import CRF

class RNNLabeler(nn.Module):
    def __init__(self, hyperParams):
        super(RNNLabeler,self).__init__()

        self.hyperParams = hyperParams
        if hyperParams.wordEmbFile == "":
            self.wordEmb = nn.Embedding(hyperParams.wordNum, hyperParams.wordEmbSize)
        else:
            self.wordEmb = self.load_pretrain(hyperParams.wordEmbFile, hyperParams.wordAlpha)
        self.wordEmb.weight.requires_grad = hyperParams.wordFineTune
        self.LSTM = nn.LSTM(hyperParams.wordEmbSize, hyperParams.rnnHiddenSize // 2, bidirectional=True)
        self.LSTMHidden = self.init_hidden()
        self.linearLayer = nn.Linear(hyperParams.rnnHiddenSize, hyperParams.labelSize, bias=True)
        self.crf = CRF(hyperParams.labelSize)

    def init_hidden(self):
       return (torch.autograd.Variable(torch.randn(2, 1, self.hyperParams.rnnHiddenSize // 2)),
                torch.autograd.Variable(torch.randn(2, 1, self.hyperParams.rnnHiddenSize // 2)))

    def load_pretrain(self, file, alpha):
        f = open(file)
        allLines = f.readlines()
        indexs = []
        info = allLines[0].strip().split(' ')
        emb = nn.Embedding(self.hyperParams.wordNum, len(info) - 1)
        oov_emb = torch.zeros(1, len(info) - 1).type(torch.FloatTensor)
        for line in allLines:
            info = line.strip().split(' ')
            wordID = alpha.from_string(info[0])
            if wordID >= 0:
                indexs.append(wordID)
                for idx in range(len(info) - 1):
                    val = float(info[idx + 1])
                    emb.weight.data[wordID][idx] = val
                    oov_emb[0][idx] += val
        f.close()
        count = len(indexs)
        for idx in range(len(info) - 1):
            oov_emb[0][idx] /= count

        unkID = self.hyperParams.wordAlpha.from_string(self.hyperParams.unk)
        print('UNK ID: ', unkID)
        if unkID != -1:
            for idx in range(len(info) - 1):
                emb.weight.data[unkID][idx] = oov_emb[0][idx]

        print("Load Embedding file: ", file, ", size: ", len(info) - 1)
        oov = 0
        for idx in range(self.hyperParams.wordNum):
            if idx not in indexs:
                oov += 1
        print("OOV Num: ", oov, "Total Num: ", self.hyperParams.wordNum,
              "OOV Ratio: ", oov / self.hyperParams.wordNum)
        print("OOV ", self.hyperParams.unk, "use avg value initialize")
        return emb

    def forward(self, feat):
        sentSize = len(feat.wordIndexs)
        wordRepresents = self.wordEmb(feat.wordIndexs)
        LSTMOutputs, self.LSTMHidden = self.LSTM(wordRepresents.view(sentSize, 1, -1), self.LSTMHidden)
        tagHiddens = self.linearLayer(LSTMOutputs.view(sentSize, -1))
        return tagHiddens










