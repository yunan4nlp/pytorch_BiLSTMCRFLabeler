import random
import  torch
from optparse import OptionParser
import torch.nn
import torch.autograd
import torch.nn.functional
from read import Reader
from instance import Feature
from instance import Example
from hyperparams import HyperParams
from model import RNNLabeler
from eval import Eval


class Labeler:
    def __init__(self):
        self.word_state = {}
        self.label_state = {}
        self.hyperParams = HyperParams()

    def createAlphabet(self, trainInsts, devInsts, testInsts):
        print("create alpha.................")
        for inst in trainInsts:
            for w in inst.words:
                if w not in self.word_state:
                    self.word_state[w] = 1
                else:
                    self.word_state[w] += 1

            for l in inst.labels:
                if l not in self.label_state:
                    self.label_state[l] = 1
                else:
                    self.label_state[l] += 1

        print("word state:", len(self.word_state))
        self.addTestAlpha(devInsts)
        print("word state:", len(self.word_state))
        self.addTestAlpha(testInsts)
        print("word state:", len(self.word_state))

        self.word_state[self.hyperParams.unk] = self.hyperParams.wordCutOff + 1
        self.hyperParams.wordAlpha.initial(self.word_state, self.hyperParams.wordCutOff)
        self.hyperParams.wordAlpha.set_fixed_flag(True)
        self.hyperParams.wordNum = self.hyperParams.wordAlpha.m_size
        self.hyperParams.unkWordID = self.hyperParams.wordAlpha.from_string(self.hyperParams.unk)

        self.hyperParams.labelAlpha.initial(self.label_state)
        self.hyperParams.labelAlpha.set_fixed_flag(True)
        self.hyperParams.labelSize = self.hyperParams.labelAlpha.m_size

        print("Label num: ", self.hyperParams.labelSize)
        print("Word num: ", self.hyperParams.wordNum)

    def addTestAlpha(self, insts):
        print("Add test alpha.............")
        if self.hyperParams.wordFineTune == False:
            for inst in insts:
                for w in inst.words:
                    if (w not in self.word_state):
                        self.word_state[w] = 1
                    else:
                        self.word_state[w] += 1

    def extractFeature(self, inst):
        feat = Feature()
        for w in inst.words:
            wordId = self.hyperParams.wordAlpha.from_string(w)
            if wordId == -1:
                feat.wordIndexs.append(self.hyperParams.unkWordID)
            else:
                feat.wordIndexs.append(wordId)
        feat.wordIndexs = torch.autograd.Variable(torch.LongTensor(feat.wordIndexs))
        return feat

    def instance2Example(self, insts):
        exams = []
        for inst in insts:
            example = Example()
            example.feat = self.extractFeature(inst)
            for l in inst.labels:
                labelId = self.hyperParams.labelAlpha.from_string(l)
                example.labelIndexs.append(labelId)
            example.labelIndexs = torch.autograd.Variable(torch.LongTensor(example.labelIndexs))
            exams.append(example)
        return exams



    def train(self, train_file, dev_file, test_file):
        self.hyperParams.show()
        torch.set_num_threads(self.hyperParams.thread)
        reader = Reader(self.hyperParams.maxInstance)

        trainInsts = reader.readInstances(train_file)
        devInsts = reader.readInstances(dev_file)
        testInsts = reader.readInstances(test_file)

        print("Training Instance: ", len(trainInsts))
        print("Dev Instance: ", len(devInsts))
        print("Test Instance: ", len(testInsts))

        self.createAlphabet(trainInsts, devInsts, testInsts)

        trainExamples = self.instance2Example(trainInsts)
        devExamples = self.instance2Example(devInsts)
        testExamples = self.instance2Example(testInsts)

        self.model = RNNLabeler(self.hyperParams)
        parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = torch.optim.Adagrad(parameters, lr=self.hyperParams.learningRate)

        indexes = []
        for idx in range(len(trainExamples)):
            indexes.append(idx)
        for iter in range(self.hyperParams.maxIter):
            print('###Iteration' + str(iter) + "###")
            random.shuffle(indexes)
            for idx in range(len(trainExamples)):
                self.model.zero_grad()
                self.model.LSTMHidden = self.model.init_hidden()
                exam = trainExamples[indexes[idx]]
                lstm_feats = self.model(exam.feat)
                loss = self.model.crf.neg_log_likelihood(lstm_feats, exam.labelIndexs)
                loss.backward()
                optimizer.step()
                if (idx + 1) % self.hyperParams.verboseIter == 0:
                    print('current: ', idx + 1,  ", cost:", loss.data[0])

            eval_dev = Eval()
            for idx in range(len(devExamples)):
                predictLabels = self.predict(devExamples[idx])
                devInsts[idx].evalPRF(predictLabels, eval_dev)
            print('Dev: ', end="")
            eval_dev.getFscore()

            eval_test = Eval()
            for idx in range(len(testExamples)):
                predictLabels = self.predict(testExamples[idx])
                testInsts[idx].evalPRF(predictLabels, eval_test)
            print('Test: ', end="")
            eval_test.getFscore()


    def predict(self, exam):
        tag_hiddens = self.model(exam.feat)
        _, best_path = self.model.crf._viterbi_decode(tag_hiddens)
        predictLabels = []
        for idx in range(len(best_path)):
            predictLabels.append(self.hyperParams.labelAlpha.from_id(best_path[idx]))
        return predictLabels

    def getMaxIndex(self, tag_score):
        max = tag_score.data[0]
        maxIndex = 0
        for idx in range(1, self.hyperParams.labelSize):
            if tag_score.data[idx] > max:
                max = tag_score.data[idx]
                maxIndex = idx
        return maxIndex


parser = OptionParser()
parser.add_option("--train", dest="trainFile",
                  help="train dataset")

parser.add_option("--dev", dest="devFile",
                  help="dev dataset")

parser.add_option("--test", dest="testFile",
                  help="test dataset")


(options, args) = parser.parse_args()
l = Labeler()
l.train(options.trainFile, options.devFile, options.testFile)

