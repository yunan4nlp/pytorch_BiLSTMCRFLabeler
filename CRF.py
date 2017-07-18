import torch
import torch.autograd as autograd
import torch.nn as nn

# Helper functions to make the code more readable.
def to_scalar(var):
    # returns a python float
    return var.view(-1).data.tolist()[0]

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return to_scalar(idx)

# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

class CRF(nn.Module):
    def __init__(self, labelSize):
        super(CRF, self).__init__()
        self.labelSize = labelSize
        self.T = nn.Parameter(torch.randn(self.labelSize, self.labelSize))

    def _forward_alg(self, feats):
        init_alphas = torch.Tensor(1, self.labelSize).fill_(0)
        # Wrap in a variable so that we will get automatic backprop
        forward_var = autograd.Variable(init_alphas)
        # Iterate through the sentence
        for idx in range(len(feats)):
            feat = feats[idx]
            alphas_t = []  # The forward variables at this timestep
            for next_tag in range(self.labelSize):
                if idx == 0:
                    alphas_t.append(feat[next_tag].view(1, -1))
                else:
                    emit_score = feat[next_tag].view(1, -1).expand(1, self.labelSize)
                    trans_score = self.T[next_tag]
                    next_tag_var = forward_var + trans_score + emit_score
                    alphas_t.append(log_sum_exp(next_tag_var))
            forward_var = torch.cat(alphas_t).view(1, -1)
        alpha_score = log_sum_exp(forward_var)
        return alpha_score


    def _viterbi_decode(self, feats):
        init_score = torch.Tensor(1, self.labelSize).fill_(0)
        forward_var = autograd.Variable(init_score)
        back = []
        #print(feats[0].view(1,-1).expand(1, self.labelSize))
        #print(feats[0])
        for idx in range(len(feats)):
            feat = feats[idx]
            bptrs_t = []
            viterbi_var = []
            for next_tag in range(self.labelSize):
                if idx == 0:
                    viterbi_var.append(feat[next_tag].view(1, -1))
                else:
                    emit_score = feat[next_tag].view(1, -1).expand(1, self.labelSize)
                    trans_score = self.T[next_tag]
                    next_tag_var = forward_var + trans_score + emit_score
                    best_label_id = argmax(next_tag_var)
                    bptrs_t.append(best_label_id)
                    viterbi_var.append(next_tag_var[0][best_label_id])
            forward_var = (torch.cat(viterbi_var)).view(1, -1)
            if idx > 0:
                back.append(bptrs_t)
        best_label_id = argmax(forward_var)
        best_path = [best_label_id]
        path_score = forward_var[0][best_label_id]
        for bptrs_t in reversed(back):
            best_label_id = bptrs_t[best_label_id]
            best_path.append(best_label_id)
        best_path.reverse()
        return path_score, best_path

    def _score_sentence(self, feats, labels):
        if len(feats) != len(labels):
            print('error in score sentence')
        score = autograd.Variable(torch.Tensor([0]))
        for idx in range(len(feats)):
            feat = feats[idx]
            if idx == 0:
                score += feat[labels[idx]]
            else:
                score += feat[labels[idx]] + self.T[labels[idx].data[0], labels[idx - 1].data[0]]
        return score

    def neg_log_likelihood(self, feats, tags):
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

