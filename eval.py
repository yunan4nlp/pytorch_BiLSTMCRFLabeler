class Eval:
    def __init__(self):
        self.predict_num = 0
        self.correct_num = 0
        self.gold_num = 0

        self.precision = 0
        self.recall = 0
        self.fscore = 0

    def clear(self):
        self.predict_num = 0
        self.correct_num = 0
        self.gold_num = 0

        self.precision = 0
        self.recall = 0
        self.fscore = 0

    def getFscore(self):
        self.precision = self.correct_num / self.predict_num
        self.recall = self.correct_num / self.gold_num
        self.fscore = 2 * (self.precision * self.recall) / (self.precision + self.recall)

        print("precision: ", self.precision, ", recall: ", self.recall, ", fscore: ", self.fscore)

