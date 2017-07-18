from instance import  Instance

class Reader:
    def __init__(self, maxInst = -1):
        self.maxInstance = maxInst

    def readInstances(self, path, maxInst = -1):
        insts = []
        r = open(path)
        inst = Instance()
        for line in r.readlines():
            line = line.strip()
            if line == "" and len(inst.words) != 0:
                if (self.maxInstance == -1) or (self.maxInstance > len(insts)):
                    insts.append(inst)
                else:
                    return insts
                inst = Instance()
            else:
                info = line.split(" ")
                if len(info) != 3:
                    print("error format")
                inst.words.append(info[0])
                inst.labels.append(info[2])
        r.close()
        if len(inst.words) != 0:
            insts.append(inst)
        return insts

