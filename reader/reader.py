import wget
import numpy as np
from os.path import exists
from sklearn.model_selection import train_test_split

from config import EDGAR_ALAN_POE_TXT_PATH, ROBERT_FROST_TXT_PATH, TEST_PERCENTAGE


class Reader:

    def __init__(self, edgarAlanPoeUrl, robertFrostUrl):
        if not exists(EDGAR_ALAN_POE_TXT_PATH):
            self.edgarAlanPoeFile = wget.download(edgarAlanPoeUrl)
        else:
            self.edgarAlanPoeFile = EDGAR_ALAN_POE_TXT_PATH
        if not exists(ROBERT_FROST_TXT_PATH):
            self.robertFrostFile = wget.download(robertFrostUrl)
        else:
            self.robertFrostFile = ROBERT_FROST_TXT_PATH

    def readEdgarAlanPoeTxt(self):
        edgarAlanPoeLineList = []
        with open(self.edgarAlanPoeFile, 'r') as ef:
            for line in ef:
                edgarAlanPoeLineList.append(line)
        edgarAlanPoeLineLabelList = np.zeros(len(edgarAlanPoeLineList))
        return edgarAlanPoeLineList, edgarAlanPoeLineLabelList

    def readRobertFrostTxt(self):
        robertFrostLineList = []
        with open(self.robertFrostFile, 'r') as rf:
            for line in rf:
                robertFrostLineList.append(line)
        robertFrostLineLabelList = np.ones(len(robertFrostLineList))
        return robertFrostLineList, robertFrostLineLabelList

    def train_test_split(self):
        edgarAlanPoeLineList, edgarAlanPoeLineLabelList = self.readEdgarAlanPoeTxt()
        robertFrostLineList, robertFrostLineLabelList = self.readRobertFrostTxt()
        data = edgarAlanPoeLineList + robertFrostLineList
        target = np.hstack((edgarAlanPoeLineLabelList, robertFrostLineLabelList))
        return train_test_split(data, target, test_size=TEST_PERCENTAGE)
