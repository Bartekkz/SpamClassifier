#!/usr/bin/env python3
import numpy as np

class DataLoader:
    def __init__(self, path, convert_to_int=True):
        self.path = path
        if convert_to_int:
            self.sms_data, self.labels = self.load_sms_data(True)
        else:
            self.sms_data, self.labels = self.load_sms_data(False)

    def __getitem__(self, idx):
        return {'SMS': self.sms_data[idx], 'Label': self.labels[idx]} 

    def __len__(self):
        return len(self.sms_data)

    def load_sms_data(self, convert_to_int):
        """
        Load sms_data from given path
        :param convert_to_int: converts labels to int
        :return: np.array of sms_text and np.array of labels
        """
        data = open(self.path) 
        sms_text = []
        labels = []
        for line in data: 
            line = line.split() 
            label = line[0]
            text = line[1:]
            text = ' '.join(word for word in text)
            sms_text.append(text)
            labels.append(label)
        if convert_to_int:
            labels = np.array([0 if label == 'ham' else 1 for label in labels])
        return np.asarray(sms_text), labels
