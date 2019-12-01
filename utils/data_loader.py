#!/usr/bin/env python3


class DataLoader:
    def __init__(self, path, one_hot_labels):
        self.path = path
        if one_hot_labels:
            self.sms_data, self.labels = self.load_sms_data(True)
        else:
            self.sms_data, self.labels = self.load_sms_data()

    def __getitem__(self, idx):
        return {'SMS': self.sms_data[idx], 'Label': self.labels[idx]} 

    def __len__(self):
        return len(self.sms_data)

    def load_sms_data(self, one_hot_labels=False):
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
        if one_hot_labels:
            #TODO: Convert labels, and sms_text to to np.arrays
            labels =[[0, 1] if label == 'ham' else [1, 0] for label in labels]
        return sms_text, labels
