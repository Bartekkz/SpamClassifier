#!/usr/bin/env python3


class DataLoader:
    def __init__(self, path):
        self.path = path
        self.sms_data, self.labels = self.load_sms_data()

    def __getitem__(self, idx):
        return {'SMS': self.sms_data[idx], 'Label': self.labels[idx]} 

    def __len__(self):
        return len(self.sms_data)

    def load_sms_data(self):
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
        return sms_text, labels

