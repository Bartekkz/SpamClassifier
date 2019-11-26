#!/usr/bin/env python3

#def load_sms_data(path)
#    '''
#    load text data from given path 
#    :param: path(str) -> path to data file
#    :returns -> list of texts of sms and labels for them
#    '''
#    data = open(path) 
#    sms_text = []
#    labels = []
#    for line in data: 
#        line = line.split() 
#        label = line[0]
#        text = line[1:]
#        sms_text.append(text)
#        labels.append(label)        
#
#    return sms_text, labels


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
            sms_text.append(text)
            labels.append(label)        
        return sms_text, labels



if __name__ == '__main__':
    loader = DataLoader('../data/sms_data')
    print(loader.__len__())
    print(loader.__getitem__(500))
