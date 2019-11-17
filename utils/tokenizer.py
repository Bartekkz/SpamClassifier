'''
Functions to extract word_key_maps and perform tokenization
'''
from sklearn.base import BaseEstimator, TransformerMixin       



class Tokenizer(BaseEstimator, TransformerMixin):

    def tokenize(self, text, key_word_map=None, strategy="unknown"):
        #TODO: add other strategies, change function
        tokenized = []
        if key_word_map is None:
            key_word_map = self.create_key_word_dict(text)
        if isinstance(text, str):
            for word in text.split():
                if word in key_word_map.keys():
                    token = key_word_map.get(word)
                    tokenized.append(token)
                else:
                    if strategy == "unknown":
                        token = key_word_map.get("<unk>")
                        tokenized.append(token)
                    elif strategy == 'zeros':
                        tokenized.append(0)
            return tokenized, key_word_map
        for sent in text:
            tokenized_text = []
            for word in sent.split():
                if word in key_word_map.keys():
                    token = key_word_map.get(word)
                    tokenized_text.append(token)
                else:
                    if strategy == "unknown":
                        token = key_word_map.get("<unk>")
                        tokenized_text.append(token)
                    elif strategy == 'zeros':
                        tokenized_text.append(0)
            tokenized.append(tokenized_text)
        return tokenized, key_word_map


    def create_key_word_dict(self, text: [str]):
        '''
        Create dictionary of word, token pairs
        @params:
        - text: str or list(str)
        - clean_text: bool -> lowercase text and remove punctuation
        '''
        key_word_map = {}
        if isinstance(text, str):
            text = [text]
        i = 1 
        for sentence in text:
            for word in sentence.split(): 
                if word in key_word_map.keys():
                    continue 
                key_word_map[word] = i 
                i += 1
        key_word_map["<unk>"] = i
        return key_word_map

    
    def transform(self, X, y=None):
        print('Tokenizing...')
        tokenized = self.tokenize(X)
        print('Finished!')
        return tokenized
    

    def fit(self, X, y=None):
        return self

