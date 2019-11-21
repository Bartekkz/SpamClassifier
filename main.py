from utils.sms_preprocessor import SmsPreprocessor
from utils.tokenizer import Tokenizer
from sklearn.pipeline import Pipeline



text = ["Hello man what do you want? what is going on", 'what do going You want']

pipeline = Pipeline([
    ("preprocessor", SmsPreprocessor(False)),
    ("tokenizer", Tokenizer())
])


if __name__ == '__main__':
    tokenized, key_word_map = pipeline.fit_transform(text)
    print(tokenized)
    print(key_word_map)





