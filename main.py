from utils.sms_preprocessor import SmsPreprocessor
from utils.tokenizer import Tokenizer
from sklearn.pipeline import Pipeline



pipeline = Pipeline([
    ("preprocessor", SmsPreprocessor(False)),
    ("tokenizer", Tokenizer())
])






