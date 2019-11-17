import pytest
import sys
sys.path.append('..')
from utils.sms_preprocessor import SmsPreprocessor



@pytest.mark.parametrize("test_input, expected ", [
    ("Hello my dear friend. My Friend told me", {"Hello": 1, "my": 2, "dear": 3, "friend.": 4,
                                                 "My": 5, "Friend": 6, "told": 7, "me": 8}),
    (["Hello what Is going on What", "fuck what"], {"Hello": 1, "what": 2, "Is": 3, "going": 4,
                                                    "on": 5, "What": 6, "fuck": 7})
])
def test_preprocessor_create_key_word_map(test_input, expected):
    pr = SmsPreprocessor(False)
    key_word_map = pr.create_key_word_dict(test_input) 
    assert key_word_map == expected 


def test_preprocessor_preprocess():
    pr = SmsPreprocessor(False)
    text = "Hello what is going?! NOthing."
    tokenized = pr.preprocess(text, False)
    expected = 'hello what is going nothing'
    assert tokenized == expected

@pytest.mark.parametrize("test_input, expected", [
    (["hello is she", "you are good"], ["hello", "good"]),
    ("hello is she", "hello")
])
def test_preprocessor_remove_stopwords(test_input, expected):
    pr = SmsPreprocessor(False)
    cleaned_text = pr.remove_stopwords(test_input)
    assert cleaned_text == expected
    

@pytest.mark.parametrize("test_input, expected, key_word_map", [
    ("Hello my dear friend. My borther told me", [1, 2, 3, 4, 5, 6, 7, 8], None),
    ("hello what is going on. What?", [1, 2, 3, 4, 5, 7], {"hello": 1, "what": 2, "is": 3, "going": 4, "on.": 5, "What": 6, "<unk>": 7})
])
def test_preprocessor_tokenizer(test_input, expected, key_word_map):
    pr = SmsPreprocessor(False)
    tokenized, key_word_map = pr.tokenize(test_input, key_word_map=key_word_map)
    assert tokenized == expected 


@pytest.mark.parametrize("test_input, expected", [
    (["Hello my Friend?!", "What. is GOing on"], ["hello my friend", "what is going on"])
])
def test_preprocessor_clean_text(test_input, expected):
    pr = SmsPreprocessor(False)
    cleaned = pr.clean_text(test_input)
    assert cleaned == expected


@pytest.mark.parametrize("test_input, expected", [
    ("I was running and walking", "I wa run and walk"),
    (["I was running and walking", "He was singing"], ["I wa run and walk", "He wa sing"])
])
def test_preprocessor_stemmer(test_input, expected):
    from nltk.stem import PorterStemmer
    ps = PorterStemmer()
    pr = SmsPreprocessor(False)
    stemmed = pr.stem_text(test_input)
    assert stemmed == expected 
    

