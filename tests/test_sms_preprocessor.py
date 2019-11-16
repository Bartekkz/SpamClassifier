import pytest
import sys
sys.path.append('..')
from utils.sms_preprocessor import SmsPreprocessor




@pytest.mark.parametrize("test_input, expected, clean_text", [
    ("Hello my dear friend. My Friend told me", {"Hello": 1, "my": 2, "dear": 3, "friend": 4,
                                                 "My": 5, "Friend": 6, "told": 7,
                                                 "me": 8}, False),
    (["Hello what Is going on . What ?", "fuck what"], {"hello": 1, "what": 2, "is": 3, "going": 4, "on": 5, "fuck": 6}, True)
])
def test_preprocessor_create_key_word_map(test_input, clean_text, expected):
    pr = SmsPreprocessor(False)
    key_word_map = pr.create_key_word_dict(test_input, clean_text) 
    assert key_word_map == expected 


def test_preprocessor_preprocess():
    pr = SmsPreprocessor(False)
    text = "Hello what is going?! NOthing."
    tokenized = pr.preprocess(text)
    expected = 'hello what is going nothing'
    assert tokenized == expected


def test_preprocessor_remove_stopwords():
    pr = SmsPreprocessor(False)
    text = ["hello is she", "you are good"]
    cleaned_text = pr.remove_stopwords(text)
    expected = ["hello", "good"]
    assert cleaned_text == expected
    


@pytest.mark.parametrize("test_input, expected, key_word_map", [
    ("Hello my dear friend. My borther told me", [1, 2, 3, 4, 2, 5, 6, 7], None),
    ("hello what is going on. What?", [1, 2, 3, 4, 5, 2], {"hello": 1, "what": 2, "is": 3, "going": 4, "on": 5})
])
def test_preprocessor_tokenizer(test_input, expected, key_word_map):
    pr = SmsPreprocessor(False)
    tokenized = pr.tokenize(test_input, key_word_map=key_word_map)
    assert tokenized == expected 


@pytest.mark.parametrize("test_input, expected", [
    (["Hello my Friend?!", "What. is GOing on"], ["hello my friend", "what is going on"])
])
def test_preprocessor_clean_text(test_input, expected):
    pr = SmsPreprocessor(False)
    cleaned = pr.clean_text(test_input)
    assert cleaned == expected
