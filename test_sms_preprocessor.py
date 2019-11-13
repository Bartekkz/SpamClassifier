from utils.sms_preprocessor import SmsPreprocessor


def test_pre_processor_create_key_word_map():
    pr = SmsPreprocessor(False)
    text = "Hello my dear friend. My brother told me."
    key_word_map = pr.create_key_word_dict(text) 
    expected = {"hello": 1, "my": 2, "dear": 3, "friend": 4, "brother": 5, "told": 6, "me": 7}
    assert key_word_map == expected 
    

def test_pre_processor_tokenizer():
    pr = SmsPreprocessor(False)
    text = "Hello my dear friend. My brother told me."
    tokenized = pr.tokenize(text)
    expected = [1, 2, 3, 4, 2, 5, 6, 7]
    assert tokenized == expected 
