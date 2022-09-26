import numpy as np
from bert4keras.snippets import to_array


class NamedEntityRecognizer(object):
    """命名实体识别器
    """
    def __init__(self, tokenizer, model, categories, maxlen):
        self.tokenizer = tokenizer
        self.predict_model = model
        self.categories = categories
        self.maxlen = maxlen

    def recognize(self, text):
        tokens = self.tokenizer.tokenize(text, maxlen=self.maxlen)
        mapping = self.tokenizer.rematch(text, tokens)
        token_ids = self.tokenizer.tokens_to_ids(tokens)
        segment_ids = [0] * len(token_ids)
        token_ids, segment_ids = to_array([token_ids], [segment_ids])
        subject_preds = self.predict_model.predict([token_ids, segment_ids])[0]
        start = np.argmax(subject_preds[:, :, 0], axis=-1)
        end = np.argmax(subject_preds[:, :, 1], axis=-1)
        entities = []
        for i, l in enumerate(start):
            if i > 0 and i < len(start) - 1 and l != 0:
                for j in range(i, len(end) - 1):
                    if end[j] == l:
                        entities.append((mapping[i][0], mapping[j][-1], self.categories[l-1]))
                        break
        return entities
