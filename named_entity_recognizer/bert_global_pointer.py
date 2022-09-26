import numpy as np
from bert4keras.snippets import to_array


class NamedEntityRecognizer(object):
    """命名实体识别器
    """
    def __init__(self, tokenizer, model, categories, maxlen, threshold=0):
        self.tokenizer = tokenizer
        self.model = model
        self.categories = categories
        self.maxlen = maxlen
        self.threshold = threshold

    def recognize(self, text):
        tokens = self.tokenizer.tokenize(text, maxlen=self.maxlen)
        mapping = self.tokenizer.rematch(text, tokens)
        token_ids = self.tokenizer.tokens_to_ids(tokens)
        segment_ids = [0] * len(token_ids)
        token_ids, segment_ids = to_array([token_ids], [segment_ids])
        scores = self.model.predict([token_ids, segment_ids])[0]
        scores[:, [0, -1]] -= np.inf
        scores[:, :, [0, -1]] -= np.inf
        entities = []
        for i in range(1, scores.shape[1]):
            for j in range(i, scores.shape[2]):
                l = np.argmax(scores[:, i, j])
                if scores[l, i, j] > self.threshold:
                    entities.append((mapping[i][0], mapping[j][-1], self.categories[l]))

        return entities
