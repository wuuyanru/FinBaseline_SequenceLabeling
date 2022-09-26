import numpy as np
from bert4keras.snippets import to_array


class NamedEntityRecognizer(object):
    """命名实体识别器
    """
    def __init__(self, tokenizer, model, categories, maxlen):
        self.tokenizer = tokenizer
        self.model = model
        self.categories = categories
        self.maxlen = maxlen

    def recognize(self, text):
        tokens = self.tokenizer.tokenize(text, maxlen=self.maxlen)
        mapping = self.tokenizer.rematch(text, tokens)
        token_ids = self.tokenizer.tokens_to_ids(tokens)
        segment_ids = [0] * len(token_ids)
        token_ids, segment_ids = to_array([token_ids], [segment_ids])
        preds = self.model.predict([token_ids, segment_ids])[0]
        labels = np.argmax(preds[:, :], axis=-1)
        entities, starting = [], False
        for i, label in enumerate(labels):
            if label > 0:
                if label % 2 == 1:
                    starting = True
                    entities.append([[i], self.categories[(label - 1) // 2]])
                elif starting:
                    entities[-1][0].append(i)
                else:
                    starting = False
            else:
                starting = False
        return [(mapping[w[0]][0], mapping[w[-1]][-1], l) for w, l in entities]
