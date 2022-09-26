from bert4keras.snippets import DataGenerator, sequence_padding
import numpy as np


class DataGenerator(DataGenerator):
    """数据生成器
    """
    def __init__(self, tokenizer, maxlen, categories, data, rdrop_number=1, batch_size=1, buffer_size=None):
        self.tokenizer = tokenizer
        self.maxlen = maxlen
        self.categories = categories
        self.rdrop_number = rdrop_number  # r-drop参数
        super(DataGenerator, self).__init__(data, batch_size, buffer_size)

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, d in self.sample(random):
            tokens = self.tokenizer.tokenize(d[0], maxlen=self.maxlen)
            mapping = self.tokenizer.rematch(d[0], tokens)
            start_mapping = {j[0]: i for i, j in enumerate(mapping) if j}
            end_mapping = {j[-1]: i for i, j in enumerate(mapping) if j}
            token_ids = self.tokenizer.tokens_to_ids(tokens)
            segment_ids = [0] * len(token_ids)
            labels = np.zeros((len(token_ids), 2))
            for start, end, label in d[1:]:
                if start in start_mapping and end in end_mapping:
                    start = start_mapping[start]
                    end = end_mapping[end]
                    labels[start, 0] = self.categories.index(label) + 1  # 默认标签“O”索引为0，故其他标签索引+1
                    labels[end, 1] = self.categories.index(label) + 1

            # 借用r-drop思想，提高模型泛化能力
            for i in range(self.rdrop_number):
                batch_token_ids.append(token_ids)
                batch_segment_ids.append(segment_ids)
                batch_labels.append(labels)

            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids, batch_labels], None
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


if __name__ == "__main__":
    from bert4keras.tokenizers import Tokenizer
    from data_management.data_loader import load_data

    tokenizer = Tokenizer("../../pre-train_models/nezha_base/vocab.txt", do_lower_case=True)
    maxlen = 256
    data_all, length_sen, categories, categories_all = load_data("../../dataset/train_data.txt", 51)
    categories = list(sorted(categories))
    data_generator = DataGenerator(tokenizer, maxlen, categories, data_all)
    for item in data_generator.forfit():
        print(item)
        break
