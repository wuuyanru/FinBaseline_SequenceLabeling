"""
创建bert+bi-lstm+crf的实体识别模型
"""
from bert4keras.layers import ConditionalRandomField
from bert4keras.models import build_transformer_model
from keras import Model
from keras.layers import Dropout, Bidirectional, LSTM, Dense


def creat_model(pretrain_config_path, pretrain_checkpoint_path, categories, pretrain_name='nezha',
                dropout_rate=0.3, hidden_size=128, crf_lr_multiplier=1000):
    model = build_transformer_model(pretrain_config_path, pretrain_checkpoint_path, pretrain_name)  # 加载预训练的语言模型
    output = Dropout(dropout_rate)(model.output)  # 添加dropout层
    output = Bidirectional(LSTM(hidden_size, return_sequences=True))(output)  # bi-lstm
    output = Dense(len(categories) * 2 + 1)(output)  # 全连接层,对于每类标签有B、I两个状态，所以*2，同时加上标签O
    crf = ConditionalRandomField(lr_multiplier=crf_lr_multiplier)  # crf层
    output = crf(output)
    model = Model(model.input, output)

    # 正常情况下只返回model，该处返回多个值是为了保持不同模型的creat_model输出结构统一
    return model, model, crf


def get_loss(crf, **kwargs):
    return crf.sparse_loss


def get_metrics(crf, **kwargs):
    return [crf.sparse_accuracy]


if __name__ == "__main__":
    model, _ = creat_model(r'../pre-train_models/nezha_base/bert_config.json',
                        r'../pre-train_models/nezha_base/model.ckpt-900000', {'ST', 'ST-2', 'CFC', 'ST-1'})
    model.summary()  # 打印模型结构
