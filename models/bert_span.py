"""
创建bert+span的实体识别模型
"""
from bert4keras.layers import ConditionalRandomField, Loss
from bert4keras.models import build_transformer_model
from keras import Model, Input
from keras.layers import Dropout, Bidirectional, LSTM, Dense, Reshape, Average, Lambda
from bert4keras.backend import keras, K


def creat_model(pretrain_config_path, pretrain_checkpoint_path, categories, pretrain_name='nezha',
                droput_rate=0.3, hidden_size=128, **kwargs):
    input_labels = Input(shape=(None, 2), name='Labels')
    bert = build_transformer_model(pretrain_config_path, pretrain_checkpoint_path, pretrain_name)  # 加载预训练的语言模型
    dense1 = Dense(
        units=hidden_size,
        activation='relu',
        kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02)
    )
    dense2 = Dense(
        units=len(categories) * 2 + 2,
        kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02)
    )
    outputtmp = Dropout(droput_rate)(bert.output)
    outputtmp = dense1(outputtmp)
    outputtmp = dense2(outputtmp)
    outputtmp = Lambda(lambda x: x)(outputtmp)
    output = Reshape((-1, len(categories) + 1, 2))(outputtmp)
    predict_model = Model(bert.inputs, output)
    output = SpanLoss(0)([output, input_labels, bert.output])

    model = Model(bert.inputs + [input_labels], output)
    # 正常情况下只返回model，该处返回多个值是为了保持不同模型的creat_model输出结构统一
    return model, predict_model, None


class SpanLoss(Loss):
    def compute_loss(self, inputs, mask=None):
        subject_preds, subject_labels, _ = inputs
        if mask[2] is None:
            mask = 1.0
        else:
            mask = K.cast(mask[2], K.floatx())
        # sujuect部分loss
        start_loss = K.sparse_categorical_crossentropy(subject_labels[:, :, 0], subject_preds[:, :, :, 0], from_logits=True)
        end_loss = K.sparse_categorical_crossentropy(subject_labels[:, :, 1], subject_preds[:, :, :, 1], from_logits=True)
        subject_loss = start_loss+end_loss
        subject_loss = K.sum(subject_loss * mask) / K.sum(mask)
        return subject_loss


def get_loss(**kwargs):
    return None


def get_metrics(**kwargs):
    return None
