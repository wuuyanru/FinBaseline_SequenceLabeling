"""
创建bert+softmax的实体识别模型
"""
from bert4keras.layers import ConditionalRandomField
from bert4keras.models import build_transformer_model
from keras import Model
from keras.layers import Dropout, Bidirectional, LSTM, Dense
from bert4keras.backend import keras, K


def creat_model(pretrain_config_path, pretrain_checkpoint_path, categories, pretrain_name='nezha',
                droput_rate=0.3, **kwargs):
    model = build_transformer_model(pretrain_config_path, pretrain_checkpoint_path, pretrain_name)  # 加载预训练的语言模型
    output = Dropout(droput_rate)(model.output)  # 添加dropout层
    output = Dense(len(categories) * 2 + 1, activation='softmax',
                   kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02))(output)
    model = Model(model.input, output)

    # 正常情况下只返回model，该处返回多个值是为了保持不同模型的creat_model输出结构统一
    return model, model, None


def _mask_crossentropy(y_true, y_pred):
    mask = K.cast(K.greater_equal(y_true,0),K.floatx())
    y_true = y_true - (1-mask)*y_true
    loss = K.sparse_categorical_crossentropy(y_true,y_pred)
    mask = K.reshape(mask,K.shape(loss))
    loss = K.sum(loss*mask,axis=1,keepdims=True)/K.sum(mask,axis=1,keepdims=True)
    return loss


def get_loss(**kwargs):
    return _mask_crossentropy


def get_metrics(**kwargs):
    return ['sparse_categorical_accuracy']


if __name__ == "__main__":
    function = get_loss()
    print(type(function))

