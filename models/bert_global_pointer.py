"""
创建bert+global_pointer的实体识别模型
"""
from bert4keras.layers import GlobalPointer
from bert4keras.models import build_transformer_model
from bert4keras.backend import K, multilabel_categorical_crossentropy
from keras import Model


def creat_model(pretrain_config_path, pretrain_checkpoint_path, categories, pretrain_name='nezha',
                hidden_size=128, **kwargs):
    model = build_transformer_model(pretrain_config_path, pretrain_checkpoint_path, pretrain_name)  # 加载预训练的语言模型
    output = GlobalPointer(len(categories), hidden_size)(model.output)
    model = Model(model.input, output)

    # 正常情况下只返回model，该处返回多个值是为了保持不同模型的creat_model输出结构统一
    return model, model, None


def _global_pointer_crossentropy(y_true, y_pred):
    """给GlobalPointer设计的交叉熵
    """
    bh = K.prod(K.shape(y_pred)[:2])
    y_true = K.reshape(y_true, (bh, -1))
    y_pred = K.reshape(y_pred, (bh, -1))
    return K.mean(multilabel_categorical_crossentropy(y_true, y_pred))


def _global_pointer_f1_score(y_true, y_pred):
    """给GlobalPointer设计的F1
    """
    y_pred = K.cast(K.greater(y_pred, 0), K.floatx())
    return 2 * K.sum(y_true * y_pred) / K.sum(y_true + y_pred)


def get_loss(**kwargs):
    return _global_pointer_crossentropy


def get_metrics(**kwargs):
    return [_global_pointer_f1_score]
