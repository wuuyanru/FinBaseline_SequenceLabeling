import argparse
import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from bert4keras.tokenizers import Tokenizer
from tqdm import tqdm
import yaml
from bert4keras.backend import K

import models
import named_entity_recognizer
from data_management.data_loader import load_data


def evaluate(data, ner):
    """
    评测函数
    ner为命名实体识别器
    """
    X, Y, Z = 1e-10, 1e-10, 1e-10
    for d in tqdm(data, ncols=100):
        R = set(ner.recognize(d[0]))
        T = set([tuple(i) for i in d[1:]])
        X += len(R & T)
        Y += len(R)
        Z += len(T)
    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
    return f1, precision, recall


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', default="config.yaml")
    cli_args = parser.parse_args()
    with open(cli_args.config) as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)
    # config = yaml.load(open(cli_args.config, "rb"))
    if config is None:
        print("请提供配置文件！")

    # GPU相关设置
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 屏蔽通知信息和警告信息（INFO\WARNING）
    gpu_options = tf.GPUOptions(allow_growth=True)  # 设置按需分配显存
    global graph, sess
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    graph = tf.get_default_graph()
    KTF.set_session(sess)

    _, _, categories, _ = load_data(config["data_filename"], config["maxlen"])
    categories = list(sorted(categories))
    model_name = config["model_name"]
    model_module = getattr(models, model_name)
    _, predict_model, crf = model_module.creat_model(pretrain_config_path=config["pretrain_config_path"],
                                                       pretrain_checkpoint_path=config["pretrain_checkpoint_path"],
                                                       categories=categories,
                                                       pretrain_name=config["pretrain_name"],
                                                       dropout_rate=config["dropout_rate"],
                                                       hidden_size=config["hidden_size"],
                                                       crf_lr_multiplier=config["crf_lr_multiplier"])
    predict_model.load_weights(config["model_path"])
    tokenizer = Tokenizer(config["dict_path"], do_lower_case=True)
    maxlen = config["maxlen"]
    ner_module = getattr(named_entity_recognizer, model_name)
    if model_name.find("crf") != -1:
        ner = ner_module.NamedEntityRecognizer(tokenizer=tokenizer, model=predict_model, categories=categories, maxlen=maxlen,
                                               trans=K.eval(crf.trans), starts=[0], ends=[0])
    else:
        ner = ner_module.NamedEntityRecognizer(tokenizer=tokenizer, model=predict_model, categories=categories, maxlen=maxlen)
    test_data, _, _, _ = load_data(config["test_data"], config["maxlen"])
    f1, precision, recall = evaluate(test_data, ner)
    print("f1: %.5f, precision: %.5f, recall: %.5f" % (f1, precision, recall))
