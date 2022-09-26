import argparse
import os
import yaml
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from bert4keras.tokenizers import Tokenizer
from bert4keras.backend import K
import json
import time
import models
import named_entity_recognizer
from data_management.data_loader import load_data
from flask import Flask
from flask import request


def get_res_model(body, ner):
    with sess.as_default():
        with graph.as_default():
            pred_group_dict = {'pred_group': []}
            res_model = ner.recognize(body)
            pred_group_dict['pred_group'] = res_model
    return pred_group_dict


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

    if config["web_server"]:
        app = Flask(__name__)


        @app.route('/', methods=['POST'])
        def activate():
            msg = request.get_data()
            msg = msg.decode('utf-8')
            s = time.time()
            try:
                res_send = get_res_model(msg, ner)
            except Exception as e:
                res_send = {'pred_group': []}
            res = json.dumps(res_send, ensure_ascii=False)
            print('解析时间：', time.time() - s)
            return res


        app.run(host=config["web_host"], port=config["web_port"], threaded=True, debug=False)
        print("Starting server in python...")
        print("done!")

    else:
        with open(config["predict_data"], "r", encoding="utf-8") as f1:
            for line in f1:
                res_model = ner.recognize(line)
                with open(config["predict_path"], "a+", encoding="utf-8") as f2:
                    f2.write(str(res_model) + "\n")
        print("预测结束！")
