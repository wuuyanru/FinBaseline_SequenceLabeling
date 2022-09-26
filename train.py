import os
import random
import numpy as np
from datetime import datetime
from bert4keras.backend import keras, K
from bert4keras.optimizers import extend_with_parameter_wise_lr, extend_with_exponential_moving_average, \
    extend_with_weight_decay, Adam
from bert4keras.tokenizers import Tokenizer
import models
import named_entity_recognizer
from data_management import data_generators
from data_management.data_loader import load_data
from evaluate import evaluate
from models.adversarial import adversarial_training


# 自定义回调函数
class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    def __init__(self, log_path, log_name, opt, crf, ner, valid_data, predict_model, best_path, model_name):
        self.best_val_f1 = 0
        self.opt = opt
        self.crf = crf
        self.ner = ner
        self.valid_data = valid_data
        self.predict_model = predict_model
        self.best_path = best_path
        self.model_name = model_name
        nt = datetime.now().strftime('%Y%m%d_%H_%M_%S')
        log_name = log_name + '_' + nt + '.txt'
        self.logfile = os.path.join(log_path, log_name)

    def on_epoch_end(self, epoch, logs=None):
        self.opt.apply_ema_weights()
        if self.model_name.find("crf") != -1:
            self.ner.trans = K.eval(self.crf.trans)

        f1, precision, recall = evaluate(self.valid_data, self.ner)
        # 保存最优
        if f1 >= self.best_val_f1:
            self.best_val_f1 = f1
            self.predict_model.save_weights(self.best_path)
        print(
            'valid:  f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n' %
            (f1, precision, recall, self.best_val_f1)
        )
        with open(self.logfile, 'a', encoding='utf8') as f:
            f.write('epoch: %d, valid:  f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n' %
                    (epoch, f1, precision, recall, self.best_val_f1))

        self.opt.reset_old_weights()


def train(data_filename, split_ratio, dict_path, maxlen, rdrop_number, batch_size, pretrain_config_path, pretrain_checkpoint_path,
          pretrain_name, model_name, dropout_rate, hidden_size, crf_lr_multiplier, learning_rate, exclude_from_weight_decay, paramwise_lr_schedule,
          ema_momentum, eplison, log_path, log_name, best_path, epochs):

    # 加载数据
    data_all, length_sen, categories, categories_all = load_data(data_filename, maxlen)
    categories = list(sorted(categories))
    print("最大长度：%d，实体类别：%s，所有实体类别（包含B、I、E等位置信息）：%s"
          % (length_sen, str(categories), str(categories_all)))

    # 划分训练集、验证集
    data_number = len(data_all)
    print('数据总量为：', len(data_all))
    random.shuffle(data_all)  # 打乱数据集
    train_part, valid_part = int(split_ratio.split(":")[0]), int(split_ratio.split(":")[1])
    all_parts = train_part + valid_part
    train_data = data_all[data_number // all_parts * valid_part:]
    valid_data = data_all[:data_number // all_parts * valid_part]
    # train_data = data_all[:300]
    # valid_data = data_all[300:330]

    tokenizer = Tokenizer(dict_path, do_lower_case=True)  # 建立分词器

    # 训练数据生成器
    datagenerator_module = getattr(data_generators, model_name)
    train_generator = datagenerator_module.DataGenerator(tokenizer, maxlen, categories, train_data, rdrop_number, batch_size)

    # 加载模型
    model_module = getattr(models, model_name)
    model, predict_model, crf = model_module.creat_model(pretrain_config_path=pretrain_config_path,
                                          pretrain_checkpoint_path=pretrain_checkpoint_path, categories=categories,
                                          pretrain_name=pretrain_name, dropout_rate=dropout_rate,
                                          hidden_size=hidden_size, crf_lr_multiplier=crf_lr_multiplier)
    model.summary()
    if predict_model is not model:
        print("评估、预测时模型结构：")
        predict_model.summary()

    # 优化器
    AdamW = extend_with_parameter_wise_lr(extend_with_weight_decay(Adam))
    AdamEMA = extend_with_exponential_moving_average(AdamW, name='AdamEMA')
    opt = AdamEMA(learning_rate=learning_rate, exclude_from_weight_decay=exclude_from_weight_decay,
                  paramwise_lr_schedule=paramwise_lr_schedule, ema_momentum=ema_momentum)

    # 模型编译
    loss = model_module.get_loss(crf=crf)
    metrics = model_module.get_metrics(crf=crf)
    model.compile(
        loss=loss,
        optimizer=opt,
        metrics=metrics)

    # 是否加入对抗训练
    if eplison > 0:
        adversarial_training(model, 'Embedding-Token', eplison)

    # 回调函数
    ner_module = getattr(named_entity_recognizer, model_name)
    if model_name.find("crf") != -1:
        ner = ner_module.NamedEntityRecognizer(tokenizer=tokenizer, model=predict_model, categories=categories, maxlen=maxlen,
                                               trans=K.eval(crf.trans), starts=[0], ends=[0])
    else:
        ner = ner_module.NamedEntityRecognizer(tokenizer=tokenizer, model=predict_model, categories=categories, maxlen=maxlen)
    evaluator = Evaluator(log_path, log_name, opt, crf, ner, valid_data, predict_model, best_path, model_name)

    # 训练
    model.fit(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        callbacks=[evaluator])
    print("训练完成!")


if __name__ == "__main__":
    import argparse
    import yaml
    import tensorflow as tf
    import keras.backend.tensorflow_backend as KTF

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

    # 设置随机数种子
    def set_seed(seed=42):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        tf.set_random_seed(seed)
    seed = config["random_seed"]
    set_seed(seed)

    train(data_filename=config["data_filename"], split_ratio=config["split_ratio"], dict_path=config["dict_path"],
          maxlen=config["maxlen"],
          rdrop_number=config["rdrop_number"], batch_size=config["batch_size"],
          pretrain_config_path=config["pretrain_config_path"],
          pretrain_checkpoint_path=config["pretrain_checkpoint_path"], pretrain_name=config["pretrain_name"],
          model_name=config["model_name"], dropout_rate=config["dropout_rate"], hidden_size=config["hidden_size"],
          crf_lr_multiplier=config["crf_lr_multiplier"], learning_rate=float(config["learning_rate"]),
          exclude_from_weight_decay=config["exclude_from_weight_decay"],
          paramwise_lr_schedule=config["paramwise_lr_schedule"], ema_momentum=config["ema_momentum"],
          eplison=config["eplison"], log_path=config["log_path"], log_name=config["log_name"],
          best_path=config["best_path"], epochs=config["epochs"])





