# FinBaseline_SequenceLabeling
基于keras框架实现的序列标注基线系统。

## 项目结构
- **/dataset**
  - 存放数据集文件夹
- **/data_management**
  - 数据加载、生成操作
- **/pre-train_models**
  - 存放预训练模型文件夹，包括配置文件、模型以及词典
- **/model**
  - 本项目支持多种预训练模型，共5种神经网络结构
  - bi-lstm + crf
  - crf
  - softmax
  - global_pointer
  - span
  - 更多预训练模型、更多模型结构、更多训练trick，敬请期待
- **/config.yaml**
  - 本项目配置文件，所有参数均已注释
- **/train.py**
  - 训练模块
- **/evaluate.py**
  - 评估模块
- **/predict.py**
  - 预测模块
## 权重（目前已测试的权重）
- Google原版bert: https://github.com/google-research/bert
- brightmart版roberta: https://github.com/brightmart/roberta_zh
- 哈工大版roberta: https://github.com/ymcui/Chinese-BERT-wwm
- Google原版albert: https://github.com/google-research/ALBERT
- 华为的NEZHA: https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/NEZHA-TensorFlow

## Requirements
```
numpy==1.18.5
bert4keras==0.11.3
Keras==2.3.1
PyYAML==6.0
tensorflow-gpu==1.15.X
tqdm==4.62.3
h5py==2.10.0
```
可以通过命令`pip install -r requirements.txt`批量安装所需包。

## 数据集
- 数据集示例
```
原	B-ST-1
告	E-ST-1
：	O
商	B-ST-2
都	I-ST-2
县	I-ST-2
农	I-ST-2
村	I-ST-2
信	I-ST-2
用	I-ST-2
合	I-ST-2
作	I-ST-2
联	I-ST-2
社	E-ST-2
。	O
```
- 说明
  - 每条数据以空行隔开
  - 字符标注形式：字符 + ‘\t’ + 位置符号（B、I、E等） + ‘-’ + 实体类别（ST-1等），O表示该字符属于其他类别
  - 示例数据集不划分训练集、测试集，会在训练时进行split

## 训练
准备好数据集后，配置完config.yaml文件中的相关属性和超参数，可在终端采用以下命令进行：
```
python train.py --config <config.yaml> 或 python train.py
```
注意：
> 当config.yaml文件相对路径改变时，必须使用第一条命令指定路径；训练集、测试集划分比例可在train（）函数中自行调整

## 评估
准备好评估数据集后，配置完config.yaml文件中的model_path、test_data属性，可在终端采用以下命令进行：
```
python evaluate.py --config <config.yaml> 或 python evaluate.py
```
注意：
> 评估数据集标注形式与上面所述相同；程序最终输出f1, precision, recall指标值

## 预测
准备好预测数据后，配置完config.yaml文件中的model_path、predict_data属性，可在终端采用以下命令进行：
```
python predict.py --config <config.yaml> 或 python predict.py
```
注意：
> predict_data为单条数据，如果需要进行批量预测，可采用for循环进行

