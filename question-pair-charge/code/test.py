from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch import nn
from transformers import BertTokenizer, BertModel, BertConfig
import datetime

# 设置设备为GPU
device = torch.device('cuda')

# 读取测试集数据，并设置编码格式
# test = pd.read_csv('../data/Dataset/test.csv')
test = pd.read_csv('../data/Dataset/test.csv')

# 模型路径
model_path = '../chinese_L-12_H-768_A-12/'

# 将训练集中的数据存入三个列表
test_category = test['category'].values
test_query1 = test['query1'].values
test_query2 = test['query2'].values
test_rea_label = test['rea_label'].values

# 加载 bert_config 文件
bert_config = BertConfig.from_pretrained(model_path + 'bert_config.json', output_hidden_states=True)
# 加载vocab.txt 文件
tokenizer = BertTokenizer.from_pretrained(model_path + 'vocab.txt', config=bert_config)


# 构建bert模型
class BertForClass(nn.Module):
    # 构造函数，实现层的参数定义
    def __init__(self, n_classes=2):
        super(BertForClass, self).__init__()
        self.model_name = 'BertForClass'
        # self.bert_model = BertModel.from_pretrained(model_path, config=bert_config, from_tf= True)

        # 加载模型，配置为bert_config
        self.bert_model = BertModel.from_pretrained(model_path, config=bert_config)
        # 定义线性分类器
        self.classifier = nn.Linear(bert_config.hidden_size * 2, n_classes)

    # input_ids：标记化文本的数字id列表
    # input_masks：对于真实标记将设置为1，对于填充标记将设置为0
    #
    def forward(self, input_ids, input_masks, segment_ids):
        # 其中sequence_output中为每个单词的词向量，pooler_output则为全文的语义向量
        sequence_output, pooler_output, hidden_states = self.bert_model(input_ids=input_ids, token_type_ids=segment_ids,
                                                                        attention_mask=input_masks)
        # 将每个字的多个增强语义向量进行求平均
        seq_avg = torch.mean(sequence_output, dim=1)  # sequence_output为一个二维数组， dim=1,按列求平均值，得到的结果为 一维数组
        concat_out = torch.cat((seq_avg, pooler_output), dim=1)  # 按维数1拼接（横着拼），将seq_avg与pooler_output拼接在一起
        logit = self.classifier(concat_out)  # 将解码组件产生的向量投射到logit向量，即各个字融合了全文语义信息后的增强向量表示
        return logit  # 返回对数几率向量


# 定义数据生成器类
class data_generator:
    def __init__(self, data, batch_size=8, max_length=64, shuffle=False):
        self.data = data
        self.batch_size = batch_size
        self.max_length = max_length
        self.shuffle = shuffle
        self.steps = len(self.data[0]) // self.batch_size   # 返回不大于结果的一个最大的整数
        if len(self.data[0]) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    # 当函数被调用时，函数体中的代码是不会运行的，该函数仅仅是返回一个迭代器对象
    def __iter__(self):
        # data传入为一个列表，有四个元素，分别赋值给c,q1,q2,y
        c, q1, q2, y = self.data
        # 根据数据个数生成一个自然数列表
        idxs = list(range(len(self.data[0])))
        if self.shuffle:
            # 打乱原列表顺序，获取随机列表
            np.random.shuffle(idxs)
        # 定义四个列表
        input_ids, input_masks, segment_ids, labels = [], [], [], []
        # print("len:", len(idxs))

        # 相当于随机取出一条记录,其中，index为索引值，i为idxs[index] 的值
        # enumerate多用于在for循环中得到计数，利用它可以同时获得索引和值，即需要index和value值的时候可以使用,第一个值为index，第二个值为value
        for index, i in enumerate(idxs):
            text1 = q1[i]
            text2 = q2[i]
            # 获取模型的输入,最大长度不超过max_length, encode方法可以一步到位地生成对应模型的输入
            input_id = tokenizer.encode(text1, text2, max_length=self.max_length)
            # 初始化input_mask
            input_mask = [1] * len(input_id)

            if len(text1) + 2 <= len(input_id):
                segment_id = [0] * (len(text1) + 2) + [1] * (len(input_id) - 2 - len(text1))
            else:
                segment_id = [0] * len(input_id)

            padding_length = self.max_length - len(input_id)        # 计算input_id与最大长度的差值
            # 将空值都补全为0
            input_id += ([0] * padding_length)
            input_mask += ([0] * padding_length)
            segment_id += ([0] * padding_length)

            # 分别加入到对应列表中
            input_ids.append(input_id)
            input_masks.append(input_mask)
            segment_ids.append(segment_id)
            labels.append(y[i])

            if len(input_ids) == self.batch_size or i == idxs[-1]:
                # yield是python中定义为生成器函数，其本质是封装了  __iter__和__next__方法   的迭代器；
                # 与return返回的区别：return只能返回一次值，函数就终止了，而yield能多次返回值，
                # 每次返回都会将函数暂停，下一次next会从上一次暂停的位置继续执行；
                yield input_ids, input_masks, segment_ids, labels

                # 将列表重置
                input_ids, input_masks, segment_ids, labels = [], [], [], []


# 归一化指数函数, 是逻辑函数的一种推广。它能将一个含任意实数的K维向量 f(z)“压缩”到另一个K维实向量 σ(z)中，使得每一个元素的范围都在 (0,1)之间，并且所有元素的和为1。
# 函数的意义：对向量进行归一化，凸显其中最大的值并抑制远低于最大值的其他分量。
def softmax(x):
    x_row_max = x.max(axis=1)  # 按照行求取最大值,得到一个一维数组
    x_row_max = x_row_max.reshape(list(x.shape)[:-1] + [1])     # 将数组行列翻转，相当于矩阵行列交换
    # print(x_row_max)
    x = x - x_row_max           # 将二维向量中的较大值置0
    # print(x)
    x_exp = np.exp(x)           # exp以自然常数e为底的指数函数，将0的部分置1，负数部分则化为较小的小数
    x_exp_row_sum = x_exp.sum(axis=1).reshape(list(x.shape)[:-1] + [1])    # 将矩阵按行相加，且得到n行1列的矩阵
    # print(x_exp_row_sum)
    softmax = x_exp / x_exp_row_sum     # 除以膜长进行标准归一化
    # print(softmax)
    return softmax


result = np.zeros((len(test), 2), dtype=np.float32)     # 初始化一个全0的二维向量列表，用于保存预测结果

# print("result", result)

test_c = test_category[range(len(test_category))]
test_q1 = test_query1[range(len(test_category))]
test_q2 = test_query2[range(len(test_category))]
test_i = test_category[range(len(test_category))]

test_D = data_generator([test_c, test_q1, test_q2, test_i], batch_size=4)

PATH = '../model_data/bert_1.pth'

# 将模型加载到设备上
model = torch.load(PATH).to(device)
# 将模型设置为测试模式
model.eval()
y_p = []
for input_ids, input_masks, segment_ids, labels in tqdm(test_D):
    input_ids = torch.tensor(input_ids).to(device)
    input_masks = torch.tensor(input_masks).to(device)
    segment_ids = torch.tensor(segment_ids).to(device)

    y_pred = model.forward(input_ids, input_masks, segment_ids)
    # print("y_pred", y_pred)
    y_p += y_pred.detach().to("cpu").tolist()

y_p = np.array(y_p)     # 将列表转为数组，方便进行归一化
# print("sy_p", y_p.shape)

y_p = softmax(y_p)      # 对得到的预测结果向量做归一化处理
# print("gy_p", y_p)

# 将预测结果
for i in range(len(y_p)):
    result[i][np.argmax(y_p[i])] += 1

# print("result", result)

test['pre_label'] = np.argmax(result, axis=1)     # 获取每一行最大值的索引值
test_pre_label = test['pre_label'].values
test[['id', 'pre_label', 'rea_label']].to_csv('../prediction_result/result'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+'.csv', index=False)

count_right = 0
count_wrong = 0
count_t_to_t = 0
count_t_to_f = 0
count_f_to_f = 0
count_f_to_t = 0

# 计算各类预测数目，用于计算准确率，精准率，召回率以及F测度
for i in range(len(test)):
    if test_rea_label[i] == test_pre_label[i]:
        count_right += 1
        if test_rea_label[i] == 1:
            count_t_to_t += 1
        else:
            count_f_to_f += 1
    else:
        count_wrong += 1
        if test_rea_label[i] == 0:
            count_f_to_t += 1
        else:
            count_t_to_f += 1
Accuracy = count_right/len(test)                        # 计算准确率
Precision = count_t_to_t/(count_t_to_t+count_f_to_t)    # 计算精确率，即在预测为正例中的预测正确的比率
Recall = count_t_to_t/(count_t_to_t+count_t_to_f)       # 计算在所有实际正例中，预测为正例的比率
F1_Measure = 2*Precision*Recall/(Precision + Recall)     # F1-Measure是Precision和Recall加权调和平均

# print("pre_right:", count_right)
# print("pre_wrong:", count_wrong)
# print("count_t_to_t:", count_t_to_t)
# print("count_f_to_f:", count_f_to_f)
# print("count_f_to_t:", count_f_to_t)
# print("count_t_to_f:", count_t_to_f)
print("Accuracy:", Accuracy)
print("Precision:", Precision)
print("Recall:", Recall)
print("F1_Measure", F1_Measure)