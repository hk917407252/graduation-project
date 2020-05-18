from tqdm import tqdm
import numpy as np
import pandas as pd
import argparse

import torch
from torch import nn, optim

from transformers import BertTokenizer, AdamW, BertModel, BertPreTrainedModel, BertConfig
from sklearn.model_selection import StratifiedKFold

# 初始化一个命令行参数解析器
parser = argparse.ArgumentParser(allow_abbrev=False)

parser.add_argument("--seed", type=int, default=465, help="seed")           # 用于给交叉验证设定随机状态
parser.add_argument("--mdn", type=int, default=1)           # 该参数用于设定训练好的模型数据的数字命名

arg = parser.parse_args()

# 指定设备为GPU
device = torch.device('cuda')

# train = pd.read_csv('../data/Dataset/train2.csv', encoding='gb18030')
train = pd.read_csv('../data/Dataset/train.csv')

# 定义列表存储训练集数据
train_category = []
train_query1 = []
train_query2 = []
train_label = []

# 读取训练集数据
for i in range(len(train)):
    if train['label'][i] != train['label'][i]:
        continue
    train_category.append(train['category'][i])
    train_query1.append(train['query1'][i])
    train_query2.append(train['query2'][i])
    train_label.append(train['label'][i])

# 转数组
train_category = np.array(train_category)
train_query1 = np.array(train_query1)
train_query2 = np.array(train_query2)
train_label = np.array(train_label).astype(int)

# 模型路径
model_path = '../chinese_L-12_H-768_A-12/'
# 加载 bert_config 文件，bert_config.json是BERT的配置(超参数)
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
        # 定义线性层分类器,参数分别为:输入样本大小、输出样本大小
        self.classifier = nn.Linear(bert_config.hidden_size * 2, n_classes)

    # input_ids：标记化文本的数字id列表
    # input_masks：对于真实标记将设置为1，对于填充标记将设置为0
    #
    def forward(self, input_ids, input_masks, segment_ids):
        # 其中sequence_output中为每个单词的词向量，pooler_output则为全文的语义向量
        sequence_output, pooler_output, hidden_states = self.bert_model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_masks)
        # 将每个字的多个增强语义向量进行求平均
        seq_avg = torch.mean(sequence_output, dim=1)                # sequence_output为一个二维数组， dim=1,按列求平均值，得到的结果为 一维数组
        concat_out = torch.cat((seq_avg, pooler_output), dim=1)     # 按维数1拼接（横着拼），将seq_avg与pooler_output拼接在一起,即直接进行维度叠加
        logit = self.classifier(concat_out)     # 将解码组件产生的向量投射到logit向量，即全文语义信息的增强向量表示,每部分为一个二维向量
        return logit     # 返回对数几率向量


# 对抗学习，Project Gradient Descent 是一种迭代攻击，相比于普通的FGM 仅做一次迭代，PGD是做多次迭代，每次走一小步，每次迭代都会将扰动投射到规定范围内。
class PGD:
    # 构造函数，属性 model，emb_backup,grad_backup
    def __init__(self, model):
        self.model = model
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, epsilon=1., alpha=0.3, emb_name='word_embeddings', is_first_attack=False):
        # emb_name这个参数为模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)   # 求参数梯度的2范数
                if norm != 0 and not torch.isnan(norm):
                    r_at = alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, epsilon)

    # 恢复
    def restore(self, emb_name='word_embeddings'):
        # emb_name这个参数为模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    # 梯度备份
    def backup_grad(self):
        # 将model.named_parameters中的param的导数写入到导数备份列表
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.grad_backup[name] = param.grad.clone()

    # 复原梯度
    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.grad = self.grad_backup[name]


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


# 进行分层交叉验证，避免过拟合
skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=arg.seed)
# k折交叉切分，数据比为k：1分布,StratifiedKFold 分层采样交叉切分，确保训练集，测试集中各类别样本的比例与原始数据集中相同
# shuffle=True,每次进行划分时都重新进行洗牌
epoch = 5
PATH = '../model_data/bert_mul{}.pth'.format(arg.mdn)
for fold, (train_index, valid_index) in enumerate(skf.split(train_category, train_label)):
    print('\n\n------------fold:{}------------\n'.format(fold))
    # 读取划分出来的两份数据，一份训练集，一份验证集，数据比为4：1
    # 读取训练集，其中train_index为训练集数据在列表中的索引列表，保存了需要加入训练集的数据的索引信息
    c = train_category[train_index]
    q1 = train_query1[train_index]
    q2 = train_query2[train_index]
    y = train_label[train_index]

    # 读取验证集其中valid_index为验证集数据在列表中的索引列表，保存了需要加入验证集的数据的索引信息
    val_c = train_category[valid_index]
    val_q1 = train_query1[valid_index]
    val_q2 = train_query2[valid_index]
    val_y = train_label[valid_index]

    # 将数据转化为模型输入数据
    train_D = data_generator([c, q1, q2, y], batch_size=4, shuffle=True)
    val_D = data_generator([val_c, val_q1, val_q2, val_y], batch_size=4)

    # print(train_D.__len__())
    # print(next(train_D.__iter__()))

    # 模型加载到相应的设备中
    model = BertForClass().to(device)
    pgd = PGD(model)
    K = 3           # 设定攻击次数

    # nn.CrossEntropyLoss()是nn.logSoftmax()和nn.NLLLoss()的整合,可以直接使用它来替换网络中的这两个操作，
    # 计算交叉熵，有效解决二次损失函数带来的学习速率慢的问题。
    loss_fn = nn.CrossEntropyLoss()

    # 定义优化器
    # params:待优化参数的迭代或者是定义了参数组的字典
    # lr：learningRate学习率，此处设置为10^-5
    optimizer = AdamW(params=model.parameters(), lr=1e-5)

    best_acc = 0

    for e in range(epoch):
        print('\n------------epoch:{}------------'.format(e))

        # 将模型设置为训练模式
        model.train()
        acc = 0
        train_len = 0
        loss_num = 0

        # 定义进度条，用于显示运行进度，该部分为训练过程
        tq = tqdm(train_D)
        for input_ids, input_masks, segment_ids, labels in tq:
            input_ids = torch.tensor(input_ids).to(device)
            input_masks = torch.tensor(input_masks).to(device)
            segment_ids = torch.tensor(segment_ids).to(device)
            label_t = torch.tensor(labels, dtype=torch.long).to(device)

            # 将数据放入模型并得到输出向量
            y_pred = model.forward(input_ids, input_masks, segment_ids)
            # print("y_pred:", y_pred)

            # 计算输出向量与目标之间损失
            loss = loss_fn(y_pred, label_t)
            # 对loss求导，反向传播，得到正常的grad
            loss.backward()
            # 记录对抗训练的梯度
            pgd.backup_grad()

            # 对抗训练
            for t in range(K):
                pgd.attack(is_first_attack=(t == 0))  # 在embedding上添加对抗扰动, first attack时备份param.data
                if t != K - 1:      # 如果不是最后一次，则将梯度置零
                    model.zero_grad()
                else:               # 如果是最后一次，则恢复梯度
                    pgd.restore_grad()
                # 将数据给入模型并得到输出向量
                y_pred = model.forward(input_ids, input_masks, segment_ids)

                #计算损失
                loss_adv = loss_fn(y_pred, label_t)
                loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
            pgd.restore()  # 恢复embedding参数
            # 梯度下降，更新参数
            optimizer.step()
            model.zero_grad()

            # 返回axis轴方向最大值的索引
            # detach的方法，将y_pred参数从网络中隔离开，不参与参数更新。
            # .numpy()方法将 tensor转化为Numpy,注意cuda上面的变量类型只能是tensor，不能是其他,故先要将变量转到cpu上
            y_max = np.argmax(y_pred.detach().to("cpu").numpy(), axis=1)  # 获取向量中每一行最大值的索引值
            acc += sum(y_max == labels)
            loss_num += loss.item()     #
            train_len += len(labels)
            # 设置进度条显示参数
            tq.set_postfix(fold=fold, epoch=e, loss=loss_num / train_len, acc=acc / train_len)

        # 将模型设置为测试模式，以下部分为验证部分
        model.eval()
        y_p = []
        for input_ids, input_masks, segment_ids, labels in tqdm(val_D):
            input_ids = torch.tensor(input_ids).to(device)
            input_masks = torch.tensor(input_masks).to(device)
            segment_ids = torch.tensor(segment_ids).to(device)
            label_t = torch.tensor(labels, dtype=torch.long).to(device)

            # 将数据给入模型并得到输出向量
            y_pred = model.forward(input_ids, input_masks, segment_ids)

            # 返回axis轴方向最大值的索引
            # detach的方法，将y_pred参数从网络中隔离开，不参与参数更新。
            # .numpy()方法将 tensor转化为Numpy,注意cuda上面的变量类型只能是tensor，不能是其他,故先要将变量转到cpu上
            y_max = np.argmax(y_pred.detach().to("cpu").numpy(), axis=1)
            y_p.append(y_max[0])    # 记录预测结果

        acc = 0
        for i in range(len(y_p)):
            if val_y[i] == y_p[i]:
                acc += 1
        acc = acc / len(y_p)
        print("best_acc:{}  acc:{}\n".format(best_acc, acc))
        if acc >= best_acc:
            best_acc = acc
            #torch.save(model, PATH)     # 保存模型数据

    optimizer.zero_grad()
torch.save(model, PATH)  # 保存模型数据