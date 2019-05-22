import collections
import d2lzh as d2l
import math
from mxnet import autograd, gluon, nd
from mxnet.gluon import data as gdata, loss as gloss, nn
import random
import sys
import time
import zipfile

#处理数据
with open('./data/ptb/ptb.train.txt', 'r') as f:
    lines = f.readlines()
    # st是sentence的缩写
    raw_dataset = [st.split() for st in lines]

# print ('# sentences: %d' % len(raw_dataset))   #42068

#对于数据集的前3个句子，打印每个句子的词数和前5个词.句尾符为“<eos>”，生僻词全用“<unk>”表示,，数字则被替换成了“N”。
# for st in raw_dataset[:3]:
#     print('# tokens:', len(st), st[:5])
# # tokens: 24 ['aer', 'banknote', 'berlitz', 'calloway', 'centrust']
# # tokens: 15 ['pierre', '<unk>', 'N', 'years', 'old']
# # tokens: 11 ['mr.', '<unk>', 'is', 'chairman', 'of']

#建立词语索引
# tk是token的缩写
counter = collections.Counter([tk for st in raw_dataset for tk in st])
counter = dict(filter(lambda x: x[1] >= 5, counter.items()))
#然后将词映射到整数索引
idx_to_token = [tk for tk, _ in counter.items()]
token_to_idx = {tk: idx for idx, tk in enumerate(idx_to_token)}
dataset = [[token_to_idx[tk] for tk in st if tk in token_to_idx]
           for st in raw_dataset]
num_tokens = sum([len(st) for st in dataset])
# print('# tokens: %d' % num_tokens)
# tokens: 887100

def discard(idx):
    return random.uniform(0, 1) < 1 - math.sqrt(
        1e-4 / counter[idx_to_token[idx]] * num_tokens)

subsampled_dataset = [[tk for tk in st if not discard(tk)] for st in dataset]
#print('# tokens: %d' % sum([len(st) for st in subsampled_dataset]))
# tokens: 375504 二次采样后去掉了一半的词

def compare_counts(token):
    return '# %s: before=%d, after=%d' % (token, sum(
        [st.count(token_to_idx[token]) for st in dataset]), sum(
        [st.count(token_to_idx[token]) for st in subsampled_dataset]))

#print(compare_counts('the'))
# the: before=50770, after=2188 高频词“the”在二次采样前后的采样率不足1/20
#print(compare_counts('join'))
# join: before=45, after=45 低频词不变

##提取中心词和背景词
def get_centers_and_contexts(dataset, max_window_size):
    centers, contexts = [], []
    for st in dataset:
        if len(st) < 2:  # 每个句子至少要有2个词才可能组成一对“中心词-背景词”
            continue
        centers += st
        for center_i in range(len(st)):
            window_size = random.randint(1, max_window_size)
            indices = list(range(max(0, center_i - window_size),
                                 min(len(st), center_i + 1 + window_size)))
            indices.remove(center_i)  # 将中心词排除在背景词之外
            contexts.append([st[idx] for idx in indices])
    return centers, contexts

#下面我们创建一个人工数据集，其中含有词数分别为7和3的两个句子。
## 设最大背景窗口为2，打印所有中心词和它们的背景词。
# tiny_dataset = [list(range(7)), list(range(7, 10))]
# print('dataset', tiny_dataset)
# for center, context in zip(*get_centers_and_contexts(tiny_dataset, 2)):
#     print('center', center, 'has contexts', context)
# dataset [[0, 1, 2, 3, 4, 5, 6], [7, 8, 9]]
# center 0 has contexts [1]
# center 1 has contexts [0, 2]
# center 2 has contexts [0, 1, 3, 4]
# center 3 has contexts [2, 4]
# center 4 has contexts [2, 3, 5, 6]
# center 5 has contexts [3, 4, 6]
# center 6 has contexts [4, 5]
# center 7 has contexts [8, 9]
# center 8 has contexts [7, 9]
# center 9 has contexts [7, 8]

#实验中，我们设最大背景窗口大小为5
all_centers, all_contexts = get_centers_and_contexts(subsampled_dataset, 5)

##负采样 随机采样K个噪声词（实验中设K=5）
def get_negatives(all_contexts, sampling_weights, K):
    all_negatives, neg_candidates, i = [], [], 0
    population = list(range(len(sampling_weights)))
    for contexts in all_contexts:
        negatives = []
        while len(negatives) < len(contexts) * K:
            if i == len(neg_candidates):
                # 根据每个词的权重（sampling_weights）随机生成k个词的索引作为噪声词。
                # 为了高效计算，可以将k设得稍大一点
                i, neg_candidates = 0, random.choices(
                    population, sampling_weights, k=int(1e5))
            neg, i = neg_candidates[i], i + 1
            # 噪声词不能是背景词
            if neg not in set(contexts):
                negatives.append(neg)
        all_negatives.append(negatives)
    return all_negatives

sampling_weights = [counter[w]**0.75 for w in idx_to_token]
all_negatives = get_negatives(all_contexts, sampling_weights, 5)

##读取数据
def batchify(data):
    max_len = max(len(c) + len(n) for _, c, n in data)
    centers, contexts_negatives, masks, labels = [], [], [], []
    for center, context, negative in data:
        cur_len = len(context) + len(negative)
        centers += [center]
        contexts_negatives += [context + negative + [0] * (max_len - cur_len)]
        masks += [[1] * cur_len + [0] * (max_len - cur_len)]
        labels += [[1] * len(context) + [0] * (max_len - len(context))]
    return (nd.array(centers).reshape((-1, 1)), nd.array(contexts_negatives),
            nd.array(masks), nd.array(labels))

#打印读取的第一个批量中各个变量的形状
batch_size = 512
num_workers = 0 if sys.platform.startswith('win32') else 4
dataset = gdata.ArrayDataset(all_centers, all_contexts, all_negatives)
data_iter = gdata.DataLoader(dataset, batch_size, shuffle=True,
                             batchify_fn=batchify, num_workers=num_workers)
# for batch in data_iter:
#     for name, data in zip(['centers', 'contexts_negatives', 'masks',
#                            'labels'], batch):
#         print(name, 'shape:', data.shape)
#     break
# centers shape: (512, 1)
# contexts_negatives shape: (512, 60)
# masks shape: (512, 60)
# labels shape: (512, 60)

##跳字模型

#获取词嵌入的层称为嵌入层，在Gluon中可以通过创建nn.Embedding实例得到。
# 嵌入层的权重是一个矩阵，其行数为词典大小（input_dim），列数为每个词向量的维度（output_dim）。
# 我们设词典大小为20，词向量的维度为4。

embed = nn.Embedding(input_dim=20, output_dim=4)
embed.initialize()
embed.weight
#嵌入层的输入为词的索引。输入一个词的索引 i ，嵌入层返回权重矩阵的第 i 行作为它的词向量。
# 下面我们将形状为(2, 3)的索引输入进嵌入层，由于词向量的维度为4，我们得到形状为(2, 3, 4)的词向量。
#由于我们的输入向量是用one-hot来表示的，与权重矩阵W1相乘就相当于简单的选择W1中的一行
# x = nd.array([[1, 2, 3], [4, 5, 6]])
# print(embed(x))
# [[[ 0.01438687  0.05011239  0.00628365  0.04861524]
#   [-0.01068833  0.01729892  0.02042518 -0.01618656]
#   [-0.00873779 -0.02834515  0.05484822 -0.06206018]]
#
#  [[ 0.06491279 -0.03182812 -0.01631819 -0.00312688]
#   [ 0.0408415   0.04370362  0.00404529 -0.0028032 ]
#   [ 0.00952624 -0.01501013  0.05958354  0.04705103]]]
# <NDArray 2x3x4 @cpu(0)>

#小批量乘法
#假设第一个小批量中包含 n 个形状为 a×b 的矩阵 X1,…,Xn ，
#第二个小批量中包含 n 个形状为 b×c 的矩阵 Y1,…,Yn 。
#这两个小批量的矩阵乘法输出为 n 个形状为 a×c 的矩阵 X1Y1,…,XnYn 。
# X = nd.ones((2, 1, 4))
# Y = nd.ones((2, 4, 6))
# print(nd.batch_dot(X, Y).shape)
#(2, 1, 6)

# 跳字模型前向计算
#其中center变量的形状为(批量大小, 1)，而contexts_and_negatives变量的形状为(批量大小, max_len)。
# 这两个变量先通过词嵌入层分别由词索引变换为词向量，再通过小批量乘法得到形状为(批量大小, 1, max_len)的输出。
# 输出中的每个元素是中心词向量与背景词向量或噪声词向量的内积
def skip_gram(center, contexts_and_negatives, embed_v, embed_u):
    v = embed_v(center)
    u = embed_u(contexts_and_negatives)
    pred = nd.batch_dot(v, u.swapaxes(1, 2))
    return pred

##训练模型

#首先定义损失函数
#二元交叉熵损失函数
loss = gloss.SigmoidBinaryCrossEntropyLoss()

#通过掩码变量指定小批量中参与损失函数计算的部分预测值和标签
# pred = nd.array([[1.5, 0.3, -1, 2], [1.1, -0.6, 2.2, 0.4]])
# # 标签变量label中的1和0分别代表背景词和噪声词
# label = nd.array([[1, 0, 0, 0], [1, 1, 0, 0]])
# mask = nd.array([[1, 1, 1, 1], [1, 1, 1, 0]])  # 掩码变量
# print(loss(pred, label, mask) * mask.shape[1] / mask.sum(axis=1))
# [0.8739896 1.2099689]
# <NDArray 2 @cpu(0)>

#初始化模型参数
#分别构造中心词和背景词的嵌入层，并将超参数词向量维度embed_size设置成100
#值太小会导致词映射因为冲突而影响结果，值太大则会耗内存并使算法计算变慢，一般值取为100到200之间
embed_size = 100
net = nn.Sequential()
net.add(nn.Embedding(input_dim=len(idx_to_token), output_dim=embed_size),
        nn.Embedding(input_dim=len(idx_to_token), output_dim=embed_size))
#定义训练函数
def train(net, lr, num_epochs):
    ctx = d2l.try_gpu()
    net.initialize(ctx=ctx, force_reinit=True)
    trainer = gluon.Trainer(net.collect_params(), 'adam',
                            {'learning_rate': lr})
    for epoch in range(num_epochs):
        start, l_sum, n = time.time(), 0.0, 0
        for batch in data_iter:
            center, context_negative, mask, label = [
                data.as_in_context(ctx) for data in batch]
            with autograd.record():
                pred = skip_gram(center, context_negative, net[0], net[1])
                # 使用掩码变量mask来避免填充项对损失函数计算的影响
                l = (loss(pred.reshape(label.shape), label, mask) *
                     mask.shape[1] / mask.sum(axis=1))
            l.backward()
            trainer.step(batch_size)
            l_sum += l.sum().asscalar()
            n += l.size
        print('epoch %d, loss %.2f, time %.2fs'
              % (epoch + 1, l_sum / n, time.time() - start))

# train(net, 0.005, 5)

##应用词嵌入模型

#根据两个词向量的余弦相似度表示词与词之间在语义上的相似度
def get_similar_tokens(query_token, k, embed):
    embed.initialize()
    W = embed.weight.data()
    x = W[token_to_idx[query_token]]
    # 添加的1e-9是为了数值稳定性
    cos = nd.dot(W, x) / (nd.sum(W * W, axis=1) * nd.sum(x * x) + 1e-9).sqrt()
    topk = nd.topk(cos, k=k+1, ret_typ='indices').asnumpy().astype('int32')
    for i in topk[1:]:  # 除去输入词
        print('cosine sim=%.3f: %s' % (cos[i].asscalar(), (idx_to_token[i])))

# get_similar_tokens('chip', 3, net[0])

# cosine sim=0.391: heating
# csoine sim=0.390: eurocom
# cosine sim=0.386: financiere

#输出词向量 未解决
# def get_word_vector(query_token):
#     # embed.initialize()
#     W = embed.weight.data()
#     x = W[token_to_idx[query_token]]
#     print('chip vector=%.10f' %x)
#
# get_word_vector('chip')
