5.1 合并与分割

5.1.1 合并
    -----将多个张量在某个维度上合并为一个张量
        |--拼接(Concatenante)  不产生新维度
        |--tf.concat(tensors, axis)  tensors 保存了所有需要合并的张量 List,axis 参数指定需要合并的维度索引
        |---------------------
        |--堆叠(Stack)  创建新维度
        |--tf.stack(tensors, axis)  tensors 保存了所有需要合并的张量 List, axis 指定新维度插入的位置与 tf.expand_dims 的一致


5.1.2 分割
    -----合并操作的逆过程就是分割,将一个张量分拆为多个张量
        |--tf.split(x, num_or_size_splits, axis)  x为待分割张量,num_or_size_splits 为切割方案,当num_or_size_splits为单个数值时,如10,表示等长切割为10份;当num_or_size_splits为List时,List 的每个元素表示每份的长度, axis 为指定分割的维度索引号


5.2 数据统计

5.2.1 向量范数(Vector Norm)
    -----表征向量"长度"的一种度量方法,它可以推广到张量上.在神经网络中,常用来表示张量的权值大小,梯度大小等
        |-----L1范数
        |   |--定义为向量𝒙的所有元素绝对值之和
        |
        |-----L2范数
        |   |--定义为向量𝒙的所有元素的平方和,再开根号
        |
        |----- ∞ −范数
        |   |--定义为向量𝒙的所有元素绝对值的最大值
        |
        |-----tf.norm(x, ord= )  ord 指定为 1,2 时计算 L1,L2 范数,指定np.inf时计算∞ −范数


5.2.2 最值,均值,和
    -----求解张量在某个维度上的最大,最小,均值,和,也可以求全局最大,最小,均值,和信息
        |--tf.reduce_max(x, axis= )  axis维度上的最大值
        |--tf.reduce_min(x, axis= )  axis维度上的最小值
        |--tf.reduce_mean(x, axis= )  axis维度上的均值
        |--tf.reduce_sum(x, axis= )  axis维度上的和
        |--当不指定 axis 参数时,tf.reduce_*函数会求解出全局元素的最大,最小,均值,和
        |--tf.argmax(x, axis)  tf.argmin(x, axis)  可以求解在 axis 轴上,x 的最大值,最小值所在的索引号


5.3 张量比较
    -----为了计算分类任务的准确率等指标,一般需要将预测结果和真实标签比较,统计比较结果中正确的数量来计算准确率
        |--tf.math.greater(a,b)             𝑎 > 𝑏
        |--tf.math.less(a,b)                𝑎 < b
        |--tf.math.greater_equal(a,b)       𝑎 ≥ 𝑏
        |--tf.math.less_equal(a,b)          𝑎 ≤ 𝑏
        |--tf.math.not_equal(a,b)           𝑎 ≠ b
        |--tf.math.is_nan(a)                𝑎 = nan


5.4 填充与复制

5.4.1 填充(Padding)
    -----对于图片数据的高和宽,序列信号的长度,维度长度可能各不相同.为了方便网络的并行计算,需要将不同长度的数据扩张为相同长度,在需要补充长度的数据开始或结束处填充足够数量的特定数值,这些特定数值一般代表了无效意义,例如0
        |--tf.pad(x, paddings)  paddings是包含了多个[Left Padding,Right Padding]的嵌套方案 List,如[[0,0],[2,1],[1,2]]表示第一个维度不填充,第二个维度左边(起始处)填充两个单元,右边(结束处)填充一个单元,第三个维度左边填充一个单元,右边填充两个单元.
        |--keras.preprocessing.sequence.pad_sequences 函数可以快速完成句子的填充和截断工作


5.4.2 复制
    -----见ch04  readme   维度变换

5.5 数据限幅
    -----tf.maximum(x, a)  实现数据的下限幅,即𝑥 ∈ [𝑎, +∞)
      |--tf.minimum(x, a)  实现数据的上限幅,即𝑥 ∈ (−∞,𝑎]


5.6 高级操作

5.6.1 tf.gather
    -----实现根据索引号收集数据的目的
        |--tf.gather(x, index, axis= )  x为目标张量,index为需要收集数据的索引号,axis为维度


5.6.2 tf.gather_nd
    -----根据多维坐标收集数据
        |--tf.gather_nd(x, 多维坐标)


5.6.3 tf.boolean_mask
    -----通过给定掩码(Mask)的方式进行采样
        |--tf.boolean_mask(x, mask, axis)  axis 轴上根据mask(可多维) 方案进行采样


5.6.4 tf.where(cond, a, b)
    -----tf.where(cond, a, b)操作可以根据 cond 条件的真假从参数𝑨或𝑩中读取数据,𝑜𝑖 = 𝑎𝑖  cond𝑖为True , 𝑏𝑖  cond𝑖为False
        |--当参数 a=b=None 时,即 a 和 b 参数不指定,tf.where 会返回 cond 张量中所有 True 的元素的索引坐标


5.6.5 scatter_nd
    -----通过 tf.scatter_nd(indices, updates, shape)函数可以高效地刷新张量的部分数据
        |--只能在全 0 的白板张量上面执行刷新操作


5.6.6 meshgrid
    -----通过 tf.meshgrid 函数可以方便地生成二维网格的采样点坐标,方便可视化等应用场合


5.7 经典数据集加载
    -----keras.datasets 模块提供了常用经典数据集的自动下载,管理,加载与转换功能,并且提供了 tf.data.Dataset 数据集对象,方便实现多线程(Multi-threading),预处理(Preprocessing),随机打散(Shuffle)和批训练(Training on Batch)等常用数据集的功能
        |--Boston Housing,波士顿房价趋势数据集,用于回归模型训练与测试
        |--CIFAR10/100,真实图片数据集,用于图片分类任务
        |--MNIST/Fashion_MNIST,手写数字图片数据集,用于图片分类任务
        |--IMDB,情感分类任务数据集,用于文本分类任务
        |--通过 datasets.xxx.load_data()函数即可实现经典数据集的自动加载,其中 xxx 代表具体的数据集名称,   数据缓存在用户目录下的.keras/datasets 文件夹
        |--通过 load_data()函数会返回相应格式的数据,对于图片数据集 MNIST,CIFAR10 等,会返回 2 个 tuple,第一个 tuple 保存了用于训练的数据 x 和 y 训练集对象;第 2 个 tuple 则保存了用于测试的数据 x_test 和 y_test 测试集对象,所有的数据都用 Numpy 数组容器保存
        |--通过 Dataset.from_tensor_slices 可以将训练部分的数据图片 x 和标签 y 都转换成Dataset 对象


5.7.1 随机打散
    -----通过 Dataset.shuffle(buffer_size)工具可以设置 Dataset 对象随机打散数据之间的顺序,防止每次训练时数据按固定顺序产生,从而使得模型尝试"记忆"住标签信息,buffer_size 参数指定缓冲池的大小


5.7.2 批训练
    -----为了利用显卡的并行计算能力,一般在网络的计算过程中会同时计算多个样本,我们把这种训练方式叫做批训练,其中一个批中样本的数量叫做 Batch Size
        |--train_db = train_db.batch(128)  128 为 Batch Size 参数,即一次并行计算 128 个样本的数据
        |--Batch Size 一般根据用户的 GPU 显存资源来设置


5.7.3 预处理
    -----Dataset 对象通过提供 map(func)工具函数,可以非常方便地调用用户自定义的预处理逻辑,它实现在 func 函数里


5.7.4 循环训练
    -----当对 train_db 的所有样本完成一次迭代后,for 循环终止退出.这样完成一个 Batch 的数据训练,叫做一个 Step;通过多个 step 来完成整个训练集的一次迭代,叫做一个 Epoch.在实际训练时,通常需要对数据集迭代多个 Epoch 才能取得较好地训练效果


5.8 MNIST 测试实战


