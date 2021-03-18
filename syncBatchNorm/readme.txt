什么是SyncBN？
SyncBN就是Batch Normalization(BN)。其跟一般所说的普通BN的不同在于工程实现方式：
SyncBN能够完美支持多卡训练，而普通BN在多卡模式下实际上就是单卡模式。

我们知道，BN中有moving mean和moving variance这两个buffer，这两个buffer的更新依赖于当前训练轮次的batch数据的计算结果。
但是在普通多卡DP模式下，各个模型只能拿到自己的那部分计算结果，
所以在DP模式下的普通BN被设计为只利用主卡上的计算结果来计算moving mean和moving variance，
之后再广播给其他卡。这样，实际上BN的batch size就只是主卡上的batch size那么大。
当模型很大、batch size很小时，这样的BN无疑会限制模型的性能。

为了解决这个问题，PyTorch新引入了一个叫SyncBN的结构，利用DDP的分布式计算接口来实现真正的多卡BN。


SyncBN的原理
SyncBN的原理很简单：SyncBN利用分布式通讯接口在各卡间进行通讯，从而能利用所有数据进行BN计算。
为了尽可能地减少跨卡传输量，SyncBN做了一个关键的优化，即只传输各自进程的各自的 小batch mean和 小batch variance，
而不是所有数据。具体流程请见下面：

前向传播
在各进程上计算各自的 小batch mean和小batch variance
各自的进程对各自的 小batch mean和小batch variance进行all_gather操作，每个进程都得到s的全局量。
注释：只传递mean和variance，而不是整体数据，可以大大减少通讯量，提高速度。
每个进程分别计算总体mean和总体variance，得到一样的结果
注释：在数学上是可行的，有兴趣的同学可以自己推导一下。
接下来，延续正常的BN计算。
注释：因为从前向传播的计算数据中得到的batch mean和batch variance在各卡间保持一致，
所以，running_mean和running_variance就能保持一致，不需要显式地同步了！
后向传播：和正常的一样


SyncBN与DDP的关系
一句话总结，当前PyTorch SyncBN只在DDP单进程单卡模式中支持。SyncBN用到 all_gather这个分布式计算接口，
而使用这个接口需要先初始化DDP环境。

复习一下DDP的伪代码中的准备阶段中的DDP初始化阶段

d. 创建管理器reducer，给每个parameter注册梯度平均的hook。
i. 注释：这一步的具体实现是在C++代码里面的，即reducer.h文件。
e. （可能）为可能的SyncBN层做准备
这里有三个点需要注意：

这里的为可能的SyncBN层做准备，实际上就是检测当前是否是DDP单进程单卡模式，如果不是，会直接停止。
这告诉我们，SyncBN需要在DDP环境初始化后初始化，但是要在DDP模型前就准备好。
为什么当前PyTorch SyncBN只支持DDP单进程单卡模式？
从SyncBN原理中我们可以看到，其强依赖了all_gather计算，而这个分布式接口当前是不支持单进程多卡或者DP模式的。
当然，不排除未来也是有可能支持的。


怎么用SyncBN？
怎么样才能在我们的代码引入SyncBN呢？很简单：

# DDP init
dist.init_process_group(backend='nccl')

# 按照原来的方式定义模型，这里的BN都使用普通BN就行了。
model = MyModel()
# 引入SyncBN，这句代码，会将普通BN替换成SyncBN。
model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)

# 构造DDP模型
model = DDP(model, device_ids=[local_rank], output_device=local_rank)
又是熟悉的模样，像DDP一样，一句代码就解决了问题。这是怎么做到的呢？

convert_sync_batchnorm的原理：

torch.nn.SyncBatchNorm.convert_sync_batchnorm会搜索model里面的每一个module，
如果发现这个module是、或者继承了torch.nn.modules.batchnorm._BatchNorm类，
就把它替换成SyncBN。也就是说，如果你的Normalization层是自己定义的特殊类，
没有继承过_BatchNorm类，那么convert_sync_batchnorm是不支持的，需要你自己实现一个新的SyncBN！