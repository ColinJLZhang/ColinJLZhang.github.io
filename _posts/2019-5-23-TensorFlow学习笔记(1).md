---
layout:     post
title:      2019-5-23-TensorFlow学习笔记(1)
subtitle:   底层API学习
date:       2019-05-23
author:     Colin
header-img: img/post-bg-mma-5.jpg
catalog: true
comments: true
tags:
    - tensorflow
    - python
    - 基础学习
---
### 简介

众所周知，tensorflow具有高层的API可以方便的构建模型，通过tf.keras,tf.Data,tf.Estimators可以方便的构建模型，我们无需操心模型的底层运算，但是这也让我们无法对底层的数据进行操作，如果我希望自己控制我的训练循环情况我想要对某一个tensor进行特殊化的操作。另外了解底层的操作会对我们对代码的理解和调试有一个更好的指导。

##### 1. Tensor Values

张量值我认为比较类似于一个矩阵，它包含了秩和形状两个属性，可以这样直观的理解，在python中秩等于中括号“[“的层数，而形状则是先从最外层的括号中计算包含多少元素直到最内层，请看官网给出的例子：
>3\. \# a rank 0 tensor; a scalar with shape [],,
>
>[1., 2., 3.] # a rank 1 tensor; a vector with shape [3]
>
>[[1., 2., 3.], [4., 5., 6.]] # a rank 2 tensor; a matrix with shape [2, 3]
>
>[[[1., 2., 3.]], [[7., 8., 9.]]] # a rank 3 tensor with shape [2, 1, 3]

##### 2. TensorFlow Core

* 构建图(bulid graph)

    tensorflow的图包含了operation（op，节点，描述对张量的计算） 和 tensor两个重要的部分，通过对tensor的operations的行为构建了整个一张图，我们看一个例子<br>
    
        
        a = tf.constant(3.0, dtype=tf.float32)
        b = tf.constant(4.0) # also tf.float32 implicitly
        total = a + b
        print(a)
        print(b)
        print(total)        
        
        >>>Tensor("Const:0", shape=(), dtype=float32)
        >>>Tensor("Const_1:0", shape=(), dtype=float32)
        >>>Tensor("add:0", shape=(), dtype=float32)

* 计算图(run graph)

    从前面的打印信息我们可以看到，输出的并不是我们预想中的计算值，实际上我们只是完成了整个张量图的构建，在运行这个图之前我们无法得到一个值。

##### 3. Session

要评估张量，需要实例化一个 tf.Session 对象（非正式名称为会话）。会话会封装 TensorFlow 运行时的状态，并运行 TensorFlow 操作。如果说 tf.Graph 像一个 .py 文件，那么 tf.Session 就像一个 python 可执行对象。

    vec = tf.random_uniform(shape=(3,))
    out1 = vec + 1
    out2 = vec + 2
    print(sess.run(vec))
    print(sess.run(vec))
    print(sess.run((out1, out2)))

    >>>[ 0.52917576  0.64076328  0.68353939]
    >>>[ 0.66192627  0.89126778  0.06254101]
    >>>(
      array([ 1.88408756,  1.87149239,  1.84057522], dtype=float32),
      array([ 2.88408756,  2.87149239,  2.84057522], dtype=float32)
    )        

必须要注意到，在调用tf.Session.run期间，任何Tensor都只能有一个单值不可更改。如上述代码每个out的运算过程都会调用random_uniform函数但是输出的过程却是相同的。

##### 4. feed_dict

到目前为止，我们可以看到这个graph还是比较无趣的只能计算一些常量的加减乘除，为了一些更有趣的玩法我们要引入placeholder的概念，这里有点类似于python函数中传入的参数，在中文里我们一般翻译为占位符。

    x = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)
    z = x + y

上面定义了两个参数x,y然后计算他们的和，我们可以使用run方法将参数通过feed_dict的方式把参数给”喂“进去，

    print(sess.run(z, feed_dict={x: 3, y: 4.5}))
    print(sess.run(z, feed_dict={x: [1, 3], y: [2, 4]}))    
    
    >>>7.5
    >>>[ 3.  7.]

##### 5. Data

placeholder的方式适用于简单的少量的输入，比如我们的模型训练完成以后运用这个模型去预测新的样本值时我们的输入只有一个样本。但是在深度学习中，我们训练时的样本都是一个大型的数据集，这时我就可以运用tf.data。

要从数据集中获取可运行的tf.Tensor，您必须先将其转换成tf.data.Iterator，然后调用迭代器的 get_next 方法。

创建迭代器的最简单的方式是采用 make_one_shot_iterator方法。例如，在下面的代码中，next_item 张量将在每次 run 调用时从 my_data阵列返回一行：

    my_data = [
        [0, 1,],
        [2, 3,],
        [4, 5,],
        [6, 7,],
    ]
    slices = tf.data.Dataset.from_tensor_slices(my_data)
    next_item = slices.make_one_shot_iterator().get_next()

到达数据流末端时，Dataset 会抛出 OutOfRangeError。例如，下面的代码会一直读取 next_item，直到没有数据可读：

    while True:
      try:
        print(sess.run(next_item))
      except tf.errors.OutOfRangeError:
        break

如果 Dataset 依赖于有状态操作，则可能需要在使用迭代器之前先初始化它，如下所示：

    r = tf.random_normal([10,3])
    dataset = tf.data.Dataset.from_tensor_slices(r)
    iterator = dataset.make_initializable_iterator()
    next_row = iterator.get_next()

    sess.run(iterator.initializer)
    while True:
      try:
        print(sess.run(next_row))
      except tf.errors.OutOfRangeError:
        break

##### 6. layers

层的概念主要是把变量和操作打包，比如Dense层是对应的将每一个输入进行加权和以及非线性（optional）。连接的权重和偏置项由层对象进行管理。下面我来试试Dense层：

    x = tf.placeholder(tf.float32, shape=[None, 3])
    linear_model = tf.layers.Dense(units=1) #输出结果的维度
    y = linear_model(x)

上面我们已经创建了一个 [Dense](https://www.tensorflow.org/api_docs/python/tf/layers/Dense) 层，层会检查其输入数据，以确定其内部变量的大小。因此，我们必须在这里设置x占位符的形状，以便层构建正确大小的权重矩阵。所以我觉得这其实是一个中级的API，至少这里的Tensor变量是包含在层的内部而且能够根据输入变量的大小自动设置适应训练参数的大小但是官网放在Low Level APIs的Introduction里面我也没办法了。

因为这里的层包含了一组变量，我们首先需要将其初始化才能够进行运用。BTY这里的Dense是被封装成了一个对象。

*初始化层：*
    
    init = tf.global_variables_initializer()
    sess.run(init)         #init在为经过session的运行之前只是一个初始化函数的句柄，而并没有初始化任何变量

*执行层：*

    print(sess.run(y, {x: [[1, 2, 3],[4, 5, 6]]}))

    >>>[[-3.41378999]
    >>>[-9.14999008]]

*层以函数的快捷方式：*

对于每个层类（如 tf.layers.Dense)，TensorFlow 还提供了一个快捷函数（如 tf.layers.dense）。两者唯一的区别是快捷函数版本是在单次调用中创建和运行层。例如，以下代码等同于较早的版本：

    x = tf.placeholder(tf.float32, shape=[None, 3])
    y = tf.layers.dense(x, units=1)

    init = tf.global_variables_initializer()
    sess.run(init)

    print(sess.run(y, {x: [[1, 2, 3], [4, 5, 6]]}))

尽管这种方法很方便，但无法访问tf.layers.Layer对象。这会让自省和调试变得更加困难，并且无法重复使用相应的层。

##### 7. feature_column

这个概念我暂时还没有完全的理解，目前我的看法是，通过feature_column可以对输入的特征进行整合和编码，比如下面的例子中就是如果我有两个向量都是对目标的特征描述，那么我可以通过tf.feature_column_layer把它转化成一个输入特征的tensor？ Anyway先放上代码再说：

    features = {
        'sales' : [[5], [10], [8], [9]],
        'department': ['sports', 'sports', 'gardening', 'gardening']}

    department_column = tf.feature_column.categorical_column_with_vocabulary_list(
            'department', ['sports', 'gardening'])
    department_column = tf.feature_column.indicator_column(department_column)

    columns = [
        tf.feature_column.numeric_column('sales'),
        department_column
    ]

    inputs = tf.feature_column.input_layer(features, columns)    

运行inpts会将features解析成一批向量。

特征列和层一样具有内部状态，因此通常需要将它们初始化。类别列会在内部使用对照表，而这些表需要单独的初始化操作 tf.tables_initializer。

    var_init = tf.global_variables_initializer()
    table_init = tf.tables_initializer()
    sess = tf.Session()
    sess.run((var_init, table_init))

    print(sess.run(inputs))

    >>>[[  1.   0.   5.]
        [  1.   0.  10.]
        [  0.   1.   8.]
        [  0.   1.   9.]]        

##### 8. 训练一个简单的回归模型

1. 定义数据

    首先先定义一些输入值x,维度是一维，以及每个输入数据对应的输出值y_ture

        x = tf.constant([[1], [2], [3], [4]], dtype=tf.float32)
        y_true = tf.constant([[0], [-1], [-2], [-3]], dtype=tf.float32)


2. 定义模型
   
    定义一个简单的线性模型，输出值为一个实数，数学模型为 $y = \sigma(w*x+b)$

        linear_model = tf.layers.Dense(units=1)

        y_pred = linear_model(x)
    
    那么这个模型的预测值是多少我们现在就可以看到了：

        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)

        print(sess.run(y_pred))

        >>>[[1.1371852]
            [2.2743704]
            [3.4115558]
            [4.548741 ]] 

    也许你会想我都还没有训练怎么就可以开始预测了呢？因为通过初始化得到了所有参数的初始值，那么这个图就可以开始计算了，只是最后的结果和y_true相差很大。         

3. 定义损失函数

    损失函数一般的理解就是模型预测值和我们数据库中真实的输出值的距离，目前意义上的nn模型都是在尽可能的拟合我们数据库中所提供的样本，并在大数据的前提下提供一定程度上的泛化能力。为了拟合数据集的样本当然是差距越小越好，所以我们整个网络的训练也是基于loss。这里我们定义一个均方误差。tensorflow提供了一系列的损失函数所我们不用自己去写损失函数了。

        loss = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)

        print(sess.run(loss))

4. 训练模型

    如前所述，我的模型目的就是以降低loss为目标调整参数，tensorflow提供了一系列的的优化方法供我们使用，些优化器被实现为 tf.train.Optimizer 的子类。它们会逐渐改变每个变量，以便将损失最小化。最简单的优化算法是梯度下降法，由tf.train.GradientDescentOptimizer实现。它会根据损失相对于变量的导数大小来修改各个变量。例如：

        optimizer = tf.train.GradientDescentOptimizer(0.01)
        train = optimizer.minimize(loss)

    该代码构建了优化所需的所有图组件，并返回一个训练指令。该训练指令在运行时会更新图中的变量。您可以按以下方式运行该指令：

        for i in range(100):
          _, loss_value = sess.run((train, loss))
          print(loss_value)

        >>>24.570778
        17.16924
        12.03275
        8.467935
        5.9936767
        4.276136
        ....

### Tensor

在编写TensorFlow程序的时候，大部分是对tensor的操作，tensor是对向量和矩阵想更高维度的泛化。在定义一个tensor的对象的时候我们更像是在计算机上输入一个计算的公式，但是在我们按下“=”之前是没有办法得到这个公式的输出值的。

声明一个常数类型的tensor：
    
    tem = tf.constant([1,2,3])

tf.Tensor具有以下属性：
 
 * 数据类型 （float32、int32、string） 
    
        tem.dtape

    [数据类型列表](https://www.tensorflow.org/api_docs/python/tf/dtypes/DType)
 
 * 形状,一个tensor的形状有可能是被部分定义的或者是依赖于别的tensor所以只有在运行整个图的时候才能完全确定他的形状。

        tem.shape

    如何改变tensor的形状：tf.reshape()

        range9 = tf.range(9)
        3_3_mat = tf.reshape(range9, (3,3)) #you can't do it by range9.reshap()

 * 秩（Rank),需要注意这里的秩与线性代数中的秩不同，这里的意义为维度。

    | Rank | Math entity
    |------|------------
    | 0    | Scalar (magnitude only)
    | 1    | Vector (magnitude and direction)
    | 2    | Matrix (table of numbers)
    | 3    | 3-Tensor (cube of numbers)
    | n    | n-Tensor (you get the idea)

    生成一个0阶矩阵：

        mammal = tf.Variable("Elephant", tf.string)
        ignition = tf.Variable(451, tf.int16)
        floating = tf.Variable(3.14159265359, tf.float64)
        its_complicated = tf.Variable(12.3 - 4.85j, tf.complex64)

    生成一个1阶矩阵：

        mystr = tf.Variable(["Hello"], tf.string)
        cool_numbers  = tf.Variable([3.14159, 2.71828], tf.float32)
        first_primes = tf.Variable([2, 3, 5, 7, 11], tf.int32)
        its_very_complicated = tf.Variable([12.3 - 4.85j, 7.5 - 6.23j], tf.complex64)

    更高阶的矩阵：

        mymat = tf.Variable([[7],[11]], tf.int16)
        myxor = tf.Variable([[False, True],[True, False]], tf.bool)
        linear_squares = tf.Variable([[4], [9], [16], [25]], tf.int32)
        squarish_squares = tf.Variable([ [4, 9], [16, 25] ], tf.int32)
        rank_of_squares = tf.rank(squarish_squares)
        mymatC = tf.Variable([[7],[11]], tf.int32)

    在典型的CNN网络中我们经常用到的就是四阶的张量，分别对应批次的大小，图像的宽度，图像的高度，以及颜色通道。

        my_image = tf.zeros([10,300,300,3]) #batch x height x width x color

    获取对象的阶(注意通过tf.rank()获得的对象的阶仍然是一个tensor的对象，需要进行评估后才能获得值)：

        r = tf.rank(my_image)  # sess.run(r) resulting 4 

tensor与numpy array有很多性质相似，支持下标索引元素，切片操作，改变形状。

0 阶张量（标量）不需要索引，因为其本身便是单一数字。

对于 1 阶张量（矢量），可以通过传递一个索引返回一个标量的数字（after sess.run()）：

    my_scalar = my_vector[2]  #index could be scalar Tensor variable

同样的对于 2 阶的张量：
    
    my_scalar = my_matrix[1,2]        #return a scalar
    my_raw_vector = my_matrix[2]      #return a vector, the second row
    my_column_vector = my_matrix[:,2] #return a vector, the second column

张量的评估，除了Session还有一个常用的办法就是eval():

    t = tf.constant([1,2,3])
    print(t)  # This will print the symbolic tensor when the graph is being built.
              # This tensor does not have a value in this context.
    print(t.eval())   # >>[1,2,3] 

注意，如果要使用tensor.eval()那么一个默认的tf.Session必须是激活的，否则需要传入会话参数eval(session=sess),下面再加一个例子：

    p = tf.placeholder(tf.float32)
    t = p + 1.0
    t.eval()  # This will fail, since the placeholder did not get a value.
    t.eval(feed_dict={p:2.0})  # This will succeed because we're feeding a value
                               # to the placeholder.
    
在调试过程中，我们还经常使用tf.Print()，通过t.Print()我们在打印最后的tensor值的同时也可以将一些中间变量同时打印出来，比如：
    
    t = tf.constant([1,2,3])
    t = tf.Print(t.[t])   #by adding this code, when you print any tensor the value of t will be printed
    result = t + 1

    print(sess.run(result))

    >>>[1 2 3]       #current value of the tensor t
    array([2, 3, 4]) #current value of the tensor result

### Variables

TensorFlow 中的变量与其他程序中的变量是相识的，用来表示那些会变化的量，比如在训练过程中的参数随着迭代会改变。<br>
variable 可以通过tf.Variable类进行操作，通过tf.variable的操作可以改变它的值，与tf.Tensor不同的是tf.variable可以存在多个不同的session之间。

在 TensorFlow 内部，tf.Variable 会存储持久性张量。具体 op 允许您读取和修改此张量的值。这些修改在多个 tf.Session 之间是可见的，因此对于一个 tf.Variable，多个工作器可以看到相同的值。

##### 1. 创建变量

通过tf.get_variable()可以轻松的创建变量，只需要提供名称和形状即可,注意名字不可重复
    
    my_variable = tf.get_variable("my_varible", [1,2,3])

上面的代码会创建一个形状为[1，2，3]的三维张量，默认情况这个变量具有dtype tf.float32，初始值通过tf.glorot_uniform_initializer随机设置。

我们也可以指定dtype和initializer：
    
    my_int_variable = tf.get_variable("my_int_variable", [1,2,3],
        dtype=tf.int32, initializer=tf.zeros_initializer)

除此之外，也可以直接将variable初始化为tensor值：

    other_variable = tf.get_variable("other_variable", dtype=tf.int32, initializer=tf.constant([23,42]))

注意，如果初始化器是 tf.Tensor 时，**不能指定形状**，这时的形状应该和tensor一致。

- 变量的集合

    由于TensoFlow存在一些独立的部分未被连接，为了访问所有的变量tensorflow提供了集合，集合提供了变量的名称列表。

    一般情况下，每个tf.Variable都放置在一下两个集合中：

    - tf.GraphKeys.GLOBAL_VARIABLES -在多台设备间共享的变量
    - tf.GraphKeys.TRAINABLE_VARIABLES-TensorFlow将计算其梯度的变量

    如果不希望变量得到训练，可以将其添加到tf.GraphKeys.LOCAL_VARIABLES集合中：

        my_local = tf.get_variable("my_local_variable",
                                    shape=()
                                    collections=[tf.GraphKeys.LOCAL_VARIABLES])

    或者显示的指定不可训练，trainable = Flase

        my_non_trainable = tf.get_variable("my_non_trainable",
                                            shape=()
                                            trainable=False)

    当然可以使用我们自己的集合，集合名称可以为任何可哈希字符串，而且无须显示的创建，开下面的例子：

        tf.add_to_collection("my_collection_name", my_local)

    上面的代码中我们没有显示的创建 my_collection_name 这个变量，当我们执行这段代码的时候就会自动创建这个集合。

    tf.get_collection("my_collection_name")可以检索所有存在该集合的变量

        tf.get_collection("my_collection_name")
        >>> [<tf.Variable 'my_local:0' shape=() dtype=float32_ref>]


- 分布式下指定变量
    
    与任何其他TensorFlow指令一样，您可以将变量放置在特定设备上。例如，以下代码段创建了名为 v 的变量并将其放置在第二个 GPU 设备上：

        with tf.device("/device:GPU:1"):
          v = tf.get_variable("v", [1])

    在分布式设置中，将变量放置在正确设备上尤为重要。如果不小心将变量放在工作器而不是参数服务器上，可能会严重减慢训练速度，最坏的情况下，可能会让每个工作器不断复制各个变量。为此，我们提供了 tf.train.replica_device_setter，它可以自动将变量放置在参数服务器中。例如：

        cluster_spec = {
            "ps": ["ps0:2222", "ps1:2222"],
            "worker": ["worker0:2222", "worker1:2222", "worker2:2222"]}
        with tf.device(tf.train.replica_device_setter(cluster=cluster_spec)):
                  v = tf.get_variable("v", shape=[20, 20])  # this variable is placed
                                                            # in the parameter server
                                                            # by the replica_device_setter

##### 2. 初始化变量

变量必须先初始化以后才可以使用。当使用低级API显示的构建了图和会话那么必须明确的进行变量初始化。[tf.contrib.slim](https://www.tensorflow.org/api_docs/python/tf/contrib/slim)、[tf.estimator.Estimator](https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator) 和Keras等大多数高级框架在训练模型前会自动为您初始化变量。

显示的初始化变量有众多好处，其中一个就是可以允许我们从检查点开始初始化参数，避免从头开始训练。

在训练开始前一次性的初始化所有的可训练变量方法：tf.global_variables_initializer(), 这个函数会返回一个operation，负责初始话tf.GraphKeys.GLOBAL_VARIABLES集合中的所有变量。运行这个操作会初始化所有的变量：

  1. 全部初始化 &nbsp;&nbsp; tf.global_variables_initializer()

        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)

    注意：init = tf.global_variables_initializer() 函数的调用一定是在变量创建完成以后才能有效的初始化，因为tf.global_variables_initializer()是把目前tf.GraphKeys.GLOBAL_VARIABLES集合里面的变量初始化而后面创建的变量现在是没有在这个集合中的，比如：

        init = tf.global_variables_initializer()
        my_variable = tf.get_varibales("my_variable", (1,2,3))
        sess.run(init)  #这里的初始化是失败的，因为init没有吧"my_variable"这个变量纳入到初始化的列表中。
  
  2. 初始化某一个变量 &nbsp;&nbsp; my_variable.initialer

        my_variable = tf.get_variable("my_variable", shape=(1,2,3))
        sess.run(my_variable.initializer)

  3. 查询哪些变量尚未初始化 &nbsp;&nbsp; tf.report_uninitialized_variable()

        print(sess.run(tf.report_uninitialized_variable()))  #this will return a list
        print(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
        print(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))

  4. 存在相关关系的变量初始化
    
    如果某一个变量的值依赖于另一个变量，由于global_variables_initializer 不会指定初始化的顺序，那么在初始化时就会出现错误。这种情况下，当我们要使用一个变量的值依赖于另一个变量的值时，我们一般情况下可以

       a. 先初始化被引用的变量然后再初始化引用变量

       b. 引用变量时，使用variable。initialized_value() 而不是直接引用。

        v = tf.get_variable("v", shape=(), initializer=tf.zeros_initializer())
        w = tf.get_variable("w", initializer=v.initialized_value() + 1)

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        sess.run((v,w))
        >>>(0.0, 1.0)


##### 3. 调用变量

在使用 variable 中的值的时候，只需要将其视作普通的Tensor即可：

    v = tf.get_variable("v", shape=(), initializer=tf.zeros_initializer())
    w = v + 1

改变变量所带有的值，可以使用assign，assign_add方法等，比如：

    v = tf.get_variable("v", shape=(), initializer=tf.zeros_initializer())
    assignment = v.assign_add(1)
    tf.global_variables_initializer().run()
    sess.run(assignment)  # or assignment.op.run(), or assignment.eval()

##### 4. 共享变量



参考：

[tfensorflow官方文档](https://www.tensorflow.org/guide/)

**HAPPY CODING!**