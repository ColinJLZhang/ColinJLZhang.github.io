---
layout:     post
title:      TensorFlow学习笔记(1)
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
    sess.run(v)
    >>> 0
    sess.run(assignment)  # or assignment.op.run(), or assignment.eval()
    >>> 1
    sess.run(v)           # comes to 1 because of the assignment
    >>> 1
    sess.run(assignment)  # try to do it again
    >>> 2
    sess.run(v)           # add 1 more on v
    >>> 2

可以看到改变变量所带有的值是通过类似与assign的操作来实现的，而且必须注意到这些操作必须要通过一个会话（session）来运行才会显示出实际的效果，而且这些操作可以重复的执行，对变量进行操作。

##### 4. 查看变量

由于变量的值是变化的所以整个过程我们最好能有办法对变量的值进行监控，也许你会想，直接 sess.run() 或者是 eval() 就可以查看了吗？ 事实上有的时候你可以这么做，但是这样做的实际意义是把变量的图运行一遍等到的变量值，但是如果我们运用了上面的 tf.assign 的方法不但得不到我们想要的结果我们甚至会把 tf.assign() 的结果覆盖掉，所以正确的查看变量方法是 tf.read_value()

    v = tf.get_variable("v", shape=(), initializer=tf.zeros_initializer())
    assignment = v.assign_add(1)
    w = v.read_value()     # w is guaranteed to reflect v's value after the
                           # assign_add operation.

    sess.run(v)
    >>> 0
    sess.run(assignment)   # execute assignment operation
    >>> 1
    sess.run(w)            # check value of the v
    >>> 1
    sess.run(v)            # look at v (I can do this 
    >>> 1                  #  beacuse v is not in a graph)

##### 5. 共享变量

TensorFlow 支持两种共享变量的方式：

  - 显式传递 tf.Variable 对象。
  - 将 tf.Variable 对象隐式封装在 tf.variable_scope 对象内。

虽然显式传递变量的代码非常清晰，但有时编写在其实现中隐式使用变量的 TensorFlow 函数非常方便。tf.layers中的大多数功能层以及所有tf.metrics和部分其他库实用程序都使用这种方法。

变量作用域允许您在调用隐式创建和使用变量的函数时控制变量重用。作用域还允许您以分层和可理解的方式命名变量。

例如，假设我们编写一个函数来创建一个卷积/relu 层：
```python
    def conv_relu(input, kernel_shape, bias_shape):
        # Create variable named "weights".
        weights = tf.get_variable("weights", kernel_shape,
            initializer=tf.random_normal_initializer())
        # Create variable named "biases".
        biases = tf.get_variable("biases", bias_shape,
            initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(input, weights,
            strides=[1, 1, 1, 1], padding='SAME')
        return tf.nn.relu(conv + biases)
```
此函数使用短名称 weights 和 biases，这有利于清晰区分二者。然而，在真实模型中，我们需要很多此类卷积层，而且重复调用此函数将不起会报错，因为在 tf.GraphKeys.TRAINABLE_VARIABLES 中已经存在了改变了，无法重新创建可以相同的变量：

```python
    input1 = tf.random_normal([1,10,10,32])
    input2 = tf.random_normal([1,20,20,32])
    x = conv_relu(input1, kernel_shape=[5, 5, 32, 32], bias_shape=[32])
    x = conv_relu(x, kernel_shape=[5, 5, 32, 32], bias_shape = [32])  # This fails.
    >>> ValueError: Variable weights already exists, disallowed.
```

在上述的操作中，第二行的 conv_relu() 的调用存在明显的歧义，在conv_relu中是重新创建 weight 等参数 还是直接沿用上一次所创建的变量，如果是创建一个新的变量由于不允许相同名称 tensor 的出现那么我们需要指定新的命名空间：

```python
    def my_image_filter(input_images):
        with tf.variable_scope("conv1"):
            # Variables created here will be named "conv1/weights", "conv1/biases".
            relu1 = conv_relu(input_images, [5, 5, 32, 32], [32])
        with tf.variable_scope("conv2"):
            # Variables created here will be named "conv2/weights", "conv2/biases".
            return conv_relu(relu1, [5, 5, 32, 32], [32])
```

如果您想要共享(复用)变量，有两种方法可供选择。首先，您可以使用 reuse=True 0 创建具有相同名称的作用域：

```python
    with tf.variable_scope("model"):
      output1 = my_image_filter(input1)
    with tf.variable_scope("model", reuse=True):
      output2 = my_image_filter(input2)
```

同时，也可以调用 scope.reuse_variables()以触发重用：

```python
    with tf.variable_scope("model") as scope:
      output1 = my_image_filter(input1)
      scope.reuse_variables()
      output2 = my_image_filter(input2)
```
由于依赖于作用域的确切字符串名称可能比较危险(万一打错了岂不是完蛋)，因此也可以根据另一作用域初始化某个变量作用域：

```python
    with tf.variable_scope("model") as scope:
      output1 = my_image_filter(input1)
    with tf.variable_scope(scope, reuse=True):
      output2 = my_image_filter(input2)
```
### Graph and Session

#### 1. 什么是数据流图

TensorFlow 使用数据流图将计算表示为独立的指令之间的依赖关系。这可生成低级别的编程模型，在该模型中，您首先定义数据流图，然后创建 TensorFlow 会话，以便在一组本地和远程设备上运行图的各个部分。

如果您计划直接使用低级别编程模型，本指南将是您最实用的参考资源。较高阶的 API（例如 tf.estimator.Estimator 和 Keras）会向最终用户隐去图和会话的细节内容。

你可以把数据流图看作是一个巨大的表达式，这个表达式除了加减乘除幂次方等基本运算法则还包括了卷积，求梯度，优化变量等运算法则，甚至你可以定义自己的运算法则，最后当我们运行这个图的时候类似于我们按下计算器的等号，我们就可以得到最后我们要的参数的结果。

<center>  
    <img src="..\..\..\..\img\article\tensors_flowing.gif">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">数据流图 图源tensorflow官网</div>
</center>

数据流是一种用于并行计算的常用编程模型。在数据流图中，节点表示计算单元，边缘表示计算使用或产生的数据。例如，在 TensorFlow 图中，tf.matmul操作对应于单个节点，该节点具有两个传入边（要相乘的矩阵）和一个传出边（乘法结果）。

#### 2. 为什么使用数据流图

在执行您的程序时，数据流可以为 TensorFlow 提供多项优势：

- **并行处理。** 通过使用明确的边缘来表示操作之间的依赖关系，系统可以轻松识别能够并行执行的操作。

- **分布式执行。** 通过使用明确的边缘来表示操作之间流动的值，TensorFlow 可以将您的程序划分到连接至不同机器的多台设备上（CPU、GPU 和 TPU）。TensorFlow 将在这些设备之间进行必要的通信和协调。

- **编译。** TensorFlow 的 XLA 编译器可以使用数据流图中的信息生成更快的代码，例如将相邻的操作融合到一起。

- **可移植性。** 数据流图是一种不依赖于语言的模型代码表示法。您可以使用 Python 构建数据流图，将其存储在 SavedModel 中，并使用 C++ 程序进行恢复，从而实现低延迟的推理。

#### 3. tf.Graph

tf.Graph 包含两个信息：

- **图结构。**

    图结构是指，所有的运算是如何连接起来的，如何从一个图的边缘运算到另外的一个图边缘的

- **图集合。**

    TensorFlow为了方便对各种节点变量的管理，提供一种集合的管理方式。你可以将你的对象放置到某一个集合中，[tf.GraphKeys](https://www.tensorflow.org/api_docs/python/tf/GraphKeys) 预先定义了一些集合，他们具有特殊的意义，不同的集合下有不同的作用。通过tf.add_to_collection函数允许你把你的对象放到这些集合中，tf.get_collection 允许您查询与某个键关联的所有对象。TensorFlow库的许多部分会使用此设施资源：例如，当您创建 tf.Variable时，系统会默认将其添加到表示“全局变量”和“可训练变量”的集合中。当您后续创建tf.train.Saver 或 tf.train.Optimizer 时，这些集合中的变量将用作默认参数。

#### 4. 构建 tf.Graph

tensorflow 总是在构建一个图并对图进行计算，通过tensorflowd的函数可以构建新的 tf.Operation (节点) 和 tf.Tersor (边)。 TensorFlow 提供一个默认图，这个图是同一上下文中所有API函数的明确参数。例如：

- 调用 tf.coanstant(42.0) 可以创建单个的 tf.Operation, 该操作会生成一个值42.0，并将该值添加到默认的图中，返回一个常量的 tf.Tensor 。
- 调用 tf.matmul(x, y) 可以创建一个 tf.Operation, 该操作会将x , y 对象进行矩阵乘，将其添加到默认的图中，返回表示乘法运算结果的 tf.Tensor.
- 执行 v = tf.Variable(0)可以为图添加一个 tf.Operation 这里创建了一个可以存储和写入的张量值，该值在多个张量之间保持恒定。这个操作返回的对象具有 assign 和 assign_add 等方法， 这些方法可以创建一个 tf.Operation 对象，在执行图的过程中对图前述的值进行改变。
- 调用 tf.train_Optimizer.minimize 可以将操作和张量添加到计算梯度的默认图中，并返回一个 tf.Operation 该操作在运行时会将梯度运用到训练的参数上以改变参数值获得更低的 loss。

大多数程序仅依赖于默认图。尽管如此，请参阅处理多个图了解更加高级的用例。高阶 API（比如 tf.estimator.Estimator API）可替您管理默认图，并且还具有其他功能，例如创建不同的图以用于训练和评估。

#### 5. 命名空间

TensorFlow 通过命名空间帮助我们更好的管理变量名称。 TF在图中只会允许存在唯一性的名称，如果存在相同的名称命名，有的时候TF会自动为我们创建新的变量名，如果出现了歧义的代码甚至会出现报错，我们来看一个例子：

TF会自动帮我们改变变量名的情况：

```python
a = tf.constant(0, name="c")  # => operation named "c"
# >>> <tf.Tensor 'c:0' shape=() dtype=int32>

# Already-used names will be "uniquified".
b = tf.constant(2, name="c")  # => operation named "c_1"
# >>> <tf.Tensor 'c_1:0' shape=() dtype=int32>

# Name scopes add a prefix to all operations created in the same context.
with tf.name_scope("outer"):
  c_2 = tf.constant(2, name="c")  # => operation named "outer/c"

  # Name scopes nest like paths in a hierarchical file system.
  with tf.name_scope("inner"):
    c_3 = tf.constant(3, name="c")  # => operation named "outer/inner/c"

  # Exiting a name scope context will return to the previous prefix.
  c_4 = tf.constant(4, name="c")  # => operation named "outer/c_1"

  # Already-used name scopes will be "uniquified".
  with tf.name_scope("inner"):
    c_5 = tf.constant(5, name="c")  # => operation named "outer/inner_1/c"
```

TF会报错的情况：

```python
a = tf.get_variable("weights", shape=(3,3))  # creat a trainable variable named weights
# >>> <tf.Variable 'weights:0' shape=(3, 3) dtype=float32_ref>

b = tf.get_variable("weights", shape=(3,3)) # do u want to reuse the variable or wanna a new variable?
# >>> ValueError: Variable weights already exists,
#               disallowed. Did you mean to set reuse=True ...

```

报错的原因就是因为这里程序不确定你的目的是创建一个新的变量还是想要复用之前的变量，所以TF没有贸然的创建一个变量而是提醒你是想要复用还是重新创建一个变量。

#### 6. 将operation放置在不同的设备上

（这部分我暂时用不上先，等到以后用了再回来详细研究，这里直接放上官方API）

如果您希望 TensorFlow 程序使用多台不同的设备，则可以使用 tf.device 函数轻松地请求将在特定上下文中创建的所有操作放置到同一设备（或同一类型的设备）上。

设备规范具有以下形式：

    /job:<JOB_NAME>/task:<TASK_INDEX>/device:<DEVICE_TYPE>:<DEVICE_INDEX>

其中：

- <JOB_NAME\>  是一个字母数字字符串，并且不以数字开头。
- <DEVICE_TYPE\>  是一种注册设备类型（例如 GPU 或 CPU）。
- <TASK_INDEX\>  是一个非负整数，表示名为 <JOB_NAME> 的作业中的任务的索引。请参阅 tf.train.ClusterSpec 了解作业和任务的说明。
- <DEVICE_INDEX\>  是一个非负整数，表示设备索引，例如用于区分同一进程中使用的不同 GPU 设备。

您无需指定设备规范的每个部分。例如，如果您在具有单个 GPU 的单机器配置中运行，您可以使用 tf.device 将一些操作固定到 CPU 和 GPU 上：
```python
# Operations created outside either context will run on the "best possible"
# device. For example, if you have a GPU and a CPU available, and the operation
# has a GPU implementation, TensorFlow will choose the GPU.
weights = tf.random_normal(...)

with tf.device("/device:CPU:0"):
  # Operations created in this context will be pinned to the CPU.
  img = tf.decode_jpeg(tf.read_file("img.jpg"))

with tf.device("/device:GPU:0"):
  # Operations created in this context will be pinned to the GPU.
  result = tf.matmul(weights, img)
```

如果您在典型的分布式配置中部署 TensorFlow，您可以指定作业名称和任务 ID，以便将变量放到参数服务器作业 ("/job:ps") 中的任务上，并将其他操作放置到工作器作业 ("/job:worker") 中的任务上：

```python
with tf.device(tf.train.replica_device_setter(ps_tasks=3)):
  # tf.Variable objects are, by default, placed on tasks in "/job:ps" in a
  # round-robin fashion.
  w_0 = tf.Variable(...)  # placed on "/job:ps/task:0"
  b_0 = tf.Variable(...)  # placed on "/job:ps/task:1"
  w_1 = tf.Variable(...)  # placed on "/job:ps/task:2"
  b_1 = tf.Variable(...)  # placed on "/job:ps/task:0"

  input_data = tf.placeholder(tf.float32)     # placed on "/job:worker"
  layer_0 = tf.matmul(input_data, w_0) + b_0  # placed on "/job:worker"
  layer_1 = tf.matmul(layer_0, w_1) + b_1     # placed on "/job:worker"
```

#### 7. Tensor_like 对象

许多 TensorFlow 操作都会接受一个或多个 tf.Tensor 对象作为参数。例如，tf.matmul 接受两个 tf.Tensor 对象，tf.add_n 接受一个具有 n 个 tf.Tensor 对象的列表。为了方便起见，这些函数将接受类张量对象来取代 tf.Tensor，并将它隐式转换为 tf.Tensor（通过 tf.convert_to_tensor 方法）。类张量对象包括以下类型的元素：

- tf.Tensor
- tf.Variable
- numpy.ndarray
- list
- python 标量类型 （bool、float、int、str）

###### 注意：默认情况下，每次您使用同一个类张量对象时，TensorFlow 将创建新的 tf.Tensor。如果类张量对象很大（例如包含一组训练样本的 numpy.ndarray），且您多次使用该对象，则可能会耗尽内存。要避免出现此问题，请在类张量对象上手动调用 tf.convert_to_tensor 一次，并使用返回的 tf.Tensor。

#### 8. 在 tf.Session 中执行图

TF使用 tf.Session 类来表示图运行的客户端程序。 tf.Session 对象使我们可以访问本地机器中的设备和使用远程设备进行运算图。它还可以缓存关于Graph的信息，是代码可以高效的运行在同一计算中。

创建 tf.Session: 
```python
# Create a default in-process session.
with tf.Session() as sess:
  # ...

# Create a remote session.
with tf.Session("grpc://example.org:2222"):
  # ...
```
由于 tf.Session 与物理资源想关联，所以通常在 with 代码块中创建，方便在图运算完成之时自动的关闭会话并释放资源。当然也可以不使用with代码块，但在运算结束时必须显示的调用tf.Session.close 以便释放资源。

tf.Session.init 接受三个可选参数：

- target。 如果将此参数留空（默认设置），会话将仅使用本地机器中的设备。但是，您也可以指定 grpc:// 网址，以便指定 TensorFlow 服务器的地址，这使得会话可以访问该服务器控制的机器上的所有设备。请参阅 tf.train.Server 以详细了解如何创建 TensorFlow 服务器。例如，在常见的图间复制配置中，tf.Session 连接到 tf.train.Server 的流程与客户端相同。分布式 TensorFlow 部署指南介绍了其他常见情形。

- graph。 默认情况下，新的 tf.Session 将绑定到当前的默认图，并且仅能够在当前的默认图中运行操作。如果您在程序中使用了多个图（更多详情请参阅使用多个图进行编程），则可以在构建会话时指定明确的 tf.Graph。

- config。 此参数允许您指定一个控制会话行为的 tf.ConfigProto。例如，部分配置选项包括：

    - allow_soft_placement。将此参数设置为 True 可启用“软”设备放置算法，该算法会忽略尝试将仅限 CPU 的操作分配到 GPU 设备上的 tf.device 注解，并将这些操作放置到 CPU 上。

    - cluster_def。使用分布式 TensorFlow 时，此选项允许您指定要在计算中使用的机器，并提供作业名称、任务索引和网络地址之间的映射。详情请参阅 tf.train.ClusterSpec.as_cluster_def。

    - graph_options.optimizer_options。在执行图之前使您能够控制 TensorFlow 对图实施的优化。

    - gpu_options.allow_growth。将此参数设置为 True 可更改 GPU 内存分配器，使该分配器逐渐增加分配的内存量，而不是在启动时分配掉大多数内存。

#### 9. 使用 tf.Session.run 执行操作

按下计算机的等号！

做到这一步基本上相当于我们已经作完了大部分的准备工作现在万事俱备只欠东风，使用 tf.Session.run 方法是运行 tf.Opeation 或者评估 tf.Tensor 的主要机制。

tf.Session.run 要求指定一组 fetch(这个fetch我认为就是要计算的图的边缘最靠近的那个operation，可以翻译为 计算子 ？)
，这些fetch可以返回确定的值，并且可能是tf.Operation、tf.Tensor 或者是Tensor_like 类型。 例如tf.Variable就是一个fetch。 这些 fetch 指明了要执行哪些子图来获得结果，子图包含了所有的节点运算过程。看一个荔枝：

```python
x = tf.constant([[37.0, -23.0], [1.0, 4.0]])
w = tf.Variable(tf.random_uniform([2, 2]))
y = tf.matmul(x, w)
y = tf.matmul(y, w)   # use y as the input of the y 
output = tf.nn.softmax(y)
init_op = w.initializer

with tf.Session() as sess:
  # Run the initializer on `w`.
  sess.run(init_op)

  # Evaluate `output`. `sess.run(output)` will return a NumPy array containing
  # the result of the computation.
  print(sess.run(output))

  # Evaluate `y` and `output`. Note that `y` will only be computed once, and its
  # result used both to return `y_val` and as an input to the `tf.nn.softmax()`
  # op. Both `y_val` and `output_val` will be NumPy arrays.
  y_val, output_val = sess.run([y, output])

```

在上面的荔枝中特别的写了一行：

    y = tf.matmul(y,w)

我的目的是，想使用y作为这个操作子的输入，我们可以发现上面的代码 output 执行了两次，但是我们得到的output却是相同的，这与我的预期不一样，我的预想是这样：

```python
x = 1
y = x + 1
y = y + 1 #第一次执行
print(y)  # >>> 3
y = y + 1 #第二次执行
print(y)  # >>> 4
```

为什么在TF程序里面得到的结果是相同的呢？
必须要注意到每次通过 tf.Session.run 都是将 fetch 的子图完全从头开始运行一遍，上面的荔枝可以看到，通过 sess.run(init_op) 使得w获得了初始化，在每次运行 output 这个节点的时候都是从 w 开始数据通过graph逐步传到了 output 所以与上一次的运算是毫无关系的。 

另一方面也必须注意到，在同一个 tf.Session.run 里面的所运行的图是同一性的，比如在上面的荔枝里面我们同时在一个 tf.Session.run里面运行了 y 和 output 但是实际上 y 只被运行了一次，他的结果一方面用来返回给 y_val 另一个方面被用于进一步的计算 output。

tf.Session.run 也可以接受 feed 字典，通常是在使用 tf.placeholder 时用于替代其值。这和我们在写一个代数表达式时设立的某些未知数x,a,b,c等类似，我们先用一个代数符号替代具体的计算值，等最后开始计算表达式时在把值带入到表达式中。 feed 字典接受的数据类型可以是 Python 标量、列表、Numpy数组。例如：

```python
# Define a placeholder that expects a vector of three floating-point values,
# and a computation that depends on it.
x = tf.placeholder(tf.float32, shape=[3])
y = tf.square(x)

with tf.Session() as sess:
  # Feeding a value changes the result that is returned when you evaluate `y`.
  print(sess.run(y, {x: [1.0, 2.0, 3.0]}))  # => "[1.0, 4.0, 9.0]"
  print(sess.run(y, {x: [0.0, 0.0, 5.0]}))  # => "[0.0, 0.0, 25.0]"

  # Raises <a href="../api_docs/python/tf/errors/InvalidArgumentError"><code>tf.errors.InvalidArgumentError</code></a>, because you must feed a value for
  # a `tf.placeholder()` when evaluating a tensor that depends on it.
  sess.run(y)

  # Raises `ValueError`, because the shape of `37.0` does not match the shape
  # of placeholder `x`.
  sess.run(y, {x: 37.0})

```

可以看到，feed 所接受的变量一定是和 tf.placeholder 的数据维度是一致的否者就会报错。

tf.Session.run 也接受可选的 options 参数（允许您指定与调用有关的选项）和可选的 run_metadata 参数（允许您收集与执行有关的元数据）。例如，您可以同时使用这些选项来收集与执行有关的跟踪信息：

```python
y = tf.matmul([[37.0, -23.0], [1.0, 4.0]], tf.random_uniform([2, 2]))

with tf.Session() as sess:
  # Define options for the `sess.run()` call.
  options = tf.RunOptions()
  options.output_partition_graphs = True
  options.trace_level = tf.RunOptions.FULL_TRACE

  # Define a container for the returned metadata.
  metadata = tf.RunMetadata()

  sess.run(y, options=options, run_metadata=metadata)

  # Print the subgraphs that executed on each device.
  print(metadata.partition_graphs)

  # Print the timings of each operation that executed.
  print(metadata.step_stats)
  ```

#### 10. 可视化你的图  

TF提供了种可视化计算图的模块 TensorBoard 的一个组件，可以在浏览器中可视化图的结构。 创建可视化图最简单的办法是传递 tf.Graph 到 tf.summary.FileWriter ：

```python
# Build your graph.
x = tf.constant([[37.0, -23.0], [1.0, 4.0]])
w = tf.Variable(tf.random_uniform([2, 2]))
y = tf.matmul(x, w)
# ...
loss = ...
train_op = tf.train.AdagradOptimizer(0.01).minimize(loss)

with tf.Session() as sess:
  # `sess.graph` provides access to the graph used in a <a href="../api_docs/python/tf/Session"><code>tf.Session</code></a>.
  writer = tf.summary.FileWriter("/tmp/log/...", sess.graph)

  # Perform your computation...
  for i in range(1000):
    sess.run(train_op)
    # ...

  writer.close() 
```

随后，您可以在 tensorboard 中打开日志并转到“图”标签，查看图结构的概要可视化图表。请注意，典型的 TensorFlow 图（尤其是具有自动计算的梯度的训练图）包含的节点太多，无法一次性完成直观展示。图可视化工具使用名称范围来将相关指令分组到“超级”节点中。您可以点击任意超级节点上的橙色“+”按钮以展开内部的子图。

更多的详细可视化操作参考 [TensorBoard](https://www.tensorflow.org/guide/summaries_and_tensorboard)

#### 11. 使用多个图进行编程




参考：

[tfensorflow官方文档](https://www.tensorflow.org/guide/)

**HAPPY CODING!**