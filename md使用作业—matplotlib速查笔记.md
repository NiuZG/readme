# numpy入门笔记
用于速成numpy以配合其他库的基础了解使用  
深入学习推荐三个网址  

[numpy官方中文文档](https://www.numpy.org.cn/user/quickstart.html#数组创建)  
[菜鸟教程|numpy](https://www.runoob.com/numpy/numpy-tutorial.html)

## 一、NumPy介绍---数组与矩阵的好基友
Numpy是Python中科学计算的基础包。提供多维数组对象，各种派生对象，以及用于数组快速操作的各种API，有包括数学、逻辑、形状操作、排序、选择、输入输出、离散傅立叶变换、基本线性代数，基本统计运算和随机模拟等等。
* 优点：Numpy以近C速度执行运算，且NumPy的语法简单！
* 两个特征：矢量化和广播。（大部分功能的基础）

灵活性使NumPy数组方言和NumPy ndarray 类成为在Python中使用的多维数据交换的首选对象。

NumPy 通常与 SciPy（Scientific Python）和 Matplotlib（绘图库）一起使用， 这种组合广泛用于替代 MatLab，是一个强大的科学计算环境，有助于我们通过 Python 学习数据科学或者机器学习。

## 二、基本概念
1. 轴: 在NumPy维度称为轴。即数组的“行”。
    
        3D空间中的点的坐标：
        [1, 2, 1]    具有一个轴。该轴有3个元素，所以我们说它的长度为3

        [[ 1., 0., 0.],
        [ 0., 1., 2.]]     数组有2个轴。第一轴的长度为2，第二轴的长度为3

2. N维数组对象 ndarray，用于存放同类型元素的多维数组。  
numpy.array与标准Python库类array.array不同，后者只处理一维数组并提供较少的功能。  

    ndarray对象的属性：
* ndarray.ndim - 数组的轴（维度）个数。Python中，维度数量称为rank.
* ndarray.shape - 数组的维度。整数元组表示每个维度中数组的大小。对于有 n 行和 m 列的矩阵，shape 将是 (n,m)。因此，shape 元组的长度就是rank或维度的个数 ndim。
* ndarray.size - 数组元素的总数。这等于 shape 的元素的乘积。
* ndarray.dtype - 描述数组中元素类型。可以使用标准的Python类型创建或指定dtype。NumPy提供它自己的类型。例如numpy.int32、numpy.int16和numpy.float64。
* ndarray.itemsize - 数组中每个元素的字节大小。等于 ndarray.dtype.itemsize 。
* ndarray.data - 该缓冲区包含数组的实际元素。通常不用，因为用索引访问数组元素。
  
例

    >>> import numpy as np
    >>> a = np.arange(15).reshape(3, 5)
    >>> a
    array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14]])
    >>> a.shape
    (3, 5)
    >>> a.ndim
    2
    >>> a.dtype.name
    'int64'
    >>> a.itemsize
    8
    >>> a.size
    15
    >>> type(a)
    <type 'numpy.ndarray'>
    >>> b = np.array([6, 7, 8])
    >>> b
    array([6, 7, 8])
    >>> type(b)
    <type 'numpy.ndarray'>

## 三、数组创建
    （1）新建数组
        numpy.array(object, dtype = None, copy = True, order = None, subok = False, ndmin = 0)
        
        参数说明：
        名称	描述
        object	数组或嵌套的数列
        dtype	数组元素的数据类型，可选
        copy	对象是否需要复制，可选
        order	创建数组的样式，C为行方向，F为列方向，A为任意方向（默认）
        subok	默认返回一个与基类类型一致的数组
        ndmin	指定生成数组的最小维度

    （2）已知数据转化为ndarray数组：
        numpy.asarray(a, dtype = None, order = None)

        x =  (1,2,3) 
        a = np.asarray(x)
        
        其他方法
        numpy.frombuffer 用于实现动态数组。接受输入参数，以流的形式读入转化
        numpy.fromiter 从可迭代对象中建立 ndarray 对象，返回一维数组

    （3）从数值范围创建数组
        1. 未知元素个数
            numpy.arange(start, stop, step, dtype)

            >>> np.arange( 0, 2, 0.3 )
            array([ 0. ,  0.3,  0.6,  0.9,  1.2,  1.5,  1.8])

                reshape函数：    
                    >>> c = np.arange(24).reshape(2,3,4)         # 3d array
                    >>> print(c)
                    [[[ 0  1  2  3]
                    [ 4  5  6  7]
                    [ 8  9 10 11]]
                    [[12 13 14 15]
                    [16 17 18 19]
                    [20 21 22 23]]]
    
        2. 已知元素个数
            np.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)

        参数说明：
        名称	   描述
        num	       要生成的等步长的样本数量，默认为50
        endpoint   该值为 true 时，数列中包含stop值，反之不包含，默认是True。
        retstep	   如果为 True 时，生成的数组中会显示间距，反之不显示。
            
          
            >>> np.linspace( 0, 2, 9 )                # 9 numbers from 0 to 2
            array([ 0. ,  0.25,  0.5 ,  0.75,  1.  ,  1.25,  1.5 ,  1.75,  2.  ])

    （4）创建函数数组！！！（绘图常用！！！）
        a = np.fromfunction(func, <shape>)
            func是函数
            shape相当于自变量数组集合
            a 返回相当于由因变量组成的数组
        
        比如9x9乘法表：
            def func(i, j):
                return (i+1)*(j+1)
            np.fromfunction(func, (9, 9))
关于函数数组更多了解见下资料  
[例子说明](https://blog.csdn.net/qq_28618765/article/details/78085793)  
[原理说明](https://www.zhihu.com/tardis/landing/360/ans/276643251?query=np.fromfunction&mid=f51a5bef7e016aa0a70a2cd7a4676cad&guid=ABB23B9795103C646B636BE24C7B9210.1581772482107)


例举几种创建数组方法：

    1. 一维数组
        你可以使用array函数从常规Python列表或元组中创建数组。得到的数组的类型是从Python列表中元素的类型推导出来的。
            >>> import numpy as np
            >>> a = np.array([2,3,4])
            >>> a
            array([2, 3, 4])
            >>> a.dtype
            dtype('int64')
            >>> b = np.array([1.2, 3.5, 5.1])
            >>> b.dtype
            dtype('float64')

        常见的错误：调用array时传入多个数字参数，而不是提供单个数字的列表作参数：
            >>> a = np.array(1,2,3,4)    # WRONG
            >>> a = np.array([1,2,3,4])  # RIGHT
        或者如下解决：（推荐）
            x =  (1,2,3) 
            a = np.asarray(x)

    2. 多维数组
        将序列的序列转换成二维数组，将序列的序列的序列转换成三维数组等等。
            >>> b = np.array([(1.5,2,3), (4,5,6)])
            >>> b
            array([[ 1.5,  2. ,  3. ],
                [ 4. ,  5. ,  6. ]])

    3. 指定数组的类型
            >>> c = np.array( [ [1,2], [3,4] ], dtype=complex )
            >>> c
            array([[ 1.+0.j,  2.+0.j],
                [ 3.+0.j,  4.+0.j]])

    4. 创建未知元素已知大小的数组
        * zeros创建一个由0组成的数组，
        * ones创建一个完整的数组，
        * empty 创建一个数组，其初始内容是随机的，取决于内存的状态。
        默认情况下，创建的数组的dtype是 float64 类型的。
            >>> np.zeros( (3,4) )
            array([[ 0.,  0.,  0.,  0.],
                   [ 0.,  0.,  0.,  0.],
                   [ 0.,  0.,  0.,  0.]])
            >>> np.ones( (2,3,4), dtype=np.int16 )  
            array([[[ 1, 1, 1, 1],
                    [ 1, 1, 1, 1],
                    [ 1, 1, 1, 1]],
                   [[ 1, 1, 1, 1],
                    [ 1, 1, 1, 1],
                    [ 1, 1, 1, 1]]], dtype=int16)
            >>> np.empty( (2,3) ) 
            array([[  3.73603959e-262,   6.02658058e-154,   6.55490914e-260],
                   [  5.30498948e-313,   3.14673309e-307,   1.00000000e+000]])
            


## 四、切片和索引
ndarray对象的内容可以通过索引或切片来访问和修改，与 Python 中 list 的切片操作一样。

ndarray 数组可以基于 0 - n 的下标进行索引，切片对象可以通过内置的 slice 函数，并设置 start, stop 及 step 参数进行，从原数组中切割出一个新数组。

一维的数组可以进行索引、切片和迭代操作的，就像 列表 和其他Python序列类型一样。

             ”索引区间前闭后开！！！“
    a[2]                索引第三个元素，结果是数
    a[2:5]              索引3到6的元素，结果是数组
    a[:6:2] = -1000     从第1个开始到7个元素依次间隔2位赋值为-1000
    a[ : :-1]           倒序排列
    for i in a:         遍历元素处理
        print(i**(1/3.))

多维的数组每个轴可以有一个索引。这些索引以逗号​​分隔的元组给出：

    >>> def f(x,y):
    ...     return 10*x+y
    ...
    >>> b = np.fromfunction(f,(5,4),dtype=int)
    >>> b
    array([[ 0,  1,  2,  3],
        [10, 11, 12, 13],
        [20, 21, 22, 23],
        [30, 31, 32, 33],
        [40, 41, 42, 43]])
    >>> b[2,3]
    23
    >>> b[0:5, 1]                       # each row in the second column of b
    array([ 1, 11, 21, 31, 41])
    >>> b[ : ,1]                        # equivalent to the previous example
    array([ 1, 11, 21, 31, 41])
    >>> b[1:3, : ]                      # each column in the second and third row of b
    array([[10, 11, 12, 13],
        [20, 21, 22, 23]])

当提供的索引少于轴的数量时，缺失的索引被认为是完整的切片:

    b[-1]              最后一行，等于b[-1,:]
    x[1,2,...]         相当于 x[1,2,:,:,:]，（假设5轴）
    x[...,3]           等效于 x[:,:,:,:,3]
    x[4,...,5,:]       等效于 x[4,:,:,5,:]


##  五、运算操作
    a @ b	             矩阵乘法
    a * b	             元素乘法
    a/b	                 元素除法
    a**3	             元素取3次方
    a.max()	             最大元素a
    a.max(0)	         每列矩阵的最大元素 a
    a.max(1)	         每行矩阵的最大元素 a
    maximum(a, b)	     比较a和b逐个元素，并返回每对中的最大值
    a.transpose()        或 a.T	，转置 a
    a.conj().transpose() 或 a.conj().T	， 共轭转置 a

    通函数：
    NumPy提供熟悉的数学函数，例如sin，cos和exp。被称为“通函数”（ufunc）。这些函数在数组上按元素进行运算，产生一个数组作为输出。
    B = np.arange(3)
    np.exp(B)
    np.sqrt(B)
    np.add(B, C)
   
## 六、迭代，形状操纵，数组堆叠拆分，及其他基础知识
以上知识作为一般入门了解已经足够，所以其他使用  
[见开篇所附中文手册](https://www.numpy.org.cn/user/quickstart.html#形状操纵)