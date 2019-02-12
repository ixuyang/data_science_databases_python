'''
Created on 2019年1月9日

@author: UPXY
'''
import numpy as np
from numpy.core.fromnumeric import size
'''
课时3：Array数组
'''
array = [1,2,3,4,5]
# array + 1 无法执行这样直接的操作
array = np.array([1,2,3,4,5])
# <class 'numpy.ndarray'>数据类型发生了改变，变为ndarray类型
print(type(array))
array1 = array + 1
print(array1)
# 可以正常显示结果了：[2 3 4 5 6]
print(array1 + array)
# 可以快速的显示数字的大小情况，例如这里是5个元素，显示结果为：(5,)
print(array.shape)
array2 = np.array([[1,2,3],[2,3,4]])
# 多维数组，显示结果如下：
# [[1 2 3]
#  [2 3 4]]
# (2, 3)
print(array2)
print(array2.shape)
# .dtype可以显示数组中的元素的数据类型，
# 并且要求数据类型都是统一的，如果出现了一个元素的数据类型是字符，那么全部会转换为字符
print(array2.dtype)
# 查看每个元素占了几个字节
print(array2.itemsize)
# 查看数组中一共有多少个元素，不关心维度结构
print(array2.size)
# 对二维数组进行切片处理，如下，获取第二个数组记录的第三个元素
print(array2[1][2:3])

'''
课时4：数组结构
'''
# 矩阵格式（多维的形式）
zhang_array = np.array([[1,2,3],
                        [4,5,6],
                        [7,8,9]])
print(zhang_array)
print(zhang_array.shape)
print(zhang_array.size)
# 输出有几个维度，这里是两个维度
print(zhang_array.ndim)
# 取矩阵中的一行，如下为第二行，从零开始
print(zhang_array[1])
# 取矩阵中的一列，如下第二列，从零开始
print(zhang_array[:,1])
# 取矩阵中的某一个值，例如取5这个数值
print(zhang_array[1,1])
print(zhang_array[1][1])
# 直接赋值运算后的矩阵，新的名称只是执行原名称矩阵数据的位置，并没有新建，如果改变数值都会改变
zhang_array2 = zhang_array
# 这样copy的复制是完全全的复制，重新建立的array，完全不一样了，操作不会干扰
zhang_array2 = zhang_array.copy()
# 构建等差数列，[ 0 10 20 30 40 50 60 70 80 90],可以使用arange
zhang_array = np.arange(0,100,10)
print(zhang_array)

# 建立索引,
mask = np.array([0,0,0,1,1,1,0,0,1,1],dtype=bool)
print(mask)
print(zhang_array[mask])
random_array = np.random.rand(10)
mask = random_array > 0.5
print(mask)
print(zhang_array[mask])
tang_array = np.array([10,20,30,40,50])
# 输出数值中哪些数据大于30，并找出位置具体索引，(array([3, 4], dtype=int32),)
print(np.where(tang_array > 30))
# 找到数组中对应的值,[40 50]
print(tang_array[np.where(tang_array > 30)])
'''
课时5：数组类型
'''
# 设置数据中元素的类型，[1. 2. 3. 4. 5.] float32
tang_array = np.array([1,2,3,4,5],dtype=np.float32)
print(tang_array,tang_array.dtype,tang_array.nbytes)
# 设置数组中的元素样式类型不同也是可以的，object不要求统一的数据类型
tang_array = np.array([1,2,40,'str'],dtype=object)
print(tang_array)
# 使用np.asarray进行数组数据类型的统一，但是通过赋值新的变量名不会影响原数组，相当于建立了新的数组。ps:astype可以直接对数组的类型进行变化
tang_array2 = np.asarray(tang_array, dtype=np.str)
print(tang_array,tang_array2)
'''
课时6：array数组的数值计算
'''
tang_array = np.array([[1,2,3],[3,4,5]])
# axis可以指定按照列还是行来求和，0表示第一个轴是数轴，为列。
print(tang_array,np.sum(tang_array,axis=0),tang_array.sum(axis=-1))
# 实现乘法运算，内部元素按照要求进行乘法运算prod，还有最小值和最大值，min和max,同时还可以确认最小值的位置索引
print(tang_array.prod(axis=0),tang_array.argmin())
# 计算标准差使用std,计算方差使用var
print(tang_array.std(),tang_array.var())
# 让数组中小于某个值的都变成某个值，大于某个值的都变成某个值clip,小于2的都变成2，大于4的都变成4
print(tang_array.clip(2,4))
# 对数组进行四舍五入，使用round，使用decimals参数控制精确到哪个小数点
tang_array = np.array([1.2,3.56,4.8,7.1])
print(tang_array.round())
'''
课时7：排序操作
'''
# 排序使用sort,可以指定排序轴，默认是0轴
tang_array = np.array([[1.2,3.56],[4.8,7.1]])
print(np.sort(tang_array,axis=0))
# 使用searchsorted来进行数组中插入数组，数组刚好嵌入到数组中间，并返回对应位置
# 分开不同列或者行进行排序，使用lexsort
'''
课时8：数组形状操作
'''
tang_array = np.arange(10)
# 把一维数组改造成为二维数组，或者使用reshape()函数，但是需要保证大小是支持变化的
tang_array.shape = 2,5
# 给数组增加一个维度
# tang_array = tang_array[:,np.newaxis]
# 压缩数组，去除掉不需要的空值
tang_array.squeeze()
# 转置数组操作，行和列进行转换,或者使用tang_array.T
tang_array = tang_array.transpose()
# 数组的拼接使用np.concatenate((数组1，数组2)，axis来指定拼接维度)
# c = np.concatenate((a,b))
# 把数组拉平
tang_array = tang_array.flatten()
print(tang_array,tang_array.shape)
'''
课时9：数组生成函数
'''
# 表示从2开始到20大小，每个数字隔3，数据类型是整形
tang_array = np.arange(2,20,3,dtype = np.int32)
# 表示是从0开始到10，构造10个数,等距的分布
tang_array2 = np.linspace(0,10,10)
print(tang_array,tang_array2)
'''
课时10：常用生成函数
'''
# 构建长度为3的都是0数值的数组或者是1的
tang_array = np.zeros((3,3),dtype = np.int32)
tang_array2 = np.ones((2,2),dtype = np.int32)
# 构造一个数组都为1，与指定数组的形状一致，如下
tang_array3 = np.ones_like(tang_array)
print(tang_array,tang_array2,tang_array3)
'''
课时11：四则运算
'''
x = np.asarray([5,5])
y = np.asarray([2,1])
# 数组乘法运算,为对应位置相乘[10  5]
print(np.multiply(x,y))
# 矩阵运算为交叉相乘后求和,15。但是必须保证维度的对应关系是可以进行矩阵乘法运算的。
print(np.dot(x,y))
x = np.zeros((1,2),dtype = np.int32)
y = np.ones((1,2),dtype = np.int32)
# 可以进行逻辑运算，这里比如直接对比是否相等来逻辑判断[[False False]]
print(x,y,x == y)
# 进行与操作，使用如下[[False False]]
print(np.logical_and(x,y))
'''
课时12：随机模块
'''
# 构建随机数组，里面所有的值都是从0到1的，如下构建3行2列的随机浮点数数组
print(np.random.rand(3,2),np.random.rand())
# 构建整数的随机的数组
print(np.random.randint(10,size = (5,3)))
# 从0到10之间获取三个随机数数，[6 9 6]
print(np.random.randint(0,10,3))
# 取值单个随机数,可以使用一下的办法
print(np.random.rand(),np.random.sample())
# 设置输出的小树点后两位
np.set_printoptions(precision = 2)
print(np.random.normal(0,0.1,10))
# 洗牌操作,对有顺序的数组进行洗牌打散，每次洗牌的结果都是不一致的
tang_array3 = np.arange(10)
print(tang_array3)
np.random.shuffle(tang_array3)
print(tang_array3)
# 随机的种子，让每次随机的模式是一样的结果。seed()括号中的数字是指定的种子，只要其不变随机的结果就不变
np.random.seed(1)
print(np.random.normal(0,0.1,10))
'''
课时13：文件读写
'''
data = []
with open('tang.txt') as f:
    for line in f:
        fileds = line.split()
        cur_data = [float(x) for x in fileds]
        data.append(cur_data)
data = np.array(data)
print(data)
# 更快捷的办法，默认以空格为分隔符，如果是其他的话，
# 需要参数指定分隔符，delimiter = ','
# 如果要跳过第一行增加参数，skiprows = 1
# 使用指定的列需要增加参数，usecols = (0,1,4)
data1 = np.loadtxt('tang.txt')
print(data1)
'''
课时14：数组保存写文件
'''
tang_array = np.array([[1,2,3],[4,5,6]])
# fmt参数来指定是整数还是其他的类型
np.savetxt('tang1.txt',tang_array,fmt='%d',delimiter = ',')
# 读写array结构
np.save('tang_array.npy',tang_array)
tang = np.load('tang_array.npy')
print(tang)
# 压缩存储多个数组的时候可以使使用npz的文件来保存，可以保存多个
'''
Numpy练习题
'''
# 打印当前Numpy版本
print(np.__version__)
# 构造一个全零的矩阵，并打印其占用内存大小
tang_array = np.zeros((5,6),dtype = np.int32)
print('%d bytes'%(tang_array.size*tang_array.itemsize))
# 打印一个函数的帮助文档，比如numpy.add()
# print(help(np.info(np.add)))
# 创建一个10-49的数组，并将其倒序排列
tang_array = np.arange(10,50,dtype = np.int32)
tang_array = tang_array[::-1]
print(tang_array)
# 找到一个数组中不为0的索引
tang_array = np.array([1,0,2,3,0,6])
print(np.where(tang_array!=0))

print(np.nonzero(tang_array))
# 随机构造一个3*3的矩阵，并打印其中最大和最小值
tang_array = np.random.randint(10,size=(3,3))
print(tang_array,np.max(tang_array),tang_array.min())
# 构造一个5*5的矩阵，另其值为1，并在最外层加上一圈0,ps:这个主要服务于未来图形的构建
tang_array = np.ones((5,5),dtype = np.int32)
tang_array = np.pad(tang_array, pad_width = 1, mode = 'constant',constant_values = 0)
print(tang_array)
# 构建一个shape为（6,7,8）的矩阵，并找到第100个元素的索引值
index = np.unravel_index(100,(6,7,8))
print(index)
# 对一个5*5的矩阵做归一化操作
tang_array = np.random.randint(10,size=(3,3))
tang_array = (tang_array - tang_array.min())/(tang_array.max()-tang_array.min())
print(tang_array)
# 找到两个数组中相同的值
tang_array1 = np.random.randint(10,size=(3,3))
tang_array2 = np.random.randint(10,size=(3,3))
print(np.intersect1d(tang_array1,tang_array2))
# 得到今天 明天 昨天的日期
print(np.datetime64('today','D'),np.datetime64('today','D')-np.timedelta64(1,'D'),np.datetime64('today','D')+np.timedelta64(1,'D'))
# 得到一个月中所有的天
date_array = np.arange('2017-10','2017-11',dtype='datetime64[D]')
print(date_array)
# 得到一个数的整数部分
tang_array = np.random.uniform(0,10,10)
print(np.floor(tang_array))
# 构造一个数组，让它不能改变
tang_array = np.zeros(5)
tang_array.flags.writeable = False
# tang_array[0] = 1 这里就不可以修改了，ValueError: assignment destination is read-only
# 打印数据量大的数据的部分值，全部值
np.set_printoptions(threshold=5)
z = np.zeros((15,15))
print(z)
np.set_printoptions(threshold=np.nan)
z = np.zeros((15,15))
print(z)
# tang_array = np.random.randint((6,7,8),dtype=np.int32)
# 找到一个数组中，最接近一个数的索引
z = np.arange(100)
v = np.random.uniform(0,100)
print(v)
index = (np.abs(z-v)).argmin()
print(z[index])
# 32位float类型和32位int类型转换
z = np.arange(10,dtype=np.int32)
print(z.dtype)
z = z.astype(np.float32)
print(z.dtype)
# 打印数组元素位置坐标与数值
z = np.arange(9).reshape(3,3)
for index,value in np.ndenumerate(z):
    print(index,value)
# 按照数组的某一列进行排序
z = np.random.randint(0,10,(3,3))
print(z)
print(z[z[:,1].argsort()])
# 统计数组中每个数值出现的次数，显示的结果表示对于数字序列中出现数值的次数
z = np.array([1,2,3,11,1,1,1,1,2,2,3])
print(np.bincount(z))
# 如何对一个四维数组的最后两维来求和
z = np.random.randint(0,10,(4,4,4,4))
res = z.sum(axis=(-2,-1))
print(res)
# 交换矩阵中的两行
z = np.arange(25).reshape(5,5)
z[[0,1]] = z[[1,0]]
print(z)
# 找到一个数组中最常出现的数字
z = np.array([1,2,3,11,1,1,1,1,2,2,3])
print(np.bincount(z).argmax())
# 快速查找TOP K
z = np.arange(10000)
np.random.shuffle(z)
n = 5
print(z[np.argpartition(-z,n)[:n]])
# 去除掉一个数组中，所有元素都相同的数据
z = np.random.randint(0,5,(10,3))
e = np.all(z[:,1:] == z[:,:-1],axis=1)
e = np.any(z[:,1:] == z[:,:-1],axis=1)
print(e)
