'''
Created on 2019年1月9日

@author: UPXY
'''
import numpy as np
'''
课时2：Array数组
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
