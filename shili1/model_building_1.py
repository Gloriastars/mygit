"""
python运行gprmax
读取.in文件
运行api函数模拟
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from gprMax.gprMax import api
from tools.outputfiles_merge import get_output_data, merge_files
import scipy.io as np


# # 文件路径+文件名
# dmax = r".\GprmaxCode"  # 项目目录
# filename = os.path.join(dmax, 'model_2.in')
# # 正演  n：仿真次数（A扫描次数）->B扫描
# api(filename, n=230, geometry_only=False)  # geometry_only：仅几何图形
# merge_files(r".\GprmaxCode\model_2", removefiles=False)
#
# 获取回波数据
# A B扫描时out文件名不一样
filename = os.path.join(r".\GprmaxCode\model_2_merged.out")
rxnumber = 1
rxcomponent = 'Ez'
outputdata, dt = get_output_data(filename, rxnumber, rxcomponent)

# 保存回波数据
np.savetxt('model_2.txt', outputdata, delimiter=' ')


#  #平均抵消法去直达波
# where = outputdata.argmax(axis=0)
# print(where)
# outputdata[1:322, ] = 0
# # Remove index 2 from previous array
# print(np.delete(b, 2, axis=0))

# 将文件中数据加载到data数组里
# data_array = np.loadtxt('quzhidabo.txt')
# print(data_array)
#     '''导入数据
#     input:  file_path(string):文件的存储位置
#     output: data(mat):数据
#     '''
#     f = open(r"C:\Users\86153\PycharmProjects\shili1\filed1.txt")
#     data = []
#     for line in f.readlines():
#         row = []  # 记录每一行
#         lines = line.strip().split("\t")
#         for x in lines:
#             row.append(float(x)) # 将文本中的特征转换成浮点数
#         data.append(row)
#     f.close()
# #先将matlab中的数据存为matl类型

matfn=u'C:\Users\86153\PycharmProjects\shili1\filed1.mat'
data=sio.loadmat(matfn) #读取mat数据，转化为dict
# LM=data['L']; #读取字典中的稀疏矩阵，LM在python变量中
# # B扫描绘图
from tools.plot_Bscan import mpl_plot
plt = mpl_plot(filename, data, dt*1e9, rxnumber, rxcomponent)
plt.ylabel('Time [ns]')
plt.show()



# # B扫描绘图
# from tools.plot_Bscan import mpl_plot
# plt = mpl_plot(filename,outputdata, dt*1e9, rxnumber, rxcomponent)
# plt.ylabel('Time [ns]')
# plt.show()





# # A扫描绘图
# from tools.plot_Ascan import mpl_plot
# from gprMax.receivers import Rx
# outputs = Rx.defaultoutputs
# outputs = ['Ez']
# print(outputs)
# plt = mpl_plot(filename, outputs)
# plt.show()


# ## A扫描图
# outputdata[1:200,]=0    ## 通过置零消除天线耦合波
# output = outputdata[:,0]  # 第i道A扫信号：序号从0开始
# plt.plot(output)
# plt.show()


# ## 堆叠波形
# space_signal = 100   # 信号间隔(按实际情况变更)
# tw = 14              # 时间窗（与in文件一致）
# trace_number = len(outputdata[0])
# for i in range(trace_number):
#     plt.plot(outputdata[:,i]+(i+1)*space_signal,np.linspace(0,tw,len(outputdata)),color='m')
# plt.xticks(range(space_signal,trace_number*space_signal+1,space_signal),range(1,trace_number+1))
# plt.xlim(0, space_signal*(trace_number+2))
# plt.ylim(0, tw)
# plt.xlabel('trace_number')
# plt.ylabel('Time [ns]')
# ax = plt.gca()          # 获取句柄
# ax.invert_yaxis()       # y轴反向
# ax.xaxis.tick_top()     # x轴放在上方
# plt.show()



