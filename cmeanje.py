import numpy as np
import os
import xlrd
import random
import matplotlib.pyplot as plt
 
#输入表格名file_data
# 要读该列的行数从1到row_end
def ReadInData(file_data, LableClassNum, ClassNum, row_end):
    LableClassNum = LableClassNum+1
    row_end = row_end+2
    Data = [[i for i in range(len(ClassNum))] for i in range(row_end-2)]
    Lable = []
    ClassLenth = len(ClassNum)
    Data = np.array(Data, dtype='float32')
    for i in range(2, row_end):
        Lable.append(file_data.cell(i, LableClassNum).value)
        for j in range(ClassLenth):
            Data[i-2][j] = file_data.cell(i,ClassNum[j]+1).value
    # 读入数据归一化处理
    Mean = []
    for i in range(row_end - 2):
        sum = 0
        k = 0
        for j in range(ClassLenth):
            sum = sum + Data[i][j]
            k = k + 1
        Mean.append(sum / k)
    for i in range(row_end - 2):
        for j in range(ClassLenth):
            Data[i][j] = Data[i][j] / Mean[j]
    return Data,np.array(Lable)
 
# Func = 1 获取N个随机分类中心
# Func = 0 以样本矩阵主特征值所在列对应的特征作为一维轴，将数值由大到小排序，进行N等分得到N+1个中心点
def C_Center(Data, Lable, N, Func):
    Center_Data  = []
    Center_Lable = []
    Center_Position = []
    Lenth = np.shape(Data)[0]
    SufN  = np.shape(Data)[1]
    if Func:
        Ranlist = random.sample(range(0, Lenth), N)
        Center_Position = Ranlist
        for i in range(N):
            Center_Data.append(Data[Ranlist[i]].tolist())
            Center_Lable.append(Lable[Ranlist[i]].tolist())
    else:
        Data_Scale = np.zeros((Lenth, SufN))
        for i in range(SufN):
            SufN_Max = np.max(Data[:, i])
            SufN_Min = np.min(Data[:, i])
            SufN_Mean = np.mean(Data[:, i])
            for j in range(Lenth):
                Data_Scale[j][i] = round((Data[j][i] - SufN_Mean) / (SufN_Max - SufN_Min) + 0.5, 1)
        EigValue, EigVector = np.linalg.eig(np.cov(np.transpose(Data_Scale)))
        Max_EigValue = sorted(range(len(EigValue)), key=lambda k: EigValue[k], reverse=True)
        Center = Data_Scale[:, Max_EigValue[0]]
        Max_CenterData = sorted(range(len(Center)), key=lambda k: Center[k], reverse=True)
        if N-1 == 0:
            Scale = 0
        else:
            Scale = len(Max_CenterData)/(N-1)
        Temp = 0
        for i in range(N):
            if i <1:
                where = Max_CenterData[0]
            else:
                where = Max_CenterData[int(Temp)-1]
            Center_Position.append(where)
            Center_Data.append(Data[where].tolist())
            Center_Lable.append(Lable[where])
            if i<N-1:
                Temp = np.ceil(Temp+Scale)
            else:
                Temp = Scale*(N-1)-1
    return Center_Data, Center_Lable, Center_Position
 
#计算两点的欧氏距离
def Eucldist(coords1, coords2):
    coords1 = np.array(coords1)
    coords2 = np.array(coords2)
    return np.sqrt(np.sum((coords1 - coords2)**2))
 
#更新各个点到中心点的欧氏距离
def CentDis(Center, Data, N):
    Long = np.shape(Data)[0]
    CenterDis = [[0 for i in range(N)] for i in range(Long)]
    for i in range(Long):
        for j in range(N):
            CenterDis[i][j] = Eucldist(Data[i],Center[j])
    return CenterDis
 
#得到列表中最小值对应的序号
def MinOlder(Data):
    Length = len(Data)
    if Length <= 1:
       return 0
    return sorted(range(Length), key=lambda k: Data[k], reverse=True)[Length - 1]
#得到列表中随机序号
def RandOlder(Data):
    Length = len(Data)
    if Length <= 1:
        return 0
    return random.randint(0,Length-1)
 
#计算类样本点平均值
def ClassMean(Data, ClassNum):
    Mean = [[0 for i in range(np.shape(Data[0])[1])] for i in range(ClassNum)]
    for i in range(ClassNum):
        if np.shape(Data[i])[0] == 0:
            Mean[i][0] = 0
            continue
        m, n = np.shape(Data[i])
        for j in range(n):
            sum = 0
            for k in range(m):
                sum = sum + Data[i][k][j]
            Mean[i][j] = sum/m
    return Mean
 
# 更新准则函数的值，看是否继续缩小
def Je(ClassData,Center_Num,NewCenter):
    sum2 = 0
    for i in range(Center_Num):
        Class = ClassData[i]
        sum1 = 0
        for j in range(np.shape(Class)[0]):
            sum1 = sum1 + pow(np.array(Class[j])-np.array(NewCenter[i]),2)
        sum2 = sum2 + sum1
    return sum(sum2)
 
# 计算每一点与其他类别中心的距离，如果比该点和原类别中心的距离短，
# 则移动该点至新的类别
def PointMoving(ClassData,Center_Num,Center):
    for i in range(Center_Num):
        for j in range(np.shape(ClassData[i])[0]):
            if j < np.shape(ClassData[i])[0]:
                ClassOrder = [i for i in range(Center_Num)]
                ClassOrder.remove(i)  # 得到除特征点所属类别之外的类别序号
                Nk = np.shape(ClassData[i])[0]  # 原类别特征值个数
                # 重新计算原类别代价函数
                Pk = 0
                if Nk - 1 == 0:
                    Pk = 0
                else:
                    F = sum(pow(np.array(ClassData[i][j]) - np.array(Center[i]), 2))  # 计算二范数
                    Pk = Nk / (Nk - 1) * F
                Pj = [i for i in range(Center_Num - 1)]
                for k in range(Center_Num-1):
                    Nj = np.shape(ClassData[ClassOrder[k]])[0] #新类别特征值个数
                    F = sum(pow(np.array(ClassData[i][j]) - np.array(Center[ClassOrder[k]]), 2))  # 计算二范数
                    #重新计算新类别代价函数
                    Pj[k] = Nj/(Nj+1)*F
                # 得到该特征点除原类别外,在移动该点后与其他最近的类别是哪个即Pj最小
                PjMinNum = MinOlder(Pj)
                MiniPj = ClassOrder[PjMinNum]
                if Pj[PjMinNum] < Pk:
                    ClassData[MiniPj].append(ClassData[i][j])  # 添加该特征点至新类别
                    ClassData[i].remove(ClassData[i][j])  # 在原类别里删除该特征点
                #print(i,j,"  ",np.shape(ClassData[0])[0],np.shape(ClassData[1])[0],np.shape(ClassData[2])[0],np.shape(ClassData[3])[0])
                #print(Pk-Pj[PjMinNum])
    return ClassData
 
def main():
    # 数据打开文件路径
    Tain_set = xlrd.open_workbook(os.path.abspath('作业数据_2021合成.xls'))
    # 训练数据读取
    Tain_sheet = Tain_set["Sheet1"]
    # 1性别 2籍贯 3身高 4体重  5鞋码  6(50米成绩) 7肺活量     8喜欢颜色 9喜欢运动 10喜欢文学
    Data_Num = 200 # 提取 208 个人信息
    WhichClass = [3, 4, 6]
    Data, Lable = ReadInData(Tain_sheet, 1, WhichClass, Data_Num)
 
    X = []
    Y = []
    for k in range(20):
        print(k)
        ################################<C均值分类算法>####################################
        # 设聚类中心4个
        Center_Num = k+1
        # 设定不同类别的点池
        ClassData = [[] for i in range(Center_Num)]
        # 使用方法0，得到初始聚类中心
        Center_Data, Center_Lable, Center_Position = C_Center(Data, Lable, Center_Num,1)
        # 得到各点与中心点的距离
        Distance = CentDis(Center_Data, Data, Center_Num)
        # 得到每点所属类别
        #MinDistance = [RandOlder(Distance[i]) for i in range(Data_Num)] # 初始随机分类
        MinDistance = [MinOlder(Distance[i]) for i in range(Data_Num)]  # 初始最小距离分类
        # 将每个点分到所属的类里面
        for i in range(Data_Num):
            ClassData[MinDistance[i]].append(Data[i].tolist())
 
        if k < 1:
            sum2 = 0
            for i in range(Center_Num):
                Class = ClassData[i]
                sum1 = 0
                for j in range(np.shape(Class)[0]):
                    sum1 = sum1 + pow(np.array(Class[j]) - np.array(Center_Data[i]), 2)
                sum2 = sum2 + sum1
            X.append(1)
            Y.append(sum(sum2))
        else:
            # 类别特征点初始化
            NewClassData = ClassData
            # 初始化准则函数值
            OldJe = 3
            NewJe = 2
            RunTime = 0
            # 开始循环使准则函数Je最小
            while OldJe - NewJe > 0.0000001:
                # for j in range(10):
                # 得到新的类别中心
                NewCenter = ClassMean(NewClassData, Center_Num)
                # 更新准则函数值
                if RunTime > 0:
                    OldJe = NewJe
                NewJe = Je(NewClassData, Center_Num, NewCenter)
                # 更新类的点
                NewClassData = PointMoving(NewClassData, Center_Num, NewCenter)
                RunTime = RunTime + 1  # 调制次数统计
            X.append(k)
            Y.append(NewJe)
    print(Y)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(X, Y)
    ax.set_title("Je Trend Line")
    ax.set_xlabel("ClassNum - C")
    ax.set_ylabel("Je")
    plt.show()
 
 
if __name__ == '__main__':
    main()