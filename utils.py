import numpy as np
import pandas as pd
import plotly
import plotly.graph_objs as go

def normal(x,Max,Min):
    x = (x - Min) / (Max - Min)
    return x

def plot4dwitha(data,file_name):       # data第一个数据为颜色
    cen=[data[1].mean(),data[2].mean(),data[3].mean()]
    cen=np.array(cen)  
    vector = go.Scatter3d(x=[0, cen[0]], y=[0,cen[1]], z=[0,cen[2]],
                          marker=dict(size=1, color="rgb(84,48,5)"), line=dict(color="rgb(84,48,5)", width=6)
                          )
    markersize = normal(data[4],10,4)       #体重
    #Make Plotly figure
    colorsize = (data[0]+ 1) * 8
    fig1 = go.Scatter3d(x=data[1],
                        y=data[2],
                        z=data[3],
                        marker=dict(size=markersize,
                                    color=colorsize,
                                    opacity=0.9,
                                    reversescale=True,
                                    colorscale='Viridis'),
                        line=dict (width=0.02),
                        mode='markers')

    #Make Plot.ly Layout
    point=[fig1,vector]
    mylayout = go.Layout(scene=dict(xaxis=dict( title="肺活量"),
                                    yaxis=dict( title="50米成绩"),
                                    zaxis=dict(title="身高")),
                         margin=dict(l=0, r=0,  b=0, t=0)
                         )
    #Plot and save html
    plotly.offline.plot({"data": point,
                         "layout": mylayout},
                         auto_open=True,
                         filename=file_name)

def plotknn(features,lable,prepoint,file_name):       # data第一个数据为颜色
    point=[prepoint[0][1],prepoint[0][2],prepoint[0][3]]
    point=np.array(prepoint)
    point_pre = go.Scatter3d(x=point[0], y=point[1], z=point[2],
                          marker=dict(size=1, 
                                      color="rgb(84,48,5)"), 
                          line=dict(color="rgb(84,48,5)", width=6),
                          mode='markers'
                          )
    markersize = normal(features[3],10,4)  # 体重
    #Make Plotly figure
    colorsize=(lable+1)*8
    fig1 = go.Scatter3d(x=features[0],
                        y=features[1],
                        z=features[2],
                        marker=dict(size=markersize,
                                    color=colorsize,
                                    opacity=0.9,
                                    reversescale=True,
                                    colorscale='Viridis'),
                        line=dict (width=0.02),
                        mode='markers')

    #Make Plot.ly Layout
    point=[fig1,point_pre]
    mylayout = go.Layout(scene=dict(xaxis=dict( title="肺活量"),
                                    yaxis=dict( title="50米成绩"),
                                    zaxis=dict(title="身高")),
                         margin=dict(l=0, r=0,  b=0, t=0)
                         )
    #Plot and save html
    plotly.offline.plot({"data": point,
                         "layout": mylayout},
                         auto_open=True,
                         filename=file_name)

def plot4d(data,file_name):       # data第一个数据为颜色
    markersize = normal(data[4],10,4)       #体重
    #Make Plotly figure
    colorsize = (data[0]+ 1) * 8
    fig1 = go.Scatter3d(x=data[1],
                        y=data[2],
                        z=data[3],
                        marker=dict(size=markersize,
                                    color=colorsize,
                                    opacity=0.9,
                                    reversescale=True,
                                    colorscale='Cividis'),
                        line=dict (width=0.02),
                        mode='markers')

    #Make Plot.ly Layout
    mylayout = go.Layout(scene=dict(xaxis=dict( title="肺活量"),
                                    yaxis=dict( title="50米成绩"),
                                    zaxis=dict(title="身高")),)

    #Plot and save html
    plotly.offline.plot({"data": [fig1],
                         "layout": mylayout},
                         auto_open=True,
                         filename=file_name)


def plotla(features,lable,file_name):       # data第一个数据为颜色
    markersize = normal(features[3],10,4)  # 体重
    #Make Plotly figure
    colorsize=(lable+1)*8
    fig1 = go.Scatter3d(x=features[0],
                        y=features[1],
                        z=features[2],
                        marker=dict(size=markersize,
                                    color=colorsize,
                                    opacity=0.9,
                                    reversescale=True,
                                    colorscale='Viridis'),
                        line=dict (width=0.02),
                        mode='markers')

    #Make Plot.ly Layout
    mylayout = go.Layout(scene=dict(xaxis=dict( title="肺活量"),
                                    yaxis=dict( title="50米成绩"),
                                    zaxis=dict(title="身高")),)

    #Plot and save html
    plotly.offline.plot({"data": [fig1],
                         "layout": mylayout},
                         auto_open=True,
                         filename=file_name)

#计算欧式距离
def ed(m, n):
    return np.sqrt(np.sum((m - n) ** 2))

class Classify_cmean:

    @staticmethod
    def classify(centers, datas):
        result = []
        for i in range(centers.__len__()):
            result.append([])
            result[i].append(centers[i])

        for i in range(datas.__len__()):

            min_distance = float('inf')  # 初始化最小距离
            class_index = 0  # 每个类别下标

            for j in range(centers.__len__()):

                # 依次计算每个数据到每一个类中心的欧式距离
                # 根据欧式距离分类
                distance =ed(centers[j], datas[i])
                if min_distance > distance:
                    min_distance = distance
                    class_index = j

            # 将属于这个类别的数据加入这个类中
            result[class_index].append(datas[i])
        return result

    # 计算每一个类的中心
    @staticmethod
    def get_new_center(datas):

        # 初始类中心，默认就是第一个位置的数据点
        # 即计算聚类中的所有数据点的各自维度的算术平均数
        data = datas[0].copy()

        # 遍历本类的所有数据
        for i in range(datas.__len__()):

            for j in range(data.__len__()):
                data[j] += datas[i][j]

        # 求每个维度和的算术平均
        for k in range(data.__len__()):
            data[k] /= datas.__len__()
        return data

def handle_r(list1,list2):
    array1=np.array(list1)
    array2=np.array(list2)
    array3=np.zeros(array1.shape[0])
    array4=np.ones(array2.shape[0])
    array1=np.insert(array1,0,array3,axis=1)
    array2=np.insert(array2,0,array4,axis=1)
    res3=[array1,array2]
    result=np.concatenate(res3,axis=0)
    return result