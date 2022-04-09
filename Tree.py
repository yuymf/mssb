import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import preprocessing
from sklearn.preprocessing import OrdinalEncoder
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import graphviz


def pre_data():
    """
    考虑到不同特征属于不同的量级，需要进行标准化处理
    :return: 处理后的特征数据和标签数据
    """
    file_loc = "作业数据_2021合成.xls"
    file_data = pd.read_excel(file_loc)
    data_color = file_data[["喜欢颜色"]]
    ordinal_encoder = OrdinalEncoder()
    color_encoder = ordinal_encoder.fit_transform(data_color)
    print(color_encoder, data_color)
    data_feature = np.hstack((color_encoder, pd.DataFrame(file_data, columns=['喜欢运动', '喜欢文学']).values))
    # print(data_feature)
    scaler = preprocessing.StandardScaler().fit(data_feature)
    data_feature = scaler.transform(data_feature)
    data_label = file_data.性别男1女0.values
    return file_data, data_feature, data_label


def split_data(data_feature, data_label):
    """
    分层采样，训练集为0.8，测试集为0.2
    :param data_label: 标签数据
    :param data_feature: 特征数据集
    :return: 测试集和训练集
    """
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(data_feature, data_label):
        data_train, data_test = data_feature[train_index], data_feature[test_index]
        label_train, label_test = data_label[train_index], data_label[test_index]
    print(len(data_train), len(data_test), len(label_train))
    return data_train, data_test, label_train, label_test


if __name__ == '__main__':
    data, data_feature, data_label = pre_data()
    # print(data_feature, data_label)
    data_train, data_test, label_train, label_test = split_data(data_feature, data_label)

    # 绘制树模型
    clf = DecisionTreeClassifier(criterion="entropy", random_state=30, splitter='random',max_depth=7)
    clf = clf.fit(data_train, label_train)
    score_train = clf.score(data_train, label_train)
    score = clf.score(data_test, label_test)
    # 训练集上的准确率为0.85，测试集上的准确率为0.7571428571428571
    print(score_train, score)

    tree.export_graphviz(clf)
    dot_data = tree.export_graphviz(clf, out_file=None)
    graphviz.Source(dot_data)

    # 给图形增加标签和颜色
    feature_name = ['color', 'sport', 'literature']
    dot_data = tree.export_graphviz(clf, out_file=None, feature_names=['color', 'sport', 'literature'],
                                    class_names=['girl', 'boy'], filled=True, rounded=True,
                                    special_characters=True)
    graphviz.Source(dot_data)

    # 利用render方法生成图形
    graph = graphviz.Source(dot_data)
    graph.render("classify")

    print(clf.feature_importances_)
    print([*zip(feature_name, clf.feature_importances_)])

    # 最优剪枝参数
    test = []
    for i in range(10):
        clf = tree.DecisionTreeClassifier(criterion="entropy", random_state=30, splitter='random', max_depth=i + 1)
        clf = clf.fit(data_train, label_train)
        score = clf.score(data_test, label_test)
        test.append(score)
    plt.plot(range(1, 11), test, color="red", label="max_depth")
    plt.xlabel('决策树深度')
    plt.ylabel('ACC')
    plt.legend()
    plt.show()
