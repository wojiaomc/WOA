import os
from urllib import request
from flask import Flask, request, render_template
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from 鲸鱼优化算法 import WhalePopulation, Start

"""读取数据集"""
data = pd.read_csv("新数据集.csv", encoding='utf-8')

text = ''

app = Flask(__name__)

@app.route('/')
def home():
    """渲染首页HTML模板，用于输入文本"""
    return render_template('index.html')


@app.route('/detail', methods=['POST'])
def detail(data=data):
    """接收文本，使用模型进行预测，然后渲染结果页面"""
    global text
    text = request.form['text']

    if text in data['城市'].values:
        # 选择要进行旅游的地区
        city_value = text
    else:
        return render_template('error.html')



    # 选择要进行旅游的地区
    #city_value = text

    # 复制data副本
    data_cy = data
    # 使用条件切片选择符合条件的行
    data_cy = data_cy[data_cy["城市"] == city_value]

    data_cy.reset_index(drop=True, inplace=True)
    data_cy['编号'] = range(len(data_cy))  # 将编号列的值进行重新编号，确保第一列能将景点按顺序排序

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='经度', y='纬度', data=data_cy)
    plt.title('景点经纬度分布')
    plt.xlabel('经度')
    plt.ylabel('纬度')
    plt.savefig('static/景点分布图.jpg')  # 保存图片到静态文件夹 static 中

    # Set the style of seaborn
    sns.set(style="whitegrid")
    # Plot 1: Distribution of the number of months of visits (Histogram)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure(figsize=(10, 6))
    sns.histplot(data_cy['价格'], kde=True)
    plt.title('价格分布')
    plt.xlabel('价格')
    plt.ylabel('频率')
    plt.savefig('static/价格分布图.jpg')  # 保存图片到静态文件夹 static 中
    # 将结果传递给结果页面
    return render_template('detail.html', text=text)


@app.route('/predict', methods=['POST'])
def predict(data=data):
    """接收文本，使用模型进行预测，然后渲染结果页面"""
    global text
    budget_limit = float(request.form['budget_limit'])
    time_limit = float(request.form['time_limit'])
    # 选择要进行旅游的地区
    city_value = text

    ##复制data副本
    data_cy = data

    # 使用条件切片选择符合条件的行
    data_cy = data_cy[data_cy["城市"] == city_value]

    data_cy.reset_index(drop=True, inplace=True)
    data_cy['编号'] = range(len(data_cy))  # 将编号列的值进行重新编号，确保第一列能将景点按顺序排序
    
    # 数据存储和管理
    Whale_pop = WhalePopulation(data_cy, budget_limit, time_limit)
    best_route_site, total_cost, total_time = Start(Whale_pop, budget_limit, time_limit)

    # 将结果传递给结果页面
    return render_template('result.html', best_route_site=best_route_site, total_cost=total_cost, total_time=total_time)


if __name__ == '__main__':
    app.run()
