import pandas as pd
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from geopy.distance import geodesic
import matplotlib.pyplot as plt
import numpy as np
import random
import math  # 导入数学运算库
from sklearn.preprocessing import MinMaxScaler

# --------------------------函数定义-------------------------------
# 计算景点之间的距离
def calculate_distance(point1, point2,data):
    info_1 = data.loc[point1]  # 获取对应行的数据
    info_2 = data.loc[point2]
    origin = (info_1['纬度'], info_1['经度'])
    destination = (info_2['纬度'], info_2['经度'])
    distance = geodesic(origin, destination).kilometers
    return distance


# 计算两个景点之间的路程时间（假设每小时对应30km）
def calculate_travel_time(point1, point2,data):
    distance = calculate_distance(point1, point2,data)
    travel_time = distance / 30  # 每小时对应30km
    return travel_time

# 计算总时间：包括游览时间和路程时间
def calculate_total_time(route, data, average_time):  # 路程，数据集,平均浏览时间
    total_time = 0
    for i in range(len(route) - 1):
        point1 = route[i]
        point2 = route[i + 1]
        # 计算路程时间
        travel_time = calculate_travel_time(point1, point2,data)
        total_time += average_time + travel_time
    total_time += average_time
    return total_time

# 验证个体位置的有效性
def Valid_Geti(Whale, budget_limit,time_limit):
    '''
    当前传进来的鲸鱼不满足条件时随机减少一个景点，条件是不超过时间和预算
    '''
    total_cost, total_time = 0, 0
    unique_route = []  # 用于存储不重复的景点
    # 对路线进行去重

    for index in Whale.route:
        if index not in unique_route:
            unique_route.append(index)
    Whale.route = unique_route

    # 计算去重后的路线的合法性
    for index in Whale.route:  # 计算总价格
        info = Whale.data.loc[index]  # 获取对应行的数据
        info_cost = float(info['价格'])
        total_cost += info_cost
    total_time = calculate_total_time(Whale.route, Whale.data, 1)  # 计算总时间

    # 去除多余的景点
    while total_cost > budget_limit or total_time > time_limit:
        # 当前个体不满足条件，减少一个景点
        if len(Whale.route) > 1:
            Whale.route.pop()  # 减少最后一个景点
            # Whale.route.pop(random.randint(0, len(Whale.route) - 1))#随机减少一个景点
            total_cost, total_time = 0, 0
            for index in Whale.route:  # 计算总价格
                info = Whale.data.loc[index]  # 获取对应行的数据
                info_cost = float(info['价格'])
                total_cost += info_cost
            # print("当前个体不满足条件，随机减少一个景点")
            total_time = calculate_total_time(Whale.route, Whale.data, 1)  # 计算总时间

    # 如果总时间和预算小于规定值，则在路线的最后面附加一个景点
    while total_cost < budget_limit and total_time < time_limit:
        # 附加一个景点到路线的最后
        new_point = random.choice(list(set(Whale.data['编号']) - set(Whale.route)))
        Whale.route.append(new_point)
        # print("总时间和预算小于规定值，附加一个景点")
        # 再次判断总时间和预算是否大于规定值
        total_cost, total_time = 0, 0
        for index in Whale.route:
            info = Whale.data.loc[index]  # 获取对应行的数据
            info_cost = float(info['价格'])
            total_cost += info_cost
        total_time = calculate_total_time(Whale.route, Whale.data, 1)  # 计算总时间
        if total_cost > budget_limit or total_time > time_limit:
            # 如果超过了规定值，则移除刚刚附加的景点
            Whale.route.pop()
            # print("附加的景点导致总时间和预算超过规定值，移除该景点")
            break


# --------------------------定义鲸鱼类----------------------------
# 定义鲸鱼个体
class Whale:
    def __init__(self, data, budget_limit, time_limit):
        self.data = data
        self.route = self.initialize_route()
        Valid_Geti(self, budget_limit, time_limit)
        self.fitness = self.calculate_fitness()
        # self.budget_limit = 500
        # self.time_limit = 100

    def initialize_route(self):
        route_length = random.randint(1, len(self.data))
        route = random.sample(self.data['编号'].tolist(), route_length)
        # route_indices = random.sample(range(len(self.data)), route_length)
        # route = self.data.iloc[route_indices]
        return route

    def calculate_fitness(self):  # 仅作示例，后续补充综合评分、销量和评分编码
        total_score, total_cost, total_time = 0.0, 0.0, 0.0
        for index in self.route:
            info = self.data.loc[index]  # 获取对应行的数据
            info_score = float(info['评分'])  # 将info['评分']强制转化为float
            total_score += info_score
            '''前面已经验证过范围，不用再验证了
            info_cost = float(info['价格'])
            total_cost += info_cost
            total_time += 2 # 数据集中没有给出游玩时间，需用户自行拟定
        # 适应度考虑评分、成本和时间
        if total_cost > budget_limit or total_time > time_limit:
            return -np.inf  # 超出预算或时间限制的适应度为无穷小
        '''
        return total_score  # 适应度函数


# 定义鲸鱼种群
class WhalePopulation:
    # 初始化
    def __init__(self, data, budget_limit, time_limit, pop_size=30, max_iter=100):
        self.data = data
        self.pop_size = pop_size  # 种群数量
        self.max_iter = max_iter  # 最大迭代次数
        self.population = [Whale(data, budget_limit, time_limit) for _ in range(pop_size)]
        self.best_whale = self.population[0]  # 将第一个鲸鱼设为最佳解
        self.current_best = self.population[0]  # 当前迭代轮次的最佳解
        self.best_fitness = self.best_whale.fitness
        self.Convergence_curve = []

    # 优化函数，选出最佳鲸鱼，没用上
    def optimize(self):
        best_whale = max(self.population, key=lambda whale: whale.fitness)

        for iteration in range(self.max_iter):
            for whale in self.population:
                whale.update_route(best_whale.route)

            current_best = max(self.population, key=lambda whale: whale.fitness)
            if current_best.fitness > best_whale.fitness:
                best_whale = current_best

            print(f"Iteration {iteration + 1}: Best Fitness: {best_whale.fitness}, Best Route: {best_whale.route}")

        return best_whale.route


# --------------------------------鲸鱼优化算法--------------------------------------------
def Start(Whale_pop, budget_limit, time_limit):
    t = 0  # 初始化迭代次数
    BestFitness = Whale_pop.best_whale.fitness
    while t < Whale_pop.max_iter:  # 当迭代次数小于最大迭代次数时
        # print("第",t,"次迭代")
        a = 2 - t * (2 / Whale_pop.max_iter)  # 计算参数a
        a2 = -1 + t * ((-1) / Whale_pop.max_iter)  # 计算参数a2
        for i in range(0, Whale_pop.pop_size):  # 遍历种群大小
            # print("第", i ,"个个体")
            # print("个体路线长度:", len(Whale_pop.population[i].route))
            r1 = random.random()  # 生成随机数r1
            r2 = random.random()  # 生成随机数r2
            A = 2 * a * r1 - a  # 计算参数A
            C = 2 * r2  # 计算参数C
            b = 1
            l = (a2 - 1) * random.random() + 1  # 计算参数l
            p = random.random()  # 生成随机数p
            for j in range(0, len(Whale_pop.population[i].route)):  # 遍历个体的每个维度
                # print("第",j,"个属性")
                if p < 0.5:  # 如果随机数p小于0.5
                    if abs(A) >= 1:
                        # 如果参数A的绝对值大于等于1，进行探索
                        rand_leader_index = math.floor(Whale_pop.pop_size * random.random())  # 随机选择一个领导个体的索引
                        X_rand = Whale_pop.population[rand_leader_index]  # 获取随机领导个体
                        if j < len(X_rand.route):  # 防止j超过随机个体的长度
                            D_X_rand = abs(C * X_rand.route[j] - Whale_pop.population[i].route[j])  # 计算D_X_rand
                            k = int(X_rand.route[j] - A * D_X_rand)  # 更新后的位置
                            if 0 <= k < len(Whale_pop.data):  # 防止位置超出范围
                                Whale_pop.population[i].route[j] = k  # 更新个体位置
                                # print("成功更新1")
                            # else:
                            # print("更新位置超出范围1")
                        # else:
                        # print("比目标长1")
                        # rand_index = random.randint(0, len(X_rand.route) - 1)#因为每个鲸鱼个体的路径长度不一样，所以不能用j而是随机位置

                    else:  # 如果参数A的绝对值小于1
                        # 向当前最优解直线逼近
                        # rand_index = random.randint(0, len(Whale_pop.best_whale.route) - 1)
                        if j < len(Whale_pop.best_whale.route):
                            D_Leader = abs(
                                C * Whale_pop.best_whale.route[j] - Whale_pop.population[i].route[j])  # 计算D_Leader
                            k = int(Whale_pop.best_whale.route[j] - A * D_Leader)  # 更新后的位置
                            if 0 <= k < len(Whale_pop.data):  # 防止位置超出范围
                                Whale_pop.population[i].route[j] = k  # 更新个体位置
                                # print("成功更新2")
                            # else:
                            # print("更新位置超出范围2")
                        # else:
                        # print("比目标长2")
                else:  # 如果随机数p大于等于0.5
                    # 向当前最优解进行螺旋形移动
                    # rand_index = random.randint(0, len(Whale_pop.best_whale.route) - 1)
                    if j < len(Whale_pop.best_whale.route):
                        distance2Leader = abs(
                            Whale_pop.best_whale.route[j] - Whale_pop.population[i].route[j])  # 计算到最优领导个体的距离
                        k = int(distance2Leader * np.exp(b * l) * np.cos(l * 2 * math.pi) + Whale_pop.best_whale.route[
                            j])  # 更新后的位置
                        if 0 <= k < len(Whale_pop.data):  # 防止位置超出范围
                            Whale_pop.population[i].route[j] = k  # 更新个体位置
                            # print("成功更新3")
                        # else:
                        # print("更新位置超出范围3")
                    # else:
                    # print("比目标长3")

        t = t + 1  # 迭代次数加1
        Cur_Fitness = Whale_pop.current_best.fitness  # 当前迭代的最佳适应值
        Whale_pop.Convergence_curve.append(BestFitness)  # 将当前最优适应度添加到收敛曲线列表中
        # 找出当前迭代的最佳鲸鱼current_best
        for i in range(0, Whale_pop.pop_size):  # 遍历种群大小
            Valid_Geti(Whale_pop.population[i], budget_limit, time_limit)  # 验证个体位置的有效性
            fitness_i = Whale_pop.population[i].calculate_fitness()  # 计算当前个体的适应度
            if fitness_i > Cur_Fitness:  # 如果当前个体的适应度优于当前最优适应度
                Whale_pop.current_best = Whale_pop.population[i]  # 更新最优个体
                Cur_Fitness = fitness_i  # 更新最优适应度
        print(' {}%/100%'.format(t))  # 输出当前最优个体及其适应度
        if Cur_Fitness > BestFitness:
            BestFitness = Cur_Fitness
            Whale_pop.best_whale = Whale_pop.current_best
    print('{} 次迭代后WOA种群最优座鱼鲸个体 {} 的评分为 {}'.format(Whale_pop.max_iter, Whale_pop.best_whale, BestFitness))  # 输出最终最优个体及其适应度值

    #将路线转换成名称
    best_route_site = []
    for index in Whale_pop.best_whale.route:  # 计算总价格
        info = Whale_pop.data.loc[index]  # 获取对应行的数据
        info_name = info['名称']
        best_route_site.append(info_name)

    #计算最佳路线的总预算
    total_cost = 0.0
    for index in Whale_pop.best_whale.route:  # 计算总价格
        info = Whale_pop.data.loc[index]  # 获取对应行的数据
        info_cost = float(info['价格'])
        total_cost += info_cost
    total_cost = round(total_cost, 2)#保留两位小数
    #计算最佳路线的总时间
    total_time = calculate_total_time(Whale_pop.best_whale.route, Whale_pop.data, 1)  # 计算总时间
    total_time = round(total_time, 1)
    return best_route_site, total_cost, total_time
    # budget_limit 和 time_limit 需要根据实际情况设定

