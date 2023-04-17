#!/usr/bin/env python
# coding: utf-8

# In[504]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#クラスタ数宣言
cluster_number=5
#max繰り返し回数の宣言
max_iter=100
#dfをnumpyで扱えるnp.ndarray変換
df = pd.read_csv("input.csv",header=None)
X = df.values
#配列の行数、列数の格納
X_size,n_features = X.shape
#epsilonの定義
epsilon=0.0001
#前の中心と比較するために、仮に新しい重心を入れておく配列を用意
new_v = np.zeros((cluster_number, n_features))
#メンバーシップuを作成(配列初期値は０にしないとおかしくなる)
u = np.zeros((X_size,cluster_number))
new_u = np.zeros((X_size,cluster_number))
#距離の格納 d_ik
distances=np.zeros((X_size,cluster_number))

m=3

best_J = 0
v = X[np.random.choice(X_size,cluster_number)]

#FCMアルゴリズム
for epoch in range(max_iter):
    #FCM2における最適解(new_uの導出)
    for i in range(X_size):    
        # データから各重心までの距離の二乗を計算
        distances[i] = np.sum((v - X[i]) ** 2, axis=1)
        #distances[i] = np.sum((v - X[i]) ** 2, axis=1)
        #print(distances[i])
        #x_k!=v_iの時
        if 0 not in distances[i]:
            for j in range(cluster_number):
                for k in range(cluster_number):
                    new_u[i][j] = new_u[i][j] + (distances[i][j]/distances[i][k])**(1/(m-1))
                new_u[i][j]=1/new_u[i][j]
        #x_k=v_iの時
        else:
            for j in range(cluster_number):
                if distances[i][j]==0:
                    new_u[i][j] = 1
                else:
                    new_u[i][j]=0
    #s = np.sum(new_u,axis=1)
    #print(s)
    
    #FCM3における最適解(new_vの導出)
    new_u_m=np.power(new_u,m)
    #分母計算
    v_denominator=np.sum(new_u_m,axis=0)
    #分子計算
    v_numerator = np.zeros((cluster_number,n_features))
    for i in range(cluster_number):
        for j in range(X_size):
            v_numerator[i]=v_numerator[i]+new_u_m[j][i]*X[j]
            #print("分子")
            #print(v_numerator)
            #print("new_v")
    for i in range(cluster_number):
        new_v[i] = v_numerator[i]/v_denominator[i]
    #print(new_v)

    #vの収束条件
    v_difference = np.zeros((cluster_number,n_features))
    for i in range(cluster_number):
        for j in range (n_features):
            v_difference[i][j]=np.abs(new_v[i][j]-v[i][j])

    #uの収束条件
    u_difference = np.zeros((X_size,n_features))
    for i in range(X_size):
        for j in range (n_features):
            u_difference[i][j]=np.abs(new_u[i][j]-u[i][j])   

    #目的関数の計算
    for i in range(X_size):
        for j in range(cluster_number):
            J=J+((new_u[i][j])**m)*(distances[i][j])
    #print("Jの値は"+str(J))
    #print("best_Jの値は"+str(best_J))
    #print("v_differenceは"+str(np.amax(v_difference)))
    #print("u_differenceは"+str(np.amax(u_difference)))
     
    #目的関数の値が良いものに更新していく
    if J < best_J or best_J==0:
        best_J = J
        best_u = np.copy(new_u)
        best_v = np.copy(new_v)
    #値のリセット
    J=0
    u = np.copy(new_u)
    v = np.copy(new_v)
    new_u = np.zeros((X_size,cluster_number))
    
    #収束条件
    if np.amax(v_difference) < epsilon or np.amax(u_difference) < epsilon:  
        print("breakしました")
        break
print(str(epoch)+"回目で収束しました")

#matplotlibに描写
if n_features == 2:
    colors = ['red', 'blue', 'green','yellow','fuchsia']

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for i in range(k):
        ax.scatter(X[:, 0][cluster==i], X[:, 1][cluster==i], color=colors[i])

    ax.set_title('k-means', size=16)
    ax.set_xlabel("X", size=14)
    ax.set_ylabel("Y", size=14)

    plt.show()
    fig.savefig("img2d.pdf")

    #pandasで分割結果を記入
    df[u_1]=best_u[:,1]
    df.to_csv('fcm2d.csv')
    #目的関数を記入
    with open ('fcm2d.csv','a') as f:
        writer = csv.writer(f)
        writer.writerow(["J",best_J])


# In[ ]:




