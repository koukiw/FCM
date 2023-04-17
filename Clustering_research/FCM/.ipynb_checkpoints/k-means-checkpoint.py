import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#sysはコマンドライン引数
import sys
import csv
#不動小数点計算のエラーが起きたときの動作を指定、divideは0で除算した時、invalidはよくわからん
np.seterr(divide='ignore', invalid='ignore')
#csvファイルの読み込み、一行目はヘッダーじゃない
df = pd.read_csv(sys.argv[1],header=None)

#クラスタ数宣言
k=int(sys.argv[2])

#max繰り返し回数の宣言
max_iter=10
#dfをnumpyで扱えるnp.ndarray変換
X = df.values
#配列の行数、列数の格納
X_size,n_features = X.shape


# 前の重心と比較するために、仮に新しい重心を入れておく配列を用意
new_centroids = np.zeros((k, n_features))
    
# 各データ所属クラスタ情報を保存する配列を用意
cluster = np.zeros(X_size)

#重心と各点との最小距離を格納
sum_distance = []
min_distance = []

#目的関数のリスト作成
J = []
#目的関数の初期値作成
best_J = 10000
#目的関数の時の重心の初期値
best_centroids = np.ones((k, n_features))

#HCMアルゴリズム
for s in range(100):
    # ランダムに重心の初期値を初期化
    centroids  = X[np.random.choice(X_size,k)]
    # ループ上限回数まで繰り返し
    for epoch in range(max_iter):
    # 入力データ全てに対して繰り返し
        for i in range(X_size):
            # データから各重心までの距離を計算（ルートを取らなくても大小関係は変わらないので省略）
            distances = np.sum((centroids - X[i]) ** 2, axis=1)
            #一番近い重心との距離を格納
            sum_distance.append(min(distances))

            # データの所属クラスタを距離の一番近い重心を持つものに更新
            cluster[i] = np.argsort(distances)[0]

        #目的関数の値をリストmin_distanceに格納    
        min_distance.append(sum(sum_distance))
        #重心と各点との最小距離を格納するリストの初期化
        sum_distance =list()

        # すべてのクラスタに対して重心を再計算
        for j in range(k):
            new_centroids[j] = X[cluster==j].mean(axis=0)

        # もしも重心が変わっていなかったら終了
        if (np.allclose(new_centroids, centroids)):
            break
        centroids =  np.copy(new_centroids)

    if best_J > min_distance[-1]:
        #最適な目的関数の更新
        best_J = min_distance[-1]
        #目的関数が小さくなるときの重心を格納
        best_centroids = np.copy(centroids)

#best_centroidsにてもう一回HCMを行う
for epoch in range(max_iter):
    # 入力データ全てに対して繰り返し
    for i in range(X_size):
        # データから各重心までの距離を計算（ルートを取らなくても大小関係は変わらないので省略）
        distances = np.sum((best_centroids - X[i]) ** 2, axis=1)
        #一番近い重心との距離を格納
        sum_distance.append(min(distances))

        # データの所属クラスタを距離の一番近い重心を持つものに更新
        cluster[i] = np.argsort(distances)[0]
    

    # すべてのクラスタに対して重心を再計算
    for j in range(k):
        new_centroids[j] = X[cluster==j].mean(axis=0)

    # もしも重心が変わっていなかったら終了
    if (np.allclose(new_centroids, best_centroids)):
        break
    best_centroids =  np.copy(new_centroids)
print("目的関数の値は"+str(best_J))
print(centroids)


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
    df["cluster"]=cluster
    df.to_csv('result2d.csv')
    #目的関数を記入
    with open ('result2d.csv','a') as f:
        writer = csv.writer(f)
        writer.writerow(["J",best_J])


#matplotlib3次元version
else:
    # matplotで３Dモデル表示
    colors = ['red', 'blue', 'green','yellow','fuchsia']
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel("X", size=16)
    ax.set_ylabel("Y",size=14)
    ax.set_zlabel("Z",size=14)

    for i in range(k):
        ax.scatter(X[:, 0][cluster==i], X[:, 1][cluster==i], X[:, 2][cluster==i], color=colors[i])

    plt.show()
    fig.savefig("img3d.pdf")

    #pandasで分割結果を記入
    df["cluster"]=cluster   
    df.to_csv('result3d.csv')
    #目的関数を記入
    with open ('result3d.csv','a') as f:
        writer = csv.writer(f)
        writer.writerow(["J",best_J])


    


                   