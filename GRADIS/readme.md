## 说明：
### 1、主程序入口：main
#### 参数：
##### threshold是过滤噪声标记的阈值
##### rate是聚类的比例超参数
##### k是k近邻参数,即控制矩阵稀疏程度的参数
##### alpha是指数移动加权平均的加权参数
##### emotions.mat是示例数据集,p3r3,p5r3,p7r3分别代表三种噪声比例的标记矩阵
### 2、运行示例
##### 进入文件路径，直接在matlab命令行输入：
##### main('emotions.mat','emotions',0.1,0.1,8,0.95)便可以运行，
##### 后面几个参数分别为path(数据集路径),
##### dataname(数据集名称,用于保存结果用),
##### threshold(过滤噪声标记的阈值),
##### rate(聚类簇心的超参数),
##### k(k近邻参数,控制矩阵稀疏化的程度),
##### alpha(指数移动加权平均的加权系数,控制初始标记矩阵的影响程度)
