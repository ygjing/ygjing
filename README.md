import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
!unzip "/kaggle/input/facial-keypoints-detection/training.zip"
train = pd.read_csv('training.csv')
train.head()
import tensorflow as tf
from tensorflow import keras
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense, BatchNormalization
from keras.models import Sequential
from keras.optimizers import Adam
from keras.losses import MeanSquaredError
# 检查训练数据中的缺失值并进行处理
train.isnull().sum()
train.fillna(method='ffill', inplace=True)
train.isnull().sum()
# 读取IdLookupTable和SampleSubmission数据
IdLookupTable = pd.read_csv('/kaggle/input/facial-keypoints-detection/IdLookupTable.csv')
IdLookupTable.head()
sub = pd.read_csv('/kaggle/input/facial-keypoints-detection/SampleSubmission.csv')
sub.head()
# 查看训练数据的形状，并打印出其行数和列数
train.shape
m,n = train.shape
print(m, '\t', n)
# 查看测试数据的形状
test.shape
# 处理训练数据中的图像信息
img = []
img_size = 96
for i in range(m):
    spliting = np.array(train['Image'][i].split(' '),dtype = 'float64')
    splitting = np.reshape(spliting,(img_size,img_size,1))
    splitting /= 255 
    img.append(splitting)
img = np.array(img)
X_train = img
# 删除训练数据中的图像列
train.drop('Image', axis = 1, inplace = True)
# 提取训练数据中的关键点信息
y_train = []
for i in range(len(train)):
    y = train.iloc[i,:].values
    y_train.append(y)
y_train= np.array(y_train, dtype = 'float')
# 构建卷积神经网络模型
model = Sequential([
    Conv2D(128, (3, 3),strides=1,activation='relu',padding = 'same', input_shape=(96, 96, 1)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), strides=1,activation='relu',padding = 'same'),
    Dropout(0.1),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), strides=1,activation='relu',padding = 'same'),
    Dropout(0.2),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(256, (3, 3), strides=1,activation='relu',padding = 'same'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.1),
    Dense(256, activation='relu'),
    Dropout(0.1),
    Dense(128, activation='relu'),
    Dense(30)
])
# 配置模型的优化器、损失函数和评估指标
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mae',metrics = ['accuracy'])
# 训练模型，并将训练过程保存在history变量中
history = model.fit(X_train.reshape(-1, 96, 96, 1), y_train, epochs=2, batch_size=32,validation_split=0.2)
# 处理测试集数据
X_test = np.array([np.fromstring(x, dtype=int, sep=' ') for x in test['Image']])
X_test = X_test.reshape(-1, 96, 96, 1) / 255.0
# 对测试集进行预测
y_test_pred = model.predict(X_test)
# 将预测结果保存为变量
y_test_pred
# 准备测试集图像数据并进行预测
test_images = []
for i in range(len(test)):
    item = np.array(test['Image'][i].split(' '), dtype='float')
    item = np.reshape(item, (img_size, img_size, 1))
    item /= 255
    test_images.append(item)
    # 转换测试集图像数据为NumPy数组
test_images = np.array(test_images, dtype='float')
# 使用模型对测试集进行预测
predict = model.predict(test_images)
# 读取IdLookupTable.csv文件
IdLookupTable=pd.read_csv('/kaggle/input/facial-keypoints-detection/IdLookupTable.csv')
# 获取特征名、图像ID和行ID
feature_names = list(IdLookupTable['FeatureName'])
image_ids = list(IdLookupTable['ImageId']-1)
row_ids = list(IdLookupTable['RowId'])
# 获取特征列表
feature_list = []
for feature in feature_names:
feature_list.append(feature_names.index(feature))
# 存储预测结果
predictions = []
for x,y in zip(image_ids, feature_list):
predictions.append(predict[x][y])
# 创建行ID、位置的Series
row_ids = pd.Series(row_ids, name = 'RowId')
locations = pd.Series(predictions, name = 'Location')
# 将位置数据限制在[0.0, 96.0]范围内
locations = locations.clip(0.0,96.0)
# 合并行ID和位置数据
submission_result = pd.concat([row_ids,locations],axis = 1)
# 将结果保存为CSV文件
submission_result.to_csv('submission.csv',index = False)

