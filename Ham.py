import os
import numpy as np
from skimage import io
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import hamming_loss
from sklearn.preprocessing import LabelEncoder
from PIL import Image

PATH = "D:\MachineLearning01"
def check_corrupted_image(img_file):
    try:
        with Image.open(img_file) as img:
            img.verify()
            img_new = io.imread(img_file)
        return False
    except Exception as e:
        print(e)
        return True
def read_img_data(path, label, size):
    X = []
    y = []
    files = os.listdir(path)
    for img_file in files:
        if not(check_corrupted_image(os.path.join(path, img_file))):
            img = io.imread(os.path.join(path, img_file), as_gray = True)
            img = resize(img, size).flatten()
            X.append(img)
            y.append(label)
    return X, y
#Phân chia train test theo tỉ lệ 70% - 30%
def train_test_split(X, y, testsize, randomstate):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = testsize, random_state = randomstate)
    return X_train, X_test, y_train, y_test
#Mã hóa nhãn lớp
def extract_label(y):
    lb = LabelEncoder()
    return lb.fit_transform(y).reshape(y.shape[0], )
#hàm chuyển ảnh màn hình thành vector 1024
def convert_D_2_vector(path,label,size):
    labels = []
    img_data = []
    images = os.listdir(path)
    for img_file in images:
        if not(check_corrupted_image(os.path.join(path,img_file))):
            img_grey = io.imread(os.path.join(path,img_file), as_grey = True)
            img_vector = resize(img_grey,size).flatten
            img_data.append(img_vector)
            labels.append(label)
    return img_data, labels
#Hàm huấn luyện mô hình
def kNN_grid_search_cv(X_train, y_train):
  from math import sqrt
  m = y_train.shape[0]
  k_max = int(sqrt(m)/2)
  k_values = np.arange(start = 1, stop = k_max + 1, dtype = int)
  params = {'n_neighbors': k_values}
  kNN = KNeighborsClassifier()
  kNN_grid = GridSearchCV(kNN, params, cv=3)
  kNN_grid.fit(X_train, y_train)
  return kNN_grid
def logistic_regression_cv(X_train, y_train):
    logistic_classifier = LogisticRegressionCV(cv=5, solver="sag", max_iter=2000)
    logistic_classifier.fit(X_train, y_train)
    return logistic_classifier
#Hàm đánh giá mô hình
def evaluate_model(y_test, y_pred):
  print("accuracy score: ", accuracy_score(y_test, y_pred))
  print("Balandced accuracy score: ", balanced_accuracy_score(y_test, y_pred))
  print("Haming loss: ", hamming_loss(y_test, y_pred))
def main():
        # Đọc dữ liệu ảnh, nhãn từ các folder
        X, y = read_img_data(PATH, size=(32, 32))
        # Mã hóa nhãn lớp
        y = extract_label(y)
        # Phân chia tập train và tập test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)
        # Huấn luyện mô hình
        kNN_classifier = kNN_grid_search_cv(X_train, y_train)
        logistic_classifier = logistic_regression_cv(X_train, y_train)
        # Dự đoán kết quả
        y_pred_kNN = kNN_classifier.predict(X_test)
        y_pred_logistic = logistic_classifier.predict(X_test)
        # Đánh giá mô hình
        evaluate_model(y_test, y_pred_kNN)
        evaluate_model(y_test, y_pred_logistic)

if __name__ == '__main__':
    main()
