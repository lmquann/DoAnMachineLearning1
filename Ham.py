import os
import numpy as np
from skimage import io
from skimage.transform import resize
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from PIL import Image

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
def kNN_grid_search_cv(x_train, y_train):
  from math import sqrt
  m = y_train.shape[0]
  k_max = int(sqrt(m)/2)
  k_values = np.arange(start = 1, stop = k_max + 1, dtype = int)
  params = { 'n_neighbors': k_values}
  kNN = KNeighborsClassifier()
  kNN_grid = GridsearchCV(kNN, params, cv=3)
  kNN_grid.fit(x_train, y_train)
  return kNN_grid
def logistic_regression_cv(x_train, y_train):
    logistic_classifier = LogisticRegreesionCV(cv=5, solver="sag", max_iter=2000)
    logistuc_classifier.fit(X_train, y_train)
    return logistic_classifier
#Hàm đánh giá mô hình
def evaluate_model(y_test, y_pred):
  print("accuracy score: ", accuracy_score(y_test, y_pred))
  print("Balandced accuracy score: ", balandced_accuracy_score(y_test, y_pred))
  print("Haming loss: ", hamming_loss(y_test, y_pred))
def confusion_matrix(y_test, y_pred, model, plt=None):
  ax1 = sns.heatmap(confusion_matrix(y_test, y_pred), annot = True, fmt=",of", cmap="crest"J
  ax1.set(xlabel="y pred", ylabel="y test”, title= model
  ax1.xaxis.tick_top()
  plt.show
def main():
    X, y = read_img_data('D:/MachineLearning01/PetImages/Cat','Cat', (32,32))
    X_dog, y_dog = read_img_data('D:/MachineLearning01/PetImages/Dog','Dog', (32,32))
    X.extend(X_dog)
    y.extend(y_dog)
    X = np.array(X)
    y = np.array(y)
    y = LabelBinarizer().fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X,  y, shuffle =True, random_state = 123)
    print(X.shape)
if __name__ == '__main__':
    main()
