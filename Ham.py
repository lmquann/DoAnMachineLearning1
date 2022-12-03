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
#print(os.path.join(path,img_file))
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
