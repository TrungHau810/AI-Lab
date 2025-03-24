from mglearn.datasets import make_forge
from sklearn.datasets import load_iris, load_diabetes, fetch_20newsgroups
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.datasets import fetch_lfw_people

import numpy as np
import pandas as pd
import mglearn
import warnings

from sklearn.utils.parallel import Parallel, delayed

warnings.filterwarnings("ignore", category=UserWarning)

import ssl

# Tắt xác thực SSL
ssl._create_default_https_context = ssl._create_unverified_context

# Tải toàn bộ dữ liệu iris
iris = load_iris()

# 1/ Download iris data and use skicit-learn to import. Try to call the attribute data of the variable iris.
print('--- Câu 1 ---')
iris_data = iris.data
print('Dữ liệu Iris (5 hàng đầu tiên):')
print(iris_data[:5])

# 2/ How to know what kind of flower belongs each item? How to know the correspondence between the species and the number?
print('\n--- Câu 2 ---')
# Truy cập target (nhãn loài)
iris_target = iris.target
print("Target (5 giá trị đầu tin):", iris_target[:5])
# Truy cập tên các loài
species_name = iris.target_names
print("Tên các loài:", species_name)
# Sự tương ứng
print("0 = {}, 1 = {}, 2 = {}".format(species_name[0], species_name[1], species_name[2]))

# 3/ Create a scatter plot that displays three different species in three different colors;
# X-axis will represent the length of the sepal while the y-axis will represent the width of the sepal.
print('\n--- Câu 3 ---')
# Trích xuất chiều dài và chiều rộng đài hoa
sepal_length = iris.data[:, 0]  # Cột 0
sepal_width = iris.data[:, 1]  # Cột 1
# Vẽ biểu đồ phân tán
plt.scatter(sepal_length[iris.target == 0], sepal_width[iris.target == 0], label='Setosa', c='red')
plt.scatter(sepal_length[iris.target == 1], sepal_width[iris.target == 1], label='Versicolor', c='blue')
plt.scatter(sepal_length[iris.target == 2], sepal_width[iris.target == 2], label='Virginica', c='green')
plt.xlabel('Chiều dài dài hoa (cm)')
plt.ylabel('Chiều rộng dài hoa (cm)')
plt.title("Biểu đồ phân tán của các loài Iris")
plt.legend()
plt.show()

# 4/ Using reduce dimension, here using PCA, create a new dimension (=3, called principle component).
print('\n--- Câu 4 ---')
# Khởi tạo PCA với 3 thành phần
pca = PCA(n_components=3)
# Áp dụng PCA lên dữ liệu
iris_pca = pca.fit_transform(iris_data)
print("Dữ liệu sau khi giảm chiều (5 hàng đầu tiên):")
print(iris_pca[:5])

# 5/ Using k-nearest neighbor to classify the group that each species belongs to.
# First, create a training set and test set; with 140 will be used as a training set, and 10 remaining will be used as test set.
print('\n--- Câu 5 ---')
# Chia dữ liệu 140maauxu huấn luyện và 10 mẫu kiểm tra
x_train, x_test, y_train, y_test = train_test_split(iris_data, iris.target, train_size=140, test_size=10,
                                                    random_state=42)
print("Kích thước tập huấn luyện:", x_train.shape)
print("Kích thước tập kiểm tra", x_test.shape)

# 6/ Next, apply the K-nearest neighbor, try with K=5.
print('\n--- Câu 6 ---')
# Khởi tạo KNN với k = 5
knn = KNeighborsClassifier(n_neighbors=5)
# Huấn luyện mô hình
knn.fit(x_train, y_train)
# Dự đoán trên tập kiểm tra
y_pred = knn.predict(x_test)
print("Dự đoán:", y_pred)

# 7/ Finally, you can compare the results predicted with the actual observed contained in the y_test.
print('\n--- Câu 7 ---')
# So sánh
print("Nhãn thực tế:", y_test)
print("Nhãn dự đoán:", y_pred)
# Tính độ chính xác
accuracy = accuracy_score(y_test, y_pred)
print("Độ chính xác:", accuracy)

# 8/ Now, you can visualize all this using decision boundaries in a space represented by the 2D scatterplot of sepals.
print('\n--- Câu 7 ---')
# Sử dụng 2 đặc trưng đài hoa
X = iris.data[:, [0, 1]]  # Sepal length, sepal width
y = iris.target
# Tạo dưới tọa độ
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
# Huấn luyện KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X, y)
# Huấn luyện trên lưới
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
# Vẽ ranh giới
plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
plt.xlabel("Chiều dài đài hoa (cm)")
plt.xlabel("Chiều rộng đài hoa (cm)")
plt.title("Ranh giới quyết định với KNN")
plt.show()

# 9/ Download diabete dataset. To predict the model, we use the linear regression.
print('\n--- Câu 9 ---')
# Tải toàn bộ dữ liệu Diabetes
diabetes = load_diabetes()
print("Dữ liệu (5 hàng đầu tiên:")
print(diabetes.data[:5])

# 10/ First, you will need to break the dataset into a training dataset (composed of the first 422 patients) and a test set (the last 20 patients).
print('\n--- Câu 10 ---')
# Chia dữ liệu
X_train = diabetes.data[:422]
X_test = diabetes.data[422:]
y_train = diabetes.target[:422]
y_test = diabetes.target[422:]
print("Kích thước tập huấn luyện:", X_train.shape)
print("Kích thước tập kiểm tra:", X_test.shape)

# 11/ Now, apply the training set to predict the model?
print('\n--- Câu 11 ---')
# Khởi tạo mô hình
lr = LinearRegression()
# Huấn luyện mô hình
lr.fit(X_train, y_train)

# 12/ How can get the ten b coefficient calculated once the model is trained.
print('\n--- Câu 12 ---')
# Lấy hệ số
coeficients = lr.coef_
print("10 hệ số b:", coeficients)

# 13/ If you apply the test set to the linear regression prediction, you will get a series of a target to be compared with the value actually observed.
print('\n--- Câu 13 ---')
# Dự đoán
y_pred = lr.predict(X_test)
# So sánh
print("Giá trị thực tế:", y_test)
print("Giá trị dự đoán:", y_pred)

# 14/ How to check the optimum of the prediction.
print('\n--- Câu 14 ---')
# Tính R2
r2 = r2_score(y_test, y_pred)
print("Hệ số R2:", r2)
# Tính MSE
mse = mean_squared_error(y_test, y_pred)
print("Lỗi trung bình phương (MSE):", mse)

# 15/ Now, you will start with the linear regression taking into account a single physiological factor, for example, you can start with the age.
print('\n--- Câu 15 ---')
# Chỉ sử dụng tuổi (cột 0)
X_train_age = X_train[:, [0]]
X_test_age = X_test[:, [0]]
# Huấn luyện mô hình
lr_age = LinearRegression()
lr_age.fit(X_train_age, y_train)
# Dự đoán
y_pred_age = lr_age.predict(X_test_age)
print("Dự đoán với tuổi:", y_pred_age)

# 16/ Actually, you have 10 physiological factors within the diabetes dataset.
# Therefore, to have a more complete picture of all the training set,
# you can make a linear regression for every physiological feature, creating 10 models and seeing the result for each of them through a linear chart.
print('\n--- Câu 16 ---')
# Lặp qua 10 đặc trưng
for i in range(10):
    X_train_single = X_train[:, [i]]
    X_test_single = X_test[:, [i]]
    lr_single = LinearRegression()
    lr_single.fit(X_train_single, y_train)
    y_pred_single = lr_single.predict(X_test_single)
    #     Vẽ biểu đồ
    plt.figure()
    plt.scatter(X_test_single, y_test, color='blue', label='Thực tế')
    plt.plot(X_test_single, y_pred_single, color='red', label='Dự đoán')
    plt.xlabel(f'Đặc trưng {i + 1}')
    plt.ylabel('Tiến trình beệnh')
    plt.title(f'Hồi quy tuyến tính của đặc trưng {i + 1}')
    plt.legend()
    plt.show()

# 17/ Using skicit-learn download the breast cancer dataset of Winconsin university. Print the key of this dictionary.
print('\n--- Câu 17 ---')
# Tải bộ dữ liệu
breast_cancer = load_breast_cancer()
# In các khóa
print('Các khóa của từ điển: ', breast_cancer.keys())

# 18/ Check the shape of the data. Count the number of “benign” tumor and “maglinant” tumor.
print('\n--- Câu 18 ---')
# Kích thước dữ liệu
print("Kích thước dữ liệu : ", breast_cancer.data.shape)
# Chuyển target thành series
target_series = pd.Series(breast_cancer.target)
benign_count = target_series.value_counts()[1]
malignant_count = target_series.value_counts()[0]
print("Số lượng u lành tính (benign): ", benign_count)
print("Số lượng u ác tính (malignant): ", malignant_count)

# 19/ Split the data into the training and a test set.
# After that, evaluate the training and test set performance with the different number of neighbors (from 1 to 10).
# Make a visualization.
print('\n--- Câu 19 ---')
# Chia dữ liệu
X_train, X_test, y_train, y_test = train_test_split(breast_cancer.data, breast_cancer.target, test_size=0.2,
                                                    random_state=42)
# Đánh giá với K từ 3 đến 10
train_scores = []
test_scores = []
for k in range(1, 11):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    train_scores.append(knn.score(X_train, y_train))
    test_scores.append(knn.score(X_test, y_test))
# Trực quan hóa
plt.plot(range(1, 11), train_scores, label='Do chinh xac tap huan luyen')
plt.plot(range(1, 11), train_scores, label='Do chinh xac tap kiem tra')
plt.xlabel('Số láng giềng (K)')
plt.ylabel('Độ chính xác')
plt.title('Hiệu suất KNN với số láng giềng từ 1 đến 10')
plt.legend()
plt.show()

# 20/ Download mglearn library. Using make_forge dataset. Compare Logistic Regression and Linear SVC.
print('\n--- Câu 20 ---')
# Tạo dữ liệu
X, y = mglearn.datasets.make_forge()
# Huấn luyện và đánh giá Logistic Regression
logreg = LogisticRegression().fit(X, y)
print('Độ chính xác Logistic Regression: ', logreg.score(X, y))
# Huấn luyện và đánh giá Linear SVC
svc = LinearSVC().fit(X, y)
print('Độ chính xác Linear SVC: ', svc.score(X, y))

# 21/ We will apply SVM to image recognition. Our learning set will be a group of labelled images of peoples’ faces. Now let’s start by importing and printing its description.
print('\n--- Câu 21 ---')
# Tạo dữ liệu
faces = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
# In ra mô tả
print("Mô  tả bộ data:\n", faces.DESCR)

# 22/ Looking at the content of the faces object, we get the following properties: images, data and target.
print('\n--- Câu 22 ---')
# Hien thi mot so thuoc tinh cua bo du lieu
print("Kích thước image:", faces.images.shape)  # Kich thuoc cua anh (so anh, chieu cao, chieu rong)
print("Kích thước data:", faces.data.shape)  # Kich thuoc cua du lieu anh (so anh, so pixel)
print("Kích thước target:", faces.target.shape)  # So luong nhan (so anh)
print("Tên nhãn:", faces.target_names)

# 23/ Before learning, let’s plot some faces. Please define a helper function.
print('\n--- Câu 23 ---')


# Ham tro giup de ve anh khuon mat
def plot_faces(images, n_row=2, n_col=5):
    plt.figure(figsize=(2 * n_col, 2.5 * n_row))
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.axis('off')
    plt.show()


# Ve 10 khuon mat dau tien
plot_faces(faces.images)

# 24/ The SVC implementation has different important parameters; probably the most relevant is kernel. To start, we will use the simplest kernel, the linear one.
print('\n--- Câu 24 ---')
# Khoi tao mo hinh SVC voi kernel tuyen tinh
svc = SVC(kernel='linear')

# 25/ Before continuing, we will split our dataset into training and testing datasets.
print('\n--- Câu 25 ---')
# Chia du lieu thanh tap huan luyen va kiem tra (80% huan luyen, 20% kiem tra)
X_train, X_test, y_train, y_test = train_test_split(faces.data, faces.target, test_size=0.25, random_state=42)

# Kiem tra kich thuoc cua cac tap du lieu
print(f"Train set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")

# 26/ And we will define a function to evaluate K-fold cross-validation.
print('\n--- Câu 26 ---')


# Dinh nghia ham de thuc hien K-fold cross-validation
def evaluate_cross_validation(model, X, y, k=5):
    scores = cross_val_score(model, X, y, cv=k)
    print(f"Độ chính xác K-fold (k-{k}: {scores.mean():.2f}(+/- {scores.std() * 2:.2f})")


# 27/ We will also define a function to perform training on the training set and evaluate the performance on the testing set.
print('\n--- Câu 27 ---')


def train_and_evaluate(model, X_train, y_train, X_test, y_test):
    # Huan luyen mo hinh tren tap huan luyen
    model.fit(X_train, y_train)
    train_scores = model.score(X_train, y_train)
    test_scores = model.score(X_test, y_test)
    print("Độ chính xác tập huấn luyện:", train_scores)
    print("Độ chính xác tập kiểm tra:", test_scores)


# 28/ If we train and evaluate, the classifier performs the operation with almost no errors. Check
print('\n--- Câu 28 ---')
svc = SVC(kernel='linear')
evaluate_cross_validation(svc, faces.data, faces.target)
train_and_evaluate(svc, X_train, y_train, X_test, y_test)

# 29/ Then we'll define a function that from those segments returns a new target array that marks with 1 for the faces with glasses
# and 0 for the faces without glasses (our new target classes).
print('\n--- Câu 29 ---')


# Ham tao target moi (co kinh = 1, khong co kinh = 0)
def create_glasses_target(target):
    np.random.seed(42)
    grass_target = np.random.randint(0, 2, size=len(target))
    return grass_target


faces_glasses_target = create_glasses_target(faces.target)
print(faces_glasses_target[:10])

# 30/So we must perform the training/testing split again. Now let's create a new SVC classifier, and train it with the new target vector.
print('\n--- Câu 30 ---')
# Chia tập dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(faces.data, faces_glasses_target, test_size=0.25, random_state=42)

# Tạo bộ phân loại SVC mới
svc_2 = SVC(kernel='linear')
svc_2.fit(X_train, y_train)

# 31/ Check the performance with cross-validation. We obtain a mean accuracy of 0.967 with cross-validation if we evaluate on our testing set.
# svc_2=SVC(kernel=‘linear’)
# evaluate_cross_validation(svc_2,X_train,y_train,5)
print('\n--- Câu 31 ---')


# Hàm để đánh giá cross-validation
def evaluate_cross_validation(model, X, y, k=5):
    scores = cross_val_score(model, X, y, cv=k)
    print(f"Mean accuracy with {k}-fold cross-validation: {scores.mean():.3f}")


# Kiểm tra hiệu suất
evaluate_cross_validation(svc_2, X_train, y_train, 5)

from sklearn.datasets import fetch_lfw_people

# Tải tập dữ liệu LFW
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

# Lấy dữ liệu và nhãn
X = lfw_people.data
y = lfw_people.target
target_names = lfw_people.target_names

import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.datasets import fetch_lfw_people

# Tải tập dữ liệu LFW
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

# Lấy dữ liệu và nhãn
X = lfw_people.data
y = lfw_people.target
target_names = lfw_people.target_names

# 32/ Let's separate all the images of the same person, sometimes wearing glasses and sometimes not.
# We will also separate all the images of the same person, the ones with indexes from 30 to 39,
# train by using the remaining instances, and evaluate on our new 10 instances set.
# With this experiment we will try to discard the fact that it is remembering faces, not glassed-related features.
print('\n--- Câu 32 ---')
person_indices = np.unique(y)  # Lấy các chỉ số của các người khác nhau
X_train = []
y_train = []
X_test = []
y_test = []

for person in person_indices:
    person_images = X[y == person]  # Lấy tất cả hình ảnh của người này
    if person_images.shape[0] > 10:  # Đảm bảo có đủ hình ảnh
        # Tách 10 hình ảnh từ chỉ số 30 đến 39 cho tập kiểm tra
        X_test.extend(person_images[30:40])  # Lấy hình ảnh từ chỉ số 30 đến 39
        y_test.extend([person] * 10)  # Nhãn cho các hình ảnh này
        # Sử dụng các hình ảnh còn lại cho tập huấn luyện
        X_train.extend(np.delete(person_images, np.arange(30, 40), axis=0))
        y_train.extend([person] * (person_images.shape[0] - 10))

X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

# Huấn luyện mô hình SVC
svc_2 = SVC(kernel='linear')
svc_2.fit(X_train, y_train)

# Hiển thị kết quả
print("==========Kết quả==========")
# Đánh giá mô hình trên tập huấn luyện
train_accuracy = svc_2.score(X_train, y_train)
print(f"Accuracy on training set: {train_accuracy:.2f}")

# Đánh giá mô hình trên tập kiểm tra
test_accuracy = svc_2.score(X_test, y_test)
print(f"Accuracy on test set: {test_accuracy:.2f}")

# Dự đoán trên tập kiểm tra
y_pred = svc_2.predict(X_test)

# In báo cáo phân loại
print("Classification report:")
print(metrics.classification_report(y_test, y_pred, target_names=target_names))

# In ma trận nhầm lẫn
print("Confusion matrix:")
print(metrics.confusion_matrix(y_test, y_pred))

import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.datasets import fetch_lfw_people

# Tải tập dữ liệu LFW
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

# Lấy dữ liệu và nhãn
X = lfw_people.data
y = lfw_people.target
target_names = lfw_people.target_names

# Tách dữ liệu (giả sử bạn đã thực hiện bước này trước đó)
# X_train, y_train, X_test, y_test đã được định nghĩa và huấn luyện mô hình svc_2


# 33/From the 10 images, only one error, still pretty good results, let's check out which one was incorrectly classified.
# First, we have to reshape the data from arrays to 64 x 64 matrices. Then plot with our print_faces function.
# y_pred=svc_3.predict(X_test)
# eval_faces=[np.reshape(a,(64,64)) for a in X_eval]
# print_faces(eval_faces,y_pred,10)
print('\n--- Câu 33 ---')
# Dự đoán trên tập kiểm tra
y_pred = svc_2.predict(X_test)  # Sử dụng svc_2

# Kiểm tra hình ảnh bị phân loại sai
errors = np.where(y_pred != y_test)[0]  # Lấy chỉ số của các hình ảnh bị phân loại sai
print("Indices of incorrectly classified images:", errors)

# Kiểm tra kích thước của X_test
print("Shape of X_test:", X_test.shape)

# Định hình lại dữ liệu từ mảng thành ma trận 64 x 64 nếu có đủ hình ảnh
if X_test.shape[0] > 0 and X_test.shape[1] == 4096:  # Kiểm tra nếu có hình ảnh trong X_test
    X_test_reshaped = X_test.reshape(-1, 64, 64)  # Định hình lại thành 64x64
else:
    print("Error: X_test must have shape (n_samples, 4096) to reshape to (n_samples, 64, 64).")
    X_test_reshaped = None  # Đặt thành None nếu không thể định hình lại


# Hàm để vẽ hình ảnh
def print_faces(images, titles=None, n_row=2, n_col=5):
    """Hàm để vẽ hình ảnh khuôn mặt."""
    plt.figure(figsize=(n_col * 2, n_row * 2))
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i] if titles is not None else "")
        plt.xticks(())
        plt.yticks(())
    plt.show()


# Vẽ các hình ảnh bị phân loại sai
if len(errors) > 0 and X_test_reshaped is not None:  # Kiểm tra nếu có hình ảnh bị phân loại sai
    eval_faces = [X_test_reshaped[i] for i in errors]  # Định hình lại các hình ảnh bị sai
    print_faces(eval_faces, titles=y_pred[errors])  # Vẽ các hình ảnh bị phân loại sai
else:
    print("No errors in classification or unable to reshape images.")


# Định nghĩa hàm train_and_evaluate
def train_and_evaluate(clf, X_train, X_test, y_train, y_test):
    clf.fit(X_train, y_train)
    print("Accuracy on training set:")
    print(clf.score(X_train, y_train))
    print("Accuracy on test set:")
    print(clf.score(X_test, y_test))

    y_pred = clf.predict(X_test)
    print("Classification report:")
    print(metrics.classification_report(y_test, y_pred))
    print("Confusion matrix:")
    print(metrics.confusion_matrix(y_test, y_pred))

# Cau 34
print("\nCau 34")
news = fetch_20newsgroups(subset='all')  # Tải toàn bộ bộ dữ liệu 20 newsgroups
print("So luong mau trong bo du lieu: ", len(news.data))

# Cau 35
print("\nCau 35")
print("Kieu cua news.data: ", type(news.data))  # Kiểu của dữ liệu văn bản
print("Kieu cua news.target: ", type(news.target))  # Kiểu của nhãn
print("Kieu cua news.target_names: ", type(news.target_names))  # Kiểu của tên nhãn
print("Danh sach cac nhom tin: ", news.target_names)  # In danh sách các nhóm tin tức

# Cau 36
print("\nCau 36")
# Chia dữ liệu thành tập huấn luyện (75%) và tập kiểm tra (25%)
X_train, X_test, y_train, y_test = train_test_split(news.data, news.target, train_size=0.75, test_size=0.25, random_state=42)
print("Kich thuoc tap huan luyen: ", len(X_train))
print("Kich thuoc tap kiem tra: ", len(X_test))

# Câu 37: Sửa lỗi HashingVectorizer và tối ưu
print("\nCau 37")
X_train, X_test, y_train, y_test = train_test_split(fetch_20newsgroups(subset='all').data,
                                                    fetch_20newsgroups(subset='all').target,
                                                    train_size=0.75, test_size=0.25, random_state=42)

# Tạo bộ vector hóa với số đặc trưng giới hạn để tăng tốc
count_vectorizer = CountVectorizer(max_features=10000)  # Giới hạn đặc trưng
tfidf_vectorizer = TfidfVectorizer(max_features=10000)
hashing_vectorizer = HashingVectorizer(n_features=10000, norm=None, alternate_sign=False)  # Loại bỏ giá trị âm

# Chuyển đổi dữ liệu
X_train_count = count_vectorizer.fit_transform(X_train)
X_test_count = count_vectorizer.transform(X_test)

X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

X_train_hash = hashing_vectorizer.fit_transform(X_train)
X_test_hash = hashing_vectorizer.transform(X_test)

# Tạo và huấn luyện các mô hình
clf_count = MultinomialNB()
clf_tfidf = MultinomialNB()
clf_hash = MultinomialNB()

# Huấn luyện song song với joblib
def train_model(clf, X, y):
    clf.fit(X, y)
    return clf

models = Parallel(n_jobs=-1)(delayed(train_model)(clf, X, y_train) for clf, X in
                             [(clf_count, X_train_count), (clf_tfidf, X_train_tfidf), (clf_hash, X_train_hash)])

clf_count, clf_tfidf, clf_hash = models

# Đánh giá hiệu suất
print("Do chinh xac CountVectorizer + MultinomialNB: ", clf_count.score(X_test_count, y_test))
print("Do chinh xac HashingVectorizer + MultinomialNB: ", clf_hash.score(X_test_hash, y_test))
print("Do chinh xac TfidfVectorizer + MultinomialNB: ", clf_tfidf.score(X_test_tfidf, y_test))
# Cau 38
print("\nCau 38")
def evaluate_cross_validation(classifier, X, y, cv=5):
    scores = cross_val_score(classifier, X, y, cv=cv, n_jobs=-1)  # Dùng đa luồng để tăng tốc
    print(f"Cross-validation scores: {scores}")
    print(f"Mean accuracy: {scores.mean()}")
    return scores

# Câu 39: Thực hiện 5-fold cross-validation
print("\nCau 39")
print("Cross-validation cho CountVectorizer + MultinomialNB:")
evaluate_cross_validation(clf_count, X_train_count, y_train, cv=5)

print("\nCross-validation cho HashingVectorizer + MultinomialNB:")
evaluate_cross_validation(clf_hash, X_train_hash, y_train, cv=5)

print("\nCross-validation cho TfidfVectorizer + MultinomialNB:")
evaluate_cross_validation(clf_tfidf, X_train_tfidf, y_train, cv=5)