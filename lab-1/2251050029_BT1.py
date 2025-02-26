# 1/ Print all characters in “your name” by using for loops

print("Bài 1")
name = "Tran Trung Hau"
for char in name:
    print(char)

# 2/ Print all odd numbers x such that 1<=x<=10
print("\nBài 2")
for x in range(1, 11, 2):
    print(x)

# 3/ a/Compute the sum of all numbers in 2/
print("\nBài 3.a")
tong = 0
for x in range(1, 11, 2):
    tong += x

print("Tổng các số lẻ từ 1-10:", tong)

#     b/ Compute the sum of all number from 1 to 6.
print("\nBài 3.b")
sum = 0
for i in range(1, 7):
    sum += i

print("Tổng các số từ 1-6:", sum)

# 4/ Given mydict={“a”: 1,”b”:2,”c”:3,”d”:4}.
mydict = {"a": 1, "b": 2, "c": 3, "d": 4}
# a/ Print all key in mydict
print("\nBài 4.a")
for key in mydict:
    print(key)

# b/ Print all values in mydict
print("\nBài 4.b")
for value in mydict.values():
    print(value)

# c/ Print all keys and values
print("\nBài 4.c")
for key, value in mydict.items():
    print(key, value)

# 5/ Given courses=[131,141,142,212] and names=[“Maths”,”Physics”,”Chem”, “Bio”]. Print a sequence of tuples, each of them contains one courses and one names
print("\nBài 5")
courses = [131, 141, 142, 212]
names = ["Maths", "Physics", "Chem", "Bio"]
zipped = zip(courses, names)
print(list(zipped))

# 6/ Find the number of consonants in “jabbawocky” by two ways
words = "jabbawocky"
vowels = "aeiou"
count = 0
# 	a/ Directly (i.e without using the command “continue”)
print("\nBài 6.a")
for char in words:
    if char not in vowels:
        count += 1

print("Số phụ âm:", count)

# 	b/ Check whether it’s characters are in vowels set and using the command “continue”
print("\nBài 6.b")
count = 0
for char in words:
    if char in vowels:
        continue
    count += 1

print("Số phụ âm:", count)

# 7/ a is a number such that -2<=a<3. Print out all the results of 10/a using try…except. When a=0, print out “can’t divided by zero”
print("\nBài 7")
for a in range(-2, 3):
    try:
        print(10 / a)
    except ZeroDivisionError:
        print("can’t divided by zero")

# 8/ Given ages=[23,10,80]
# And names=[Hoa,Lam,Nam]. Using lambda function to sort a list containing tuples (“age”,”name”) by increasing of the ages
ages = [23, 10, 80]
names = ['Hoa', 'Lam', 'Nam']
data = zip(ages, names)
sorted_age = sorted(data, key=lambda x: x[0])
print(sorted_age)

# 9/ Create  a file “firstname.txt”:
# a/ Open this file for reading
input_file = open("firstname.txt")

# b/Print each line of this file
for line in input_file:
    print(line, end="")

# c/ Using .read to read the file and
# Print it.
contents = input_file.read()
print(contents)
input_file.close()


# 1/ Define a function that return the sum of two numbers a and b. Try with a=3, b=4.
def sum(a, b):
    return a + b


print(sum(3, 4))

# 2/ Create a 3x3 matrix M=(1 2 3   and vector v=(1 2 3)
#                           4 5 6
#                           7 8 9)
# And check the rank and the shape of this matrix and vector v.
import numpy as np

M = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
v = np.array([1, 2, 3])

rank_M = np.linalg.matrix_rank(M)
rank_v = np.linalg.matrix_rank(v)

print("Rank of matrix M:", rank_M)
print("Shape of matrix M:", M.shape)
print(M)

print("\nRank of matrix v:", rank_M)
print("Shape of matrix v:", v.shape)
print(v)

# 3/ Create a new 3x3 matrix such that its’ elements are the sum of corresponding (position) element of M plus 3.
print("\nMatrix M plus 3")
M = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(M + 3)

# 4/ Create the transpose of M and v
M = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
v = np.array([1, 2, 3])

M_transpose = M.T
v_transpose = v.T

print(M)
print("Transpose of M:")
print(M_transpose)

print(v)
print("Transpose of v:")
print(v_transpose)

# 5/ Compute the norm of x=(2,7). Normalization vector x.
x = np.array([2, 7])
norm_x = np.linalg.norm(x)
normalized_x = x / norm_x

print("\nNorm of x:", norm_x)
print("Normalized vector x:", normalized_x)

# 6/ Given a=[10,15], b=[8,2] and c=[1,2,3]. Compute a+b, a-b, a-c. Do all of them work? Why?
a = np.array([10, 15])
b = np.array([8, 2])
c = np.array([1, 2, 3])
# a + b
try:
    a_plus_b = a + b
    print("a + b:", a_plus_b)
except ValueError as e:
    print("Error in a + b:", e)
# a - b
try:
    a_minus_b = a - b
    print("a - b:", a_minus_b)
except ValueError as e:
    print("Error in a - b:", e)
# a - c
try:
    a_minus_c = a - c
    print("a - c:", a_minus_c)
except ValueError as e:
    print("Error in a - c:", e)

# 7/ Compute the dot product of a and b.
a = np.array([10, 15])
b = np.array([8, 2])

dot_product = np.dot(a, b)

print("\nThe dot product of a and b:", dot_product)

# 8/ Given matrix A=[[2,4,9],[3,6,7]]
A = np.array([[2, 4, 9], [3, 6, 7]])
# 	a/ Check the rank and shape of A
rank_A = np.linalg.matrix_rank(A)
shape_A = A.shape

print("\nRank of A:", rank_A)
print("Shape of A:", shape_A)
print(A)
# 	b/ How can get the value 7 in A?
value_7 = A[1, 2]  # Số 7 ở dòng 1 cột 2
print(value_7)
# 	c/ Return the second column of A.
second_column = A[:, 1]
print("\nThe second column of A:", second_column)

# 9/ Create a random  3x3 matrix  with the value in range (-10,10).
random_matrix = np.random.randint(-10, 11, size=(3, 3))

print("\nRandom 3x3 matrix:\n", random_matrix)

# 10/ Create an identity (3x3) matrix.
identity_matrix = np.eye(3)

print("\nIdentity 3x3 matrix:\n", identity_matrix)

# 11/ Create a 3x3 random matrix with the value in range (1,10). Compute the trace of this matrix by 2 ways:
random_matrix = np.random.randint(1, 11, size=(3, 3))

print("\nRandom 3x3 matrix:\n", random_matrix)
# 	a/ By one command
trace_matrix = np.trace(random_matrix)

print("\nTrace of matrix:", trace_matrix)
# 	b/ By using for loops
trace_loop = 0
for i in range(random_matrix.shape[0]):
    trace_loop += random_matrix[i, i]

print("\nTrace of matrix:", trace_loop)

# 12/ Create a 3x3 diagonal matrix with the value in main diagonal 1,2,3.
diagonal_matrix = np.diag([1, 2, 3])

print("\n3x3 Diagonal matrix:\n", diagonal_matrix)

# 13/ Given A=[[1,1,2],[2,4,-3],[3,6,-5]]. Compute the determinant of A
A = np.array([[1, 1, 2], [2, 4, -3], [3, 6, -5]])

det_A = np.linalg.det(A)

print("\nDet A:", det_A)

# 14/ Given a1=[1,-2,-5] and a2=[2,5,6]. Create a matrix M such that the first column is a1 and the second column is a2.
a1 = np.array([1, -2, -5])
a2 = np.array([2, 5, 6])

M = np.column_stack((a1, a2))

print("Matrix M:")
print(M)

# 15/ Simply plot the value of the square of y with y in range (-5<=y<6).
import matplotlib.pyplot as plt
import numpy as np

# Định nghĩa phạm vi của y
y = np.arange(-5, 6)  # Các giá trị của y từ -5 đến 5 (bao gồm -5 và 5)

# Tính giá trị y^2
y_squared = y ** 2

# Vẽ đồ thị
plt.plot(y, y_squared, marker='o', color='b', label='y^2')

# Thêm nhãn và tiêu đề cho đồ thị
plt.xlabel('y')
plt.ylabel('y^2')
plt.title('Đồ thị của y^2 khi y trong khoảng (-5 <= y < 6)')
plt.grid(True)
plt.legend()

# Hiển thị đồ thị
plt.show()


# 16/ Create 4-evenly-spaced values between 0 and 32 (including endpoints)
values = np.linspace(0, 32, 4)

# In kết quả
print("4 giá trị la:", values)

# 17/ Get 50 evenly-spaced values from -5 to 5 for x. Calculate y=x**2. Plot (x,y).
import matplotlib.pyplot as plt
import numpy as np

# Lấy 50 giá trị đều đặn từ -5 đến 5 cho x
x = np.linspace(-5, 5, 50)

# Tính y = x^2
y = x ** 2

# Vẽ đồ thị
plt.plot(x, y, marker='o', color='r', label='y = x^2')

# Thêm nhãn và tiêu đề cho đồ thị
plt.xlabel('x')
plt.ylabel('y')
plt.title('Đồ thị của y = x^2 từ x = -5 đến x = 5')
plt.grid(True)
plt.legend()

# Hiển thị đồ thị
plt.show()

# 18/ Plot y=exp(x) with label and title.
import matplotlib.pyplot as plt

# Tạo giá trị x từ -2 đến 2 (có thể thay đổi tùy ý)
x = np.linspace(-2, 2, 100)

# Tính y = exp(x)
y = np.exp(x)

# Vẽ đồ thị y = exp(x)
plt.plot(x, y, label="y = exp(x)", color='b')  # Đặt màu đường vẽ là màu xanh

# Thêm nhãn và tiêu đề
plt.xlabel("x")
plt.ylabel("y")
plt.title("Plot of y = exp(x)")

# Hiển thị chú thích và lưới
plt.legend()
plt.grid(True)

# Hiển thị đồ thị
plt.show()

# 19/ Similarly for y=log(x) with x from 0 to 5
# Tạo giá trị x từ 0.1 đến 5 để tránh lỗi log(0)
x = np.linspace(0.1, 5, 100)

# Tính y = log(x) (logarit tự nhiên)
y = np.log(x)

# Vẽ đồ thị y = log(x)
plt.plot(x, y, label="y = log(x)", color='r')  # Đặt màu đường vẽ là màu đỏ

# Thêm nhãn và tiêu đề
plt.xlabel("x")
plt.ylabel("y")
plt.title("Plot of y = log(x) for x in range [0.1, 5]")

# Hiển thị chú thích và lưới
plt.legend()
plt.grid(True)

# Hiển thị đồ thị
plt.show()

# 20/ Draw two graphs y=exp(x), y=exp(2*x) in the same graph and y=log(x) and y=log(2*x) in the same graph using subplot.

# Giá trị x
x = np.linspace(0.1, 5, 400)

# Hàm
y1 = np.exp(x)
y2 = np.exp(2 * x)
y3 = np.log(x)
y4 = np.log(2 * x)

# Create subplots
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Plot for y = exp(x) and y = exp(2*x)
axs[0].plot(x, y1, label='y = exp(x)', color='blue')
axs[0].plot(x, y2, label='y = exp(2x)', color='red')
axs[0].set_title('y = exp(x) and y = exp(2x)')
axs[0].set_xlabel('x')
axs[0].set_ylabel('y')
axs[0].legend()

# Plot for y = log(x) and y = log(2x)
axs[1].plot(x, y3, label='y = log(x)', color='green')
axs[1].plot(x, y4, label='y = log(2x)', color='orange')
axs[1].set_title('y = log(x) and y = log(2x)')
axs[1].set_xlabel('x')
axs[1].set_ylabel('y')
axs[1].legend()

# Show the plot
plt.tight_layout()
plt.show()












