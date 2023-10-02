# ------------------------- Thư Viện ------------------------- #
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

# ------------------------- Biến Toàn Cục ------------------------- #
df = None
d = None
t = None
w = None
selected_column_name = None
test_size = None
train = None
test = None
predict = None
actual = None
metrics = None

# ------------------------- Hàm ------------------------- #
# 1. Hàm load dữ liệu
# file_name: Tên tập dữ liệu
# Trả về một dataframe
@st.cache_data
def load_data(file_name):
    if file_name == 'A.csv':
        df = pd.read_csv('https://drive.google.com/u/1/uc?id=1qZ0O5iUHzZ0AzqGQpm5JxDlwmFEVTXpp&export=download')
    else:
        df = pd.read_csv('https://drive.google.com/u/1/uc?id=1tZKDCKpEzI9ZPxa7il64vxIIdbtpsW2e&export=download')
    return df

# 2. Hàm tạo ra ma trận giá dùng để dự đoán X và ma trận giá thực tế T
# d: Số ngày dùng để dự đoán
# t: Số ngày muốn để dự đoán
# data: Tập dữ liệu dùng để dự đoán
# column: Cột dữ liệu muốn dự đoán
# Trả về rả về ma trận X chứa giá trị để dự đoán và ma trận T chứa giá trị thực
@st.cache_data
def PrepareData(d, t, data, column):
    X = []
    T = []
    train_data = data[column]

    for i in range(len(train_data) - d - t + 1):
        X.append(train_data[i:i+d])
        T.append(train_data[i+d:i+d+t])

    X = np.array(X)
    T = np.array(T)
    return X, T


# 3. Hàm chuyển X thành dạng mở rộng đa thức theo bậc bất kỳ
# X: Ma trận chứa dữ liệu để dự đoán
# degree: Bậc của đa thức, mặc định là 2
# Trả ma trận M chứa các dữ liệu đã được biến đổi thành các đa thức.
@st.cache_data
def PolyExpansion(X, degree=2):
    poly = PolynomialFeatures(degree)
    M = poly.fit_transform(X)

    return M

# 4. Hàm tính trọng số w theo hàm loss Least Squared Error
@st.cache_data
def Model(d, t, data, column, degree):
    X, T = PrepareData(d, t, data, column)
    M = PolyExpansion(X, degree)
    s = np.dot(M.T, M) 
    if np.linalg.det(s) == 0:
        s += 1e-3 * np.identity(M.shape[1])

    w = []
    for i in range(t):
        w.append(np.dot(np.dot(np.linalg.inv(s), M.T), T[:, i]))

    return w

# 5. Hàm Test
@st.cache_data
def Test(w, d, t, data, column, degree):
    X, T = PrepareData(d, t, data, column)
    result = np.dot(PolyExpansion(X, degree), np.array(w).T)
    return result, T

# 5. Hàm dự đoán
def Predict():
    pass

# 6. Hàm đánh giá độ chính xác
def Score(predict, actual):
    mae = mean_absolute_error(actual, predict)
    mse = mean_squared_error(actual, predict)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(actual, predict)

    metrics = {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "MAPE": mape
    }

    return metrics


# 7. Hàm tạo tập training/testing
# data: Tập dữ liệu dùng để dự đoán
# column: Cột dữ liệu muốn dự đoán
# test_size: Tỉ lệ tập test
# Trả về mảng X_train kích thước n*d và Y_train kích thước n*t
def SplitData(data, column, test_size):
    train, test = train_test_split(data[column], test_size=test_size/100, shuffle=False)
    return pd.DataFrame(train), pd.DataFrame(test)


# ------------------------- Giao Diện ------------------------- #
st.title('Dự báo chứng khoán bằng Polynomial Classifier')

# Lấy tên của các tập dữ liệu
file_names = ['A.csv', 'HSI.csv']

# Chọn tập dữ liệu
selected_file_name = st.selectbox("Chọn tập dữ liệu:", file_names)

# Thông tin tập dữ liệu
st.subheader('Thông tin tập dữ liệu ' + selected_file_name)
df = load_data(selected_file_name)

# Loại bỏ cột Index có sẵn và thay cột Date thành cột Index chính
try:
    df = df.drop("Index", axis=1)
except KeyError:
    pass

df = df.set_index('Date')
df

# Vẽ biểu đồ đường cho tập dữ liệu
st.subheader('Biểu đồ đường tập dữ liệu ' + selected_file_name)

column_names = df.columns.tolist()
selected_column_name = st.selectbox("Chọn cột:", column_names)
st.line_chart(data=df, y=selected_column_name)

# Chọn tham số
st.subheader('Tạo mô hình dự đoán trên tập dữ liệu ' + selected_file_name)

selected_predict_column_name = st.selectbox(
    "Chọn cột để dự đoán:", column_names)

col1, col2 = st.columns(2)

with col1:
    d = st.number_input('Số ngày dùng để dự đoán:', value=2, step=1)

with col2:
    t = st.number_input('Số ngày muốn dự đoán:', value=1, step=1)

col11, col22 = st.columns(2)

with col11:
    train_size = st.number_input(
        'Chia tỉ lệ training %:', min_value=1, max_value=99, value=80, step=1)
    test_size = 100 - train_size
    st.text('Tỷ lệ testing: ' + str(test_size) + '%')

with col22:
    degree = st.number_input('Số bậc của đa thức: ', value=2, step=1)

# Chia tập training/testing
train, test = SplitData(df, selected_column_name, test_size)
st.write('Kích thước tập train: ', train.size,
         'Kích thước tập test: ', test.size)

# Training
if st.button("Train Mô Hình", type="primary"):
    with st.spinner('Mô hình đang được train, xin vui lòng đợi.'):

        w = Model(d, t, train, selected_column_name, degree)
        st.session_state.w = w

# In ma trận w ra màn hình
try:
    w = st.session_state.w
    st.write('Ma trận trọng số có dạng:')
    vmatrix_latex = "\omega = \\begin{vmatrix} "

    for i in range(len(w)):
        vmatrix_latex += " & ".join(["{:.4f}".format(x) for x in w[i]]) + " \\\\ "
    vmatrix_latex += " \\end{vmatrix}"

    st.latex(vmatrix_latex)
except AttributeError:
    w = None

# Testing
if st.button("Test Mô Hình", type="primary"):
    with st.spinner('Mô hình đang được test, xin vui lòng đợi.'):
        try:
            w = st.session_state.w
        
            predict, actual = Test(w, d, t, test, selected_column_name, degree)
            predict_list = predict.tolist()
            actual_list = actual.tolist()

            result_table = pd.DataFrame({"Dự đoán": predict_list[:5], "Thực tế": actual_list[:5]})
            st.table(result_table)
        except AttributeError:
            st.write('Vui lòng train tập dữ liệu trước khi test')
        
        
# Tính điểm
st.subheader('Các thông số đánh giá')
if predict is not None:
    metrics = Score(predict, actual)
    score_table = df = pd.DataFrame(metrics, index=["Thông số đánh giá"])
    st.table(score_table)

# Vẽ biểu đồ đường dự báo và thực tế
    st.subheader('Biểu đồ so sánh')

    plt.plot(predict, label="Dự đoán", linestyle='dashed')
    plt.plot(actual, label="Thực tế", linestyle='dashed')
    plt.legend()

    st.pyplot(plt)   

# Áp dụng mô hình cho tập dữ liệu khác

# So sánh mô hình với các bậc đa thức khác
 
# Nhập dữ liệu để dự đoán   
