import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Tải dữ liệu từ tệp CSV
df = pd.read_csv('heart.csv')

# Hiển thị 5 dòng đầu tiên của dữ liệu
print(df.head())  

# Kiểm tra thông tin và kiểu dữ liệu của các cột
print(df.info())  

# Kiểm tra các giá trị bị thiếu trong dữ liệu
print(df.isnull().sum())  

# Phân tích thống kê mô tả của dữ liệu
print(df.describe())  

# Phân phối độ tuổi
plt.figure(figsize=(12, 6))  
sns.histplot(df['age'], bins=30, kde=True, color='skyblue')
plt.title('Phân phối độ tuổi', fontsize=16)
plt.xlabel('Tuổi', fontsize=12)
plt.ylabel('Tần suất', fontsize=12)
plt.grid()
plt.savefig('PhanPhoiTuoi.png') 
plt.show()  

# Mối quan hệ giữa tuổi và nhịp tim tối đa
plt.figure(figsize=(12, 6))  
sns.scatterplot(data=df, x='age', y='thalach', hue='target', alpha=0.7)
plt.title('Mối quan hệ giữa tuổi và nhịp tim tối đa', fontsize=16)
plt.xlabel('Tuổi', fontsize=12)
plt.ylabel('Nhịp tim tối đa', fontsize=12)
plt.legend(title='Bệnh tim', labels=['Không', 'Có'])
plt.grid()
plt.savefig('TuoiVaNhipTim.png')  
plt.show()  

# Ma trận phân tán giữa các đặc trưng
sns.pairplot(df, hue='target', palette='husl')  
plt.suptitle('Ma trận phân tán giữa các đặc trưng', y=1.02)
plt.savefig('MatranPhanTan.png')  
plt.show()  

# Phân phối giới tính
plt.figure(figsize=(6, 4))  
sns.countplot(x='sex', data=df, palette='pastel')
plt.title('Phân phối giới tính', fontsize=16)
plt.xlabel('Giới tính (0: Nữ, 1: Nam)', fontsize=12)
plt.ylabel('Số lượng', fontsize=12)
plt.grid()
plt.savefig('PhanPhoiGioiTinh.png')  
plt.show()  

# Tỷ lệ bệnh tim theo giới tính
plt.figure(figsize=(6, 4))  
sns.countplot(x='sex', hue='target', data=df, palette='pastel')
plt.title('Tỷ lệ bệnh tim theo giới tính', fontsize=16)
plt.xlabel('Giới tính (0: Nữ, 1: Nam)', fontsize=12)
plt.ylabel('Số lượng', fontsize=12)
plt.legend(title='Bệnh tim', labels=['Không', 'Có'])
plt.grid()
plt.savefig('TyLeBenhTimTheoGioiTinh.png')  
plt.show()  

# Phân phối bệnh tim theo độ tuổi
plt.figure(figsize=(12, 6))  
sns.boxplot(x='target', y='age', data=df, palette='Set2')
plt.title('Phân phối bệnh tim theo độ tuổi', fontsize=16)
plt.xlabel('Bệnh tim (0: Không, 1: Có)', fontsize=12)
plt.ylabel('Tuổi', fontsize=12)
plt.grid()
plt.savefig('PhanPhoiBenhTimTheoTuoi.png')  
plt.show()  

# Phân tích mối quan hệ giữa nhịp tim tối đa và bệnh tim
plt.figure(figsize=(12, 6))  
sns.boxplot(x='target', y='thalach', data=df, palette='Set1')
plt.title('Phân tích mối quan hệ giữa nhịp tim tối đa và bệnh tim', fontsize=16)
plt.xlabel('Bệnh tim (0: Không, 1: Có)', fontsize=12)
plt.ylabel('Nhịp tim tối đa', fontsize=12)
plt.grid()
plt.savefig('NhipTimVaBenhTim.png')  
plt.show()  

# Phân tích mối quan hệ giữa mức cholesterol và bệnh tim
plt.figure(figsize=(12, 6))  
sns.boxplot(x='target', y='chol', data=df, palette='Set3')
plt.title('Mối quan hệ giữa mức cholesterol và bệnh tim', fontsize=16)
plt.xlabel('Bệnh tim (0: Không, 1: Có)', fontsize=12)
plt.ylabel('Mức cholesterol', fontsize=12)
plt.grid()
plt.savefig('MucCholesterolVaBenhTim.png')  
plt.show()  # Dòng 101

# Tạo heatmap để xem mối tương quan giữa các biến
plt.figure(figsize=(10, 8))  
correlation = df.corr()
sns.heatmap(correlation, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Heatmap mối tương quan giữa các biến', fontsize=16)
plt.savefig('HeatmapTuongQuan.png')  
plt.show()  
