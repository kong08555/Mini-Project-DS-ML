import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# 1. โหลดข้อมูลจากไฟล์
data = pd.read_csv('dataset/EPL-players-stats-2020.csv')

# 2. สร้างคอลัมน์ใหม่ 'Position Group' ให้มีแค่ 4 ตำแหน่งหลัก
def categorize_position(position):
    if position == 'Goalkeeper':
        return 'Goalkeeper'
    elif position == 'Defender':
        return 'Defender'
    elif position == 'Midfielder':
        return 'Midfielder'
    elif position == 'Forward':
        return 'Forward'
    else:
        return 'Unknown'

# 3. ใช้ฟังก์ชัน categorize_position กับคอลัมน์ 'Position'
data['Position Group'] = data['Position'].apply(categorize_position)

# 4. กรองข้อมูลเฉพาะนักเตะตำแหน่ง Forward
fw_data = data[data['Position Group'] == 'Forward']
fw_features = fw_data[['Age', 'Goals']]

# 5. กำหนดโมเดล KNN (ใช้ KNeighborsClassifier เพื่อสร้าง decision boundary)
k = 7
knn_model = KNeighborsClassifier(n_neighbors=k)
knn_model.fit(fw_features[['Age', 'Goals']], np.zeros(len(fw_features)))  # เราไม่ได้ใช้ labels จึงใส่ array ของ 0

# 6. การสร้าง mesh grid เพื่อ plot decision boundary
x_min, x_max = fw_features['Age'].min() - 5, fw_features['Age'].max() + 5
y_min, y_max = fw_features['Goals'].min() - 5, fw_features['Goals'].max() + 5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.5), np.arange(y_min, y_max, 0.5))

# ทำนายค่าตาม grid points
Z = knn_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 7. Plot กราฟ decision boundary และจุดข้อมูลนักเตะ Forward ทุกคน
plt.figure(figsize=(8, 6))

# Plot decision boundary
plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')

# Plot จุดข้อมูลนักเตะทั้งหมด
plt.scatter(fw_features['Age'], fw_features['Goals'], color='green', label='Forwards')

# Plot จุดนักเตะเป้าหมาย (input player)
age_fw = 24
goal_fw = 50
plt.scatter(age_fw, goal_fw, color='blue', label='Input Player', marker='X', s=100)

# ค้นหานักเตะที่ใกล้เคียงที่สุด
distances, indices = knn_model.kneighbors([[age_fw, goal_fw]])
similar_players = fw_features.iloc[indices[0]]

# Plot เพื่อนบ้านที่ใกล้ที่สุด
for i in range(k):
    plt.scatter(similar_players.iloc[i]['Age'], similar_players.iloc[i]['Goals'], color='red', label=f'Neighbor {i+1}', edgecolor='black')

# เพิ่มวงกลมล้อมรอบเพื่อนบ้านที่ใกล้ที่สุด
for i in range(k):
    circle = plt.Circle((age_fw, goal_fw), distances[0][i], color='gray', fill=False, linestyle='--')
    plt.gca().add_patch(circle)

# ตกแต่งกราฟ
plt.title('KNN Decision Boundary Visualization')
plt.xlabel('Age')
plt.ylabel('Goals')
plt.legend(loc='upper right')
plt.grid(True)

# แสดงผลกราฟ
plt.show()
