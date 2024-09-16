import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors

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
fw_features = fw_data[['Name', 'Club', 'Position', 'Age', 'Goals']]

# 5. กำหนดโมเดล KNN
k = 7
knn_model = NearestNeighbors(n_neighbors=k)
knn_model.fit(fw_features[['Age', 'Goals']])

# 6. กำหนดค่าตัวอย่างของผู้เล่นที่ต้องการค้นหา
age_fw = 24
goal_fw = 50
input_player = [[age_fw, goal_fw]]

# ค้นหานักเตะที่ใกล้เคียงที่สุด
distances, indices = knn_model.kneighbors(input_player)

# ดึงนักเตะที่ใกล้เคียงที่สุด
similar_players = fw_features.iloc[indices[0]]

# 7. Plot กราฟ 2 มิติ (Age, Goals)
plt.figure(figsize=(8, 6))

# Plot จุดข้อมูลนักเตะ Forward ทุกคน
plt.scatter(fw_features['Age'], fw_features['Goals'], color='green', label='Forwards')

# Plot จุดนักเตะเป้าหมาย (input player)
plt.scatter(age_fw, goal_fw, color='blue', label='Input Player', marker='X', s=100)

# Plot เพื่อนบ้านที่ใกล้ที่สุด
for i in range(k):
    plt.scatter(similar_players.iloc[i]['Age'], similar_players.iloc[i]['Goals'], color='red', label=f'Neighbor {i+1}', edgecolor='black')

# เพิ่มวงกลมล้อมรอบเพื่อนบ้านที่ใกล้ที่สุด
for i in range(k):
    circle = plt.Circle((age_fw, goal_fw), distances[0][i], color='gray', fill=False, linestyle='--', edgecolor='black', linewidth=1.5)
    plt.gca().add_patch(circle)

# ตั้งค่าให้แกนมีสัดส่วนเท่ากันเพื่อให้วงกลมไม่แบน
plt.gca().set_aspect('equal', adjustable='datalim')

# จำกัดขอบเขตของแกน x และ y
plt.xlim(0, 50)  # อายุในช่วง 0 ถึง 50
plt.ylim(0, 70)   # จำนวนประตูในช่วง 0 ถึง 70

# กำหนด tick ของแกน x ให้แสดงเฉพาะช่วงที่ต้องการ
plt.xticks(np.arange(10, 51, 5))  # กำหนดช่วงแกน x ให้แสดงที่ 10, 15, 20, 25, 30, 35, 40, 45, 50

# ตกแต่งกราฟ
plt.title('K-Nearest Neighbors (KNN) Visualization')
plt.xlabel('Age')
plt.ylabel('Goals')
plt.legend(loc='upper right')
plt.grid(True)

# แสดงผลกราฟ
plt.show()
