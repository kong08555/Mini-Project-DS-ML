import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.neighbors import NearestNeighbors
import numpy as np

# predict goal: 

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
        return 'Unknown'  # ถ้ามีตำแหน่งที่ไม่อยู่ใน 4 ตำแหน่งหลัก

# 3. ใช้ฟังก์ชัน categorize_position กับคอลัมน์ 'Position'
data['Position Group'] = data['Position'].apply(categorize_position)

# 4. กรองข้อมูลเฉพาะนักเตะในตำแหน่ง Forward
fw_data = data[data['Position Group'] == 'Forward']
fw_features = fw_data[['Name', 'Club' ,'Position', 'Age', 'Goals', 'Shots on target']]

# 5. กำหนด model KNN
k = 1
knn_model = NearestNeighbors(n_neighbors=k)

# Train โมเดลด้วยฟีเจอร์ อายุและจำนวนประตู
knn_model.fit(fw_features[['Age', 'Goals']])

# ตัวอย่าง input ที่ต้องการค้นหา (อายุ 25 ปี, ยิง 100 ประตู)
age_fw = 25
goal_fw = 100
input_player = [[age_fw, goal_fw]]

# ค้นหานักเตะที่ใกล้เคียงที่สุด
distances, indices = knn_model.kneighbors(input_player)

# แสดงนักเตะที่ใกล้เคียงที่สุด
similar_players = fw_features.iloc[indices[0]]
print(similar_players[['Name', 'Age', 'Club', 'Goals']])

# 6. Plot ข้อมูลนักเตะและผลลัพธ์จาก KNN (2D Plot)
fig, ax = plt.subplots(figsize=(10, 8))

# Plot ข้อมูลนักเตะทั้งหมด
ax.scatter(fw_features['Age'], fw_features['Goals'], color='blue', label='All Players')

# Plot นักเตะที่ใกล้เคียงที่สุด (Nearest Neighbors)
ax.scatter(similar_players['Age'], similar_players['Goals'], color='red', label='Nearest Neighbors', s=100, marker='X')

# Plot จุด input ที่เรากำหนด
ax.scatter(age_fw, goal_fw, color='green', label='Input Player', s=200, marker='o')

# วาดเส้นจาก input ไปยังนักเตะที่ใกล้เคียง
for i in range(len(similar_players)):
    ax.plot([age_fw, similar_players['Age'].values[i]], 
            [goal_fw, similar_players['Goals'].values[i]], color='gray', linestyle='dashed')

# เพิ่มวงกลมล้อมรอบ k เพื่อนบ้าน
for i in range(1, k+1):
    circle = plt.Circle((age_fw, goal_fw), distances[0][i-1], color='black', fill=False, linestyle='dotted')
    ax.add_patch(circle)

# Labels
ax.set_title('KNN: Similar Players based on Age and Goals')
ax.set_xlabel('Age')
ax.set_ylabel('Goals')
ax.legend()

plt.show()

