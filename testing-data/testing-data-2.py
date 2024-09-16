import pandas as pd
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
        return 'Unknown'  # ถ้ามีตำแหน่งที่ไม่อยู่ใน 4 ตำแหน่งหลัก

# 3. ใช้ฟังก์ชัน categorize_position กับคอลัมน์ 'Position'
data['Position Group'] = data['Position'].apply(categorize_position)

# 4. กรองข้อมูลเฉพาะนักเตะในตำแหน่ง Forward
fw_data = data[data['Position Group'] == 'Forward']
fw_features = fw_data[['Name', 'Club', 'Position', 'Age', 'Goals', 'Shots on target']]

# 5. กำหนดจำนวนเพื่อนบ้าน (k) ที่ต้องการค้นหา
k = 10
knn_model = NearestNeighbors(n_neighbors=k)

# Train โมเดลด้วยฟีเจอร์อายุและจำนวนประตู
knn_model.fit(fw_features[['Age', 'Goals']])

# ฟังก์ชันสำหรับการค้นหานักเตะที่ใกล้เคียงตามอายุและจำนวนประตู


#-----------------finding player-------------------------------
def forward_find(age, goal, k=10):
    # รับค่า age และ goal ที่ผู้ใช้ระบุ
    input_player = [[age, goal]]
    
    # ค้นหานักเตะที่ใกล้เคียงที่สุด
    distances, indices = knn_model.kneighbors(input_player)
    
    # แสดงนักเตะที่ใกล้เคียงที่สุด
    similar_players = fw_features.iloc[indices[0]]
    return similar_players[['Name', 'Age', 'Club', 'Goals']]


age_fw = 20
goal_fw = 20
similar_forwards = forward_find(age_fw, goal_fw, k=10)

# แสดงผลลัพธ์
print(similar_forwards)
