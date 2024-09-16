import pandas as pd
from sklearn.neighbors import NearestNeighbors

# predict goal: 

# 1. โหลดข้อมูลจากไฟล์
#data = pd.read_csv('dataset/player-EPL-2020-cleaning.csv', encoding='utf-8')
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

# 4. กรองข้อมูลเฉพาะนักเตะในตำแหน่ง 
gk_data = data[data['Position Group'] == 'Goalkeeper']
df_data = data[data['Position Group'] == 'Defender']
mf_data = data[data['Position Group'] == 'Midfielder']
fw_data = data[data['Position Group'] == 'Forward']


# 5. ดูผลลัพธ์ของการจัดกลุ่มเฉพาะตำแหน่ง Defender ทั้งหมด
#print(gk_data[['Name', 'Position']])
#print("--------------------------------------")
#print(df_data[['Name', 'Position']])
#print("--------------------------------------")
#print(mf_data[['Name', 'Position']])
#print("--------------------------------------")
#print(fw_data[['Name', 'Position', 'Age','Goals', 'Shots on target']])
fw_features = fw_data[['Name', 'Club' ,'Position', 'Age','Goals']]
#print(fw_features)
#print(fw_data)

# กำหนด model

# กำหนดจำนวนเพื่อนบ้าน (k) ที่ต้องการค้นหา
k = 7
knn_model = NearestNeighbors(n_neighbors=k)

# Train โมเดลด้วยฟีเจอร์อายุและจำนวนประตู
knn_model.fit(fw_features[['Age', 'Goals']])

# example
age_fw = 24
goal_fw = 50
input_player = [[age_fw, goal_fw]]

# ค้นหานักเตะที่ใกล้เคียงที่สุด
distances, indices = knn_model.kneighbors(input_player)

# แสดงนักเตะที่ใกล้เคียงที่สุด
similar_players = fw_features.iloc[indices[0]]
print(similar_players[['Name', 'Age', 'Club','Goals']])