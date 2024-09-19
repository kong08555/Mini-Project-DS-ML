import pandas as pd
from sklearn.neighbors import NearestNeighbors

# โหลดข้อมูลจากไฟล์
data = pd.read_csv('dataset/EPL-players-stats-2020.csv')

# ฟังก์ชันจัดหมวดหมู่ตำแหน่งนักเตะ
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

# สร้างคอลัมน์ใหม่ 'Position Group'
data['Position Group'] = data['Position'].apply(categorize_position)

# กรองข้อมูลเฉพาะนักเตะในแต่ละตำแหน่ง
gk_data = data[data['Position Group'] == 'Goalkeeper']
df_data = data[data['Position Group'] == 'Defender']
mf_data = data[data['Position Group'] == 'Midfielder']
fw_data = data[data['Position Group'] == 'Forward']

# เลือกฟีเจอร์ที่เกี่ยวข้อง
gk_features = gk_data[['Name', 'Club', 'Position', 'Age', 'Clean sheets', 'Saves']].fillna(0)
df_features = df_data[['Name', 'Club', 'Position', 'Age', 'Tackles', 'Duels won', 'Clean sheets']].fillna(0)
mf_features = mf_data[['Name', 'Club', 'Position', 'Age', 'Assists']].fillna(0)
fw_features = fw_data[['Name', 'Club', 'Position', 'Age', 'Goals']].fillna(0)

# ฟังก์ชันสร้างโมเดล NearestNeighbors สำหรับแต่ละตำแหน่ง
def train_knn_models(k=7):
    # สร้างโมเดล KNN
    gk_knn_model = NearestNeighbors(n_neighbors=k)
    df_knn_model = NearestNeighbors(n_neighbors=k)
    mf_knn_model = NearestNeighbors(n_neighbors=k)
    fw_knn_model = NearestNeighbors(n_neighbors=k)

    # ฝึกโมเดล
    gk_knn_model.fit(gk_features[['Age', 'Clean sheets', 'Saves']])
    df_knn_model.fit(df_features[['Age', 'Tackles', 'Clean sheets']])
    mf_knn_model.fit(mf_features[['Age', 'Assists']])
    fw_knn_model.fit(fw_features[['Age', 'Goals']])

    # ส่งคืนโมเดลทั้ง 4
    return gk_knn_model, df_knn_model, mf_knn_model, fw_knn_model
