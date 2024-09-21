import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# 1. นำเข้าข้อมูล
file_path = 'dataset/DatasetFullState.csv'
data = pd.read_csv(file_path)

# ฟังก์ชันเพื่อจัดกลุ่มตำแหน่งนักเตะ
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

# กรองข้อมูลนักเตะตำแหน่ง Forward ที่ลงสนามมากกว่า 5 นัด
fw_data = data[(data['Position Group'] == 'Forward') & (data['Appearances'] > 5)]

# กำหนดน้ำหนักให้กับฟีเจอร์ที่สำคัญ
feature_weights = {
    'Appearances': 0.5,
    'Goals': 0.6,
    'Shots on target': 0.8,
    'Big Chances Created': 0.8,
    'Crosses': 0.8
}

# เลือกฟีเจอร์ที่สำคัญ
fw_features = fw_data[['Appearances', 'Goals', 'Shots on target', 
                       'Shooting accuracy %','Big Chances Created', 'Crosses', 
                       'Assists', 'Passes','Passes per match']]

# คูณแต่ละฟีเจอร์ด้วยน้ำหนักที่กำหนด
for feature, weight in feature_weights.items():
    fw_features[feature] = fw_features[feature] * weight

# ทำการ Standardize ข้อมูล
scaler = StandardScaler()
fw_features_scaled = scaler.fit_transform(fw_features)

# ใช้ K-Means ในการจัดกลุ่ม
kmeans = KMeans(n_clusters=2, random_state=42)
fw_data['Predicted Role'] = kmeans.fit_predict(fw_features_scaled)

# แปลงระยะทางให้เป็นความน่าจะเป็นในรูปแบบเปอร์เซ็นต์
striker_distances = kmeans.transform(fw_features_scaled)[:, 0]
winger_distances = kmeans.transform(fw_features_scaled)[:, 1]

# ใช้ระยะทางในการคำนวณความน่าจะเป็น
total_distance = striker_distances + winger_distances
fw_data['Striker Probability (%)'] = (1 - (striker_distances / total_distance)) * 100
fw_data['Winger Probability (%)'] = (1 - (winger_distances / total_distance)) * 100

# แสดงข้อมูลที่มีการทำนายแล้ว
print(fw_data[['Name', 'Appearances', 'Goals', 'Shots on target', 'Big Chances Created', 
               'Predicted Role', 'Striker Probability (%)', 'Winger Probability (%)']].head(50))

# # คำนวณ Silhouette Score
# silhouette_avg = silhouette_score(fw_features_scaled, fw_data['Predicted Role'])
# print(f'Silhouette Score: {silhouette_avg}')

# # พล็อตกราฟแสดงการกระจายของนักเตะตามความน่าจะเป็นเป็น Striker และ Winger
# plt.figure(figsize=(12, 6))

# # พล็อตแสดงความน่าจะเป็นเป็น Striker
# plt.subplot(1, 2, 1)
# plt.scatter(fw_data['Striker Probability (%)'], fw_data['Winger Probability (%)'], c=fw_data['Predicted Role'], cmap='viridis', alpha=0.7)
# plt.title('Striker vs Winger Probability')
# plt.xlabel('Striker Probability (%)')
# plt.ylabel('Winger Probability (%)')

# # พล็อตแสดงการจัดกลุ่มที่ได้จาก K-Means
# plt.subplot(1, 2, 2)
# plt.scatter(fw_features_scaled[:, 0], fw_features_scaled[:, 1], c=fw_data['Predicted Role'], cmap='viridis', alpha=0.7)
# plt.title('K-Means Clustering')
# plt.xlabel('Feature 1 (Scaled)')
# plt.ylabel('Feature 2 (Scaled)')

# plt.tight_layout()
# plt.show()
