import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. นำเข้าข้อมูล
file_path = 'dataset/DataStateClean.csv'
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

# แปลงค่า 'Shooting accuracy %' จาก string เป็น float
fw_data['Shooting accuracy %'] = fw_data['Shooting accuracy %'].str.rstrip('%').astype('float') / 100

# กำหนดฟีเจอร์ที่สำคัญ
fw_features = fw_data[['Appearances', 'Goals', 'Shots on target', 
                       'Shooting accuracy %', 'Big Chances Created', 'Crosses', 
                       'Assists', 'Passes', 'Passes per match']]

# สร้าง target label ตามเงื่อนไขที่คุณกำหนด
def assign_forward_role(row):
    score_striker = (row['Goals'] / row['Appearances']) * 0.4 + (row['Shots on target'] / row['Appearances']) * 0.4 + (row['Shooting accuracy %']) * 0.2
    score_winger = (row['Crosses'] / row['Appearances']) * 0.7 + (row['Assists'] / row['Appearances']) * 0.2 + (row['Big Chances Created'] / row['Appearances']) * 0.1
    
    if score_striker > score_winger:
        return 'Striker', score_striker, score_winger
    else:
        return 'Winger', score_striker, score_winger

# สร้างคอลัมน์สำหรับ Role และคะแนน
fw_data[['Role', 'Striker Score', 'Winger Score']] = fw_features.apply(assign_forward_role, axis=1, result_type='expand')

# คำนวณเปอร์เซ็นต์
total_score = fw_data['Striker Score'] + fw_data['Winger Score']
fw_data['Striker Probability (%)'] = (fw_data['Striker Score'] / total_score) * 100
fw_data['Winger Probability (%)'] = (fw_data['Winger Score'] / total_score) * 100

# กรองเฉพาะข้อมูลที่ไม่ใช่ 'Unknown'
fw_data = fw_data[fw_data['Role'] != 'Unknown']

# เตรียมข้อมูลสำหรับการฝึก (train/test split)
X = fw_data[['Appearances', 'Goals', 'Shots on target', 
             'Shooting accuracy %', 'Big Chances Created', 'Crosses', 
             'Assists', 'Passes', 'Passes per match']]
y = fw_data['Role']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ใช้ RandomForest ในการทำนาย
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# ทำนายผล
y_pred = clf.predict(X_test)

# แสดงผลลัพธ์
#print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# แสดงข้อมูลที่ทำนายแล้วพร้อมเปอร์เซ็นต์
fw_data['Predicted Role'] = clf.predict(X)
print(fw_data[['Name', 'Appearances', 'Goals', 'Shots on target', 
               'Shooting accuracy %', 'Predicted Role', 
               ]].head(60))

print("---------------Accuracy------------------")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("-----------------------------------------")
#print(fw_data[['Name', 'Appearances',  
#               'Big Chances Created', 'Crosses', 
 #              'Assists', 'Passes', 'Passes per match']].head(50))

#print(fw_data[['Name', 'Appearances','Striker Probability (%)', 'Winger Probability (%)']].head(50))
