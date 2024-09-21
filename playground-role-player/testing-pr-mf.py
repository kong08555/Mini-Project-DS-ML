import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. นำเข้าข้อมูล
file_path = 'dataset/DataStateClean.csv'
data = pd.read_csv(file_path)


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

# กรองข้อมูลนักเตะตำแหน่ง Midfielder ที่ลงสนามมากกว่า 5 นัด
mf_data = data[(data['Position Group'] == 'Midfielder') & (data['Appearances'] > 5)]

# แปลงค่า 'Tackle success %' จาก string เป็น float
mf_data['Tackle success %'] = mf_data['Tackle success %'].str.rstrip('%').astype('float') / 100

# กำหนดฟีเจอร์ที่สำคัญ
#mf_features = mf_data[['Appearances', 'Goals', 'Shots on target',
#                        'Assists', 'Big Chances Created', 'Through balls', # CAM features
 #                       'Tackles', 'Recoveries']]  # CDM features

# สร้าง target label ตามเงื่อนไขที่กำหนด โดยใช้คะแนน
def assign_midfielder_role(row):
    # คำนวณคะแนนสำหรับ CAM
    cam_score = (row['Shots on target'] / row['Appearances']) * 0.7 + \
                (row['Big Chances Created'] / row['Appearances']) * 1.0 + \
                (row['Through balls'] / row['Appearances']) * 0.6 + \
                (row['Assists'] / row['Appearances']) * 0.5  + \
                (row ['Goals'] / row['Appearances']) * 0.7
    
    # คำนวณคะแนนสำหรับ CDM
    cdm_score = (row['Tackles'] / row['Appearances']) * 0.1 + \
                (row['Recoveries'] / row['Appearances']) * 0.1
    
    # กำหนดตำแหน่งตามคะแนน
    if cam_score > cdm_score:
        return 'CAM'
    elif cdm_score > cam_score:
        return 'CDM'
    else:
        return 'Unknown'

mf_data['Role'] = mf_data.apply(assign_midfielder_role, axis=1)

# กรองเฉพาะข้อมูลที่ไม่ใช่ 'Unknown'
mf_data = mf_data[mf_data['Role'] != 'Unknown']

# เตรียมข้อมูลสำหรับการฝึก (train/test split)
X = mf_data[['Appearances', 'Goals', 'Shots on target',
              'Assists', 'Big Chances Created', 'Tackles', 
              'Recoveries']]
y = mf_data['Role']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ใช้ RandomForest ในการทำนาย
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# ทำนายผล
y_pred = clf.predict(X_test)

# แสดงผลลัพธ์
#print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# แสดงข้อมูลที่ทำนายแล้ว
mf_data['Predicted Role'] = clf.predict(X)
print(mf_data[['Name', 'Appearances', 'Goals', 'Shots on target', 
                'Assists', 'Predicted Role']].head(60))

print("---------------Accuracy------------------")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("-----------------------------------------")

