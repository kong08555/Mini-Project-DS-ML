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

# กรองข้อมูลนักเตะตำแหน่ง Defender ที่ลงสนามมากกว่า 5 นัด
df_data = data[(data['Position Group'] == 'Defender') & (data['Appearances'] > 5)]

# กำหนดฟีเจอร์ที่สำคัญ
df_features = df_data[['Appearances', 'Tackles', 'Blocked shots', 'Interceptions', 'Clearances', 
                        'Recoveries', 'Duels won', 'Aerial battles won', 'Big Chances Created', 
                        'Crosses']]

# สร้าง target label ตามเงื่อนไขที่กำหนด โดยใช้คะแนน
def assign_defender_role(row):
    # คำนวณคะแนนสำหรับ Center back (CB)
    cb_score =  (row['Tackles'] / row['Appearances']) * 0.1 + \
                (row['Interceptions'] / row['Appearances']) * 0.1 + \
                (row['Recoveries'] / row['Appearances']) * 0.1 + \
                (row['Duels won'] / row['Appearances']) * 0.1 + \
                (row['Aerial battles won'] / row['Appearances']) * 0.5
    
    # คำนวณคะแนนสำหรับ Full back (FB)
    wb_score = (row['Big Chances Created'] / row['Appearances']) * 1.0 + \
                (row['Crosses'] / row['Appearances']) * 2.0
    
    # กำหนดตำแหน่งตามคะแนน
    if cb_score > wb_score:
        return 'Center back'
    elif wb_score > cb_score:
        return 'Wing back'
    else:
        return 'Unknown'

df_data['Role'] = df_data.apply(assign_defender_role, axis=1)

# กรองเฉพาะข้อมูลที่ไม่ใช่ 'Unknown'
df_data = df_data[df_data['Role'] != 'Unknown']

# เตรียมข้อมูลสำหรับการฝึก (train/test split)
X = df_data[['Appearances', 'Tackles', 'Interceptions', 
              'Recoveries', 'Duels won', 'Aerial battles won', 
             'Big Chances Created', 'Crosses']]
y = df_data['Role']

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
df_data['Predicted Role'] = clf.predict(X)
print(df_data[['Name', 'Appearances',  'Predicted Role']].head(60))

print("---------------Accuracy------------------")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("-----------------------------------------")