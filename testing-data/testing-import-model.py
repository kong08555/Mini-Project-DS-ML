from model.KNN import train_knn_models
import pandas as pd

# โหลดข้อมูลอีกครั้งเพื่อแสดงผลลัพธ์
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

# เรียกโมเดลที่ผ่านการฝึกแล้วจาก knn_models.py
gk_knn_model, df_knn_model, mf_knn_model, fw_knn_model = train_knn_models()

# ฟังก์ชันค้นหานักเตะที่ใกล้เคียงตามอายุและสถิติ
def gk_find(age, clean_sheets, saves, k=7):
    gk_input_player = [[age, clean_sheets, saves]]
    distances, indices = gk_knn_model.kneighbors(gk_input_player)
    gk_similar_players = gk_features.iloc[indices[0]]
    return gk_similar_players[['Name', 'Age', 'Club', 'Saves', 'Clean sheets']]

def df_find(age, tackles, clean_sheets, k=7):
    df_input_player = [[age, tackles, clean_sheets]]
    distances, indices = df_knn_model.kneighbors(df_input_player)
    df_similar_players = df_features.iloc[indices[0]]
    return df_similar_players[['Name', 'Age', 'Club', 'Tackles']]

def mf_find(age, assists, k=7):
    mf_input_player = [[age, assists]]
    distances, indices = mf_knn_model.kneighbors(mf_input_player)
    mf_similar_players = mf_features.iloc[indices[0]]
    return mf_similar_players[['Name', 'Age', 'Club', 'Assists']]

def fw_find(age, goals, k=7):
    fw_input_player = [[age, goals]]
    distances, indices = fw_knn_model.kneighbors(fw_input_player)
    fw_similar_players = fw_features.iloc[indices[0]]
    return fw_similar_players[['Name', 'Age', 'Club', 'Goals']]

#-----------------example player-------------------------------

# GK input
age_gk = 27
clean_sheet_gk = 15
saves = 70
similar_goalkeeper = gk_find(age_gk, clean_sheet_gk, saves)

# DF input
age_df = 28
tackle_df = 150
clean_sheet_df = 15
similar_defender = df_find(age_df, tackle_df, clean_sheet_df)

# MF input
age_mf = 28
assists_mf = 30
similar_midfielder = mf_find(age_mf, assists_mf)

# FW input
age_fw = 24
goal_fw = 50
similar_forwards = fw_find(age_fw, goal_fw)

# แสดงผลลัพธ์
print("---------------------------------------------------------")
print(similar_goalkeeper)
print("---------------------------------------------------------")
print(similar_defender)
print("---------------------------------------------------------")
print(similar_midfielder)
print("---------------------------------------------------------")
print(similar_forwards)
print("---------------------------------------------------------")
