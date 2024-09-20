import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. นำเข้าข้อมูล
file_path = '../dataset/DataStateClean.csv'
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

# ฟังก์ชันสำหรับการพยากรณ์ตำแหน่งกองหลัง
def predict_defender(data):
    df_data = data[(data['Position Group'] == 'Defender') & (data['Appearances'] > 5)]
    df_features = df_data[['Appearances', 'Tackles', 'Interceptions', 
                            'Recoveries', 'Duels won', 'Aerial battles won', 
                            'Big Chances Created', 'Crosses']]

    def assign_defender_role(row):
        cb_score = (row['Tackles'] / row['Appearances']) * 0.1 + \
                    (row['Interceptions'] / row['Appearances']) * 0.1 + \
                    (row['Recoveries'] / row['Appearances']) * 0.1 + \
                    (row['Duels won'] / row['Appearances']) * 0.1 + \
                    (row['Aerial battles won'] / row['Appearances']) * 0.5

        wb_score = (row['Big Chances Created'] / row['Appearances']) * 1.0 + \
                    (row['Crosses'] / row['Appearances']) * 2.0
        
        return 'Center back' if cb_score > wb_score else 'Wing back'

    df_data['Role'] = df_data.apply(assign_defender_role, axis=1)

    X = df_data[['Appearances', 'Tackles', 'Interceptions', 
                 'Recoveries', 'Duels won', 'Aerial battles won', 
                 'Big Chances Created', 'Crosses']]
    y = df_data['Role']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("Defender - Classification Report:\n", classification_report(y_test, y_pred))
    print("Defender - Accuracy:", accuracy_score(y_test, y_pred))
    df_data['Predicted Role'] = clf.predict(X)
    return df_data[['Name', 'Appearances', 'Predicted Role']]

# ฟังก์ชันสำหรับการพยากรณ์ตำแหน่งกองกลาง
def predict_midfielder(data):
    mf_data = data[(data['Position Group'] == 'Midfielder') & (data['Appearances'] > 5)]
    mf_data['Tackle success %'] = mf_data['Tackle success %'].str.rstrip('%').astype('float') / 100

    def assign_midfielder_role(row):
        cam_score = (row['Shots on target'] / row['Appearances']) * 0.7 + \
                    (row['Big Chances Created'] / row['Appearances']) * 1.0 + \
                    (row['Through balls'] / row['Appearances']) * 0.6 + \
                    (row['Assists'] / row['Appearances']) * 0.5  + \
                    (row['Goals'] / row['Appearances']) * 0.7
        
        cdm_score = (row['Tackles'] / row['Appearances']) * 0.1 + \
                    (row['Recoveries'] / row['Appearances']) * 0.1
        
        return 'CAM' if cam_score > cdm_score else 'CDM'

    mf_data['Role'] = mf_data.apply(assign_midfielder_role, axis=1)

    X = mf_data[['Appearances', 'Goals', 'Shots on target',
                  'Assists', 'Big Chances Created', 'Tackles', 
                  'Recoveries']]
    y = mf_data['Role']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("Midfielder - Classification Report:\n", classification_report(y_test, y_pred))
    print("Midfielder - Accuracy:", accuracy_score(y_test, y_pred))
    mf_data['Predicted Role'] = clf.predict(X)
    return mf_data[['Name', 'Appearances', 'Goals', 'Shots on target', 
                     'Assists', 'Predicted Role']]

# ฟังก์ชันสำหรับการพยากรณ์ตำแหน่งกองหน้า
def predict_forward(data):
    fw_data = data[(data['Position Group'] == 'Forward') & (data['Appearances'] > 5)]
    fw_data['Shooting accuracy %'] = fw_data['Shooting accuracy %'].str.rstrip('%').astype('float') / 100

    def assign_forward_role(row):
        score_striker = (row['Goals'] / row['Appearances']) * 0.4 + \
                         (row['Shots on target'] / row['Appearances']) * 0.4 + \
                         (row['Shooting accuracy %']) * 0.2
        score_winger = (row['Crosses'] / row['Appearances']) * 0.7 + \
                       (row['Assists'] / row['Appearances']) * 0.2 + \
                       (row['Big Chances Created'] / row['Appearances']) * 0.1
        
        return 'Striker' if score_striker > score_winger else 'Winger'

    fw_data['Role'] = fw_data.apply(assign_forward_role, axis=1)

    X = fw_data[['Appearances', 'Goals', 'Shots on target', 
                  'Shooting accuracy %', 'Big Chances Created', 'Crosses', 
                  'Assists', 'Passes', 'Passes per match']]
    y = fw_data['Role']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("Forward - Classification Report:\n", classification_report(y_test, y_pred))
    print("Forward - Accuracy:", accuracy_score(y_test, y_pred))
    fw_data['Predicted Role'] = clf.predict(X)
    return fw_data[['Name', 'Appearances', 'Goals', 'Shots on target', 
                     'Shooting accuracy %', 'Predicted Role']]

# ฟังก์ชันหลักสำหรับการรันโปรแกรม
def main():
    print("Select Position:")
    positions = ['Defender', 'Midfielder', 'Forward']
    for idx, pos in enumerate(positions):
        print(f"{idx + 1}. {pos}")

    position_choice = int(input("Choose a position (1-3): ")) - 1
    if position_choice < 0 or position_choice >= len(positions):
        print("Invalid position!")
        return

    position = positions[position_choice]
    if position == 'Defender':
        results = predict_defender(data)
    elif position == 'Midfielder':
        results = predict_midfielder(data)
    else:
        results = predict_forward(data)

    print(f"\nPlayers in position {position}:")
    for idx, player in enumerate(results['Name']):
        print(f"{idx + 1}. {player}")

    player_choice = int(input(f"\nSelect a player (1-{len(results)}): ")) - 1
    if player_choice < 0 or player_choice >= len(results):
        print("Invalid player selection!")
        return

    selected_player = results.iloc[player_choice]
    print(f"\nSelected Player: {selected_player['Name']}")
    print(f"Predicted Role: {selected_player['Predicted Role']}")

if __name__ == "__main__":
    main()
