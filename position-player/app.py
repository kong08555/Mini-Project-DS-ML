from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

app = Flask(__name__)

# นำเข้าข้อมูล
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

@app.route('/players/<position>', methods=['GET'])
def get_players(position):
    players = data[data['Position Group'] == position]['Name'].tolist()
    return jsonify(players)

# ฟังก์ชันสำหรับการพยากรณ์ตำแหน่งกองหลัง
def predict_defender(data):
    df_data = data[(data['Position Group'] == 'Defender') & (data['Appearances'] > 5)]
    def assign_defender_role(row):
        cb_score = (row['Tackles'] / row['Appearances']) * 0.1 + \
                    (row['Interceptions'] / row['Appearances']) * 0.1 + \
                    (row['Recoveries'] / row['Appearances']) * 0.1 + \
                    (row['Duels won'] / row['Appearances']) * 0.1 + \
                    (row['Aerial battles won'] / row['Appearances']) * 0.5
        wb_score = (row['Big Chances Created'] / row['Appearances']) * 1.0 + \
                    (row['Crosses'] / row['Appearances']) * 2.0
        return 'Centre Back' if cb_score > wb_score else 'Wing Back'

    df_data['Role'] = df_data.apply(assign_defender_role, axis=1)
    X = df_data[['Appearances', 'Tackles', 'Interceptions', 
                 'Recoveries', 'Duels won', 'Aerial battles won', 
                 'Big Chances Created', 'Crosses']]
    y = df_data['Role']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)

    df_data['Predicted Role'] = clf.predict(X)
    return df_data[['Name', 'Predicted Role']]

# ฟังก์ชันสำหรับการพยากรณ์ตำแหน่งกองกลาง
def predict_midfielder(data):
    mf_data = data[(data['Position Group'] == 'Midfielder') & (data['Appearances'] > 5)]
    def assign_midfielder_role(row):
        cam_score = (row['Shots on target'] / row['Appearances']) * 0.7 + \
                    (row['Big Chances Created'] / row['Appearances']) * 1.0 + \
                    (row['Through balls'] / row['Appearances']) * 0.6 + \
                    (row['Assists'] / row['Appearances']) * 0.5  + \
                    (row['Goals'] / row['Appearances']) * 0.7
        cdm_score = (row['Tackles'] / row['Appearances']) * 0.1 + \
                    (row['Recoveries'] / row['Appearances']) * 0.1
        return 'Attacking Midfielder' if cam_score > cdm_score else 'Defensive Midfielder'

    mf_data['Role'] = mf_data.apply(assign_midfielder_role, axis=1)
    X = mf_data[['Appearances', 'Goals', 'Shots on target',
                  'Assists', 'Big Chances Created', 'Tackles', 
                  'Recoveries']]
    y = mf_data['Role']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)

    mf_data['Predicted Role'] = clf.predict(X)
    return mf_data[['Name', 'Predicted Role']]

# ฟังก์ชันสำหรับการพยากรณ์ตำแหน่งกองหน้า
def predict_forward(data):
    fw_data = data[(data['Position Group'] == 'Forward') & (data['Appearances'] > 5)]
    def assign_forward_role(row):
        score_striker = (row['Goals'] / row['Appearances']) * 0.4 + \
                         (row['Shots on target'] / row['Appearances']) * 0.4
        score_winger = (row['Crosses'] / row['Appearances']) * 0.7 + \
                       (row['Assists'] / row['Appearances']) * 0.2
        return 'Striker' if score_striker > score_winger else 'Winger'

    fw_data['Role'] = fw_data.apply(assign_forward_role, axis=1)
    X = fw_data[['Appearances', 'Goals', 'Shots on target', 
                  'Big Chances Created', 'Crosses', 'Assists']]
    y = fw_data['Role']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)

    fw_data['Predicted Role'] = clf.predict(X)
    return fw_data[['Name', 'Predicted Role']]

@app.route('/', methods=['GET', 'POST'])
def home():
    result = None
    selected_player = None
    predicted_role = None

    if request.method == 'POST':
        position = request.form.get('position')
        player_name = request.form.get('player')
        selected_position = position

        # ค้นหาและพยากรณ์ตามตำแหน่งที่เลือก
        if position == 'Defender':
            results = predict_defender(data)
        elif position == 'Midfielder':
            results = predict_midfielder(data)
        else:
            results = predict_forward(data)

        selected_player = results[results['Name'] == player_name]
        selected_player_name = selected_player['Name'].values[0]
        predicted_role = selected_player['Predicted Role'].values[0]
        result = True  # เปลี่ยนแปลงเพื่อแสดงผลลัพธ์

    return render_template('index.html', result=result, selected_player_name=selected_player_name, 
                           predicted_role=predicted_role, selected_position=selected_position)

if __name__ == '__main__':
    app.run(debug=True)