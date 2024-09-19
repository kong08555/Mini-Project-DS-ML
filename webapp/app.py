from flask import Flask, request, render_template
import pandas as pd
from sklearn.neighbors import NearestNeighbors

# โหลดและเตรียมข้อมูลเหมือนในโค้ดที่คุณมีอยู่
data = pd.read_csv('../dataset/EPL-players-stats-2020.csv')

# สร้างคอลัมน์และกรองข้อมูลเช่นเดียวกับโค้ดก่อนหน้า
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

data['Position Group'] = data['Position'].apply(categorize_position)

gk_data = data[data['Position Group'] == 'Goalkeeper']
df_data = data[data['Position Group'] == 'Defender']
mf_data = data[data['Position Group'] == 'Midfielder']
fw_data = data[data['Position Group'] == 'Forward']

gk_features = gk_data[['Name', 'Club', 'Position', 'Age', 'Clean sheets', 'Saves']].fillna(0)
df_features = df_data[['Name', 'Club', 'Position', 'Age', 'Tackles', 'Duels won', 'Clean sheets']].fillna(0)
mf_features = mf_data[['Name', 'Club', 'Position', 'Age', 'Assists', 'Passes']].fillna(0)
fw_features = fw_data[['Name', 'Club', 'Position', 'Age', 'Goals', 'Shots on target']].fillna(0)

k = 7

gk_knn_model = NearestNeighbors(n_neighbors=k).fit(gk_features[['Age', 'Clean sheets', 'Saves']])
df_knn_model = NearestNeighbors(n_neighbors=k).fit(df_features[['Age', 'Tackles', 'Clean sheets']])
mf_knn_model = NearestNeighbors(n_neighbors=k).fit(mf_features[['Age', 'Assists']])
fw_knn_model = NearestNeighbors(n_neighbors=k).fit(fw_features[['Age', 'Goals']])

app = Flask(__name__)

# การค้นหาผู้เล่นที่คล้ายกันในหน้าเดียว
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        position = request.form['position']
        age = int(request.form['age'])

        if position == 'Goalkeeper':
            clean_sheets = int(request.form['clean_sheets'])
            saves = int(request.form['saves'])
            similar_players = gk_find(age, clean_sheets, saves)
        elif position == 'Defender':
            tackles = int(request.form['tackles'])
            clean_sheets = int(request.form['clean_sheets'])
            similar_players = df_find(age, tackles, clean_sheets)
        elif position == 'Midfielder':
            assists = int(request.form['assists'])
            similar_players = mf_find(age, assists)
        elif position == 'Forward':
            goals = int(request.form['goals'])
            similar_players = fw_find(age, goals)
        else:
            similar_players = []

        return render_template('index.html', players=similar_players.to_dict(orient='records'))
    return render_template('index.html', players=[])

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

if __name__ == '__main__':
    app.run(debug=True)
