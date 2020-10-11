from flask import Flask
from flask import jsonify
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import os

app = Flask(__name__)

# home page
@app.route('/')
def home():
    return 'Beep boop.'

# predictions
@app.route('/v1/predictions/<team1>/<team2>/<patch_for_model>')
def predictions(team1=None, team2=None, patch_for_model=None):

    # inputs for teams and current patch
    firstteam = team1
    secondteam = team2
    selected_patch = patch_for_model
    selected_patch = float(selected_patch)
    # this will be troublesome
    previous_patch = selected_patch - .01
    patches = [selected_patch, previous_patch]

    # reading the data from oracle's elixir, filtering down to just teams and operative columns
    df = pd.read_csv('data.csv')
    df = df[['position', 'team', 'patch', 'gamelength', 'towers', 'opp_towers', 'result']]
    df = df.loc[df['position'] == 'team']
    trash = df.pop('position')

    # average towers per minute for teams in the past two patches
    team_gain_lost_time = {}
    def team_total_towers(row):
        team, patch, time, gain, lost, result = row
        if patch in patches:
            if team in team_gain_lost_time:
                x, y, z = team_gain_lost_time.get(team)
                x += gain
                y += lost
                z += time
                team_gain_lost_time[team] = (x, y, z)
            else:
                team_gain_lost_time[team] = (gain, lost, time)
        else:
            pass
    df.apply(team_total_towers, axis=1)
    team_total = pd.DataFrame.from_dict(team_gain_lost_time, orient='index')
    # average towers taken per minute for teams 
    team_total[3] = team_total[0] / team_total[2]
    # average towers lost per minute for teams
    team_total[4] = team_total[1] / team_total[2]

    # average towers per minute for each individual game
    df['avg_gain'] = df['towers'] / df['gamelength']
    df['avg_loss'] = df['opp_towers'] / df['gamelength']

    # machine learning
    row, col = df.shape
    df = df[df.columns[-3:]]
    train_file = df.head(int(row * .8))
    test_file = df.tail(int(row * .2))
    train_result = train_file.pop('result')
    test_result = test_file.pop('result')
    tf.random.set_seed(10)
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    checkpoint_path = 'checkpoint/cp.ckpt'
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        verbose=1
    )
    model.load_weights(checkpoint_path)
    test_loss, test_acc = model.evaluate(test_file, test_result)
    team1 = team_total.loc[firstteam]
    prediction1 = model.predict([[team1[3], team1[4]]])
    prediction1 = round(float(prediction1) * 100, 2)
    team2 = team_total.loc[secondteam]
    prediction2 = model.predict([[team2[3], team2[4]]])
    prediction2 = round(float(prediction2) * 100, 2)
    difference = round((prediction1 - prediction2), 2)
    percent1 = round((prediction1 / (prediction1 + prediction2)) * 100, 2)
    percent2 = round((prediction2 / (prediction1 + prediction2)) * 100, 2)

    final = {firstteam: percent1, secondteam: percent2}
    
    return jsonify(final)

if __name__ == '__main__':
    app.run()