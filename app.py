from flask import Flask, request, render_template
import pandas as pd
import pickle

app = Flask(__name__)

# Memuat model K-Means dan Nearest Neighbors yang telah dilatih
with open('kmeans_model.pkl', 'rb') as f:
    kmeans = pickle.load(f)

with open('nearest_neighbors_model.pkl', 'rb') as f:
    nbrs = pickle.load(f)

# Membaca data warna
colors_df = pd.read_csv('colors.csv')

def get_color_recommendations(input_rgb, n_recommendations=5):
    distances, indices = nbrs.kneighbors([input_rgb])
    recommended_colors = colors_df.iloc[indices[0]]
    return recommended_colors

def get_rgb_from_color_name(color_name):
    color_row = colors_df[colors_df['color_name'].str.lower() == color_name.lower()]
    if not color_row.empty:
        return color_row[['r', 'g', 'b']].values[0]
    else:
        return None

@app.route('/')
def index():
    color_names = colors_df['color_name'].tolist()
    return render_template('index.html', color_names=color_names)

@app.route('/recommend', methods=['POST'])
def recommend():
    color_name = request.form['color_name']
    input_rgb = get_rgb_from_color_name(color_name)
    if input_rgb is not None:
        recommended_colors = get_color_recommendations(input_rgb)
        return render_template('recommend.html', color_name=color_name, recommended_colors=recommended_colors.to_dict(orient='records'))
    else:
        return render_template('error.html', message="Color not found. Please try again.")

if __name__ == '__main__':
    app.run(debug=True)
