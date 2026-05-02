import os
import sqlite3
from flask import Flask,  request, redirect, url_for, session, flash, render_template
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from utils.preprocess import preprocess_image
from utils.train import train_model

app = Flask(__name__)
app.secret_key = 'super_secret_key'
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['MODEL_PATH'] = 'model/image_detector.h5'
app.config['GRAPHS_DIR'] = 'static/graphs/'
app.config['DATASET_TRAIN'] = 'Dataset/train/'
app.config['DATASET_TEST'] = 'Dataset/test/'

training_metrics = {}

def init_db():
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            username TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL,
            mobile TEXT NOT NULL,
            role TEXT DEFAULT 'user'
        )
    ''')
    conn.commit()
    conn.close()

init_db()

@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        full_name = request.form['full_name']
        email = request.form['email']
        username = request.form['username']
        password = request.form['password']
        mobile = request.form['mobile']
        
        hashed_password = generate_password_hash(password)
        
        try:
            conn = sqlite3.connect('database.db')
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO users (full_name, email, username, password, mobile)
                VALUES (?, ?, ?, ?, ?)
            ''', (full_name, email, username, hashed_password, mobile))
            conn.commit()
            conn.close()
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username or Email already exists.', 'danger')
            
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
        user = cursor.fetchone()
        conn.close()
        
        if user and check_password_hash(user[4], password):
            session['user_id'] = user[0]
            session['username'] = user[3]
            session['full_name'] = user[1]
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password.', 'danger')
            
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('dashboard.html')

@app.route('/train', methods=['GET', 'POST'])
def train():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    global training_metrics
    if request.method == 'POST':
        metrics, error = train_model(
            app.config['DATASET_TRAIN'],
            app.config['DATASET_TEST'],
            app.config['MODEL_PATH'],
            app.config['GRAPHS_DIR']
        )
        if error:
            flash(error, 'danger')
        else:
            training_metrics = metrics
            flash('Training completed successfully!', 'success')
            return redirect(url_for('results'))
            
    return render_template('train.html')

@app.route('/results')
def results():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    if not training_metrics:
        flash('No training metrics available. Please train the model first.', 'warning')
        return redirect(url_for('train'))
    return render_template('results.html', metrics=training_metrics)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    prediction = None
    confidence = None
    image_url = None
    
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part', 'danger')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No selected file', 'danger')
            return redirect(request.url)
            
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            if not os.path.exists(app.config['MODEL_PATH']):
                flash('Model not found. Please train the model first.', 'danger')
                return redirect(url_for('train'))
            
            model = load_model(app.config['MODEL_PATH'])
            processed_img = preprocess_image(filepath)
            if processed_img is not None:
                processed_img = np.expand_dims(processed_img, axis=0)
                score = model.predict(processed_img)[0][0]
                
                if score > 0.5:
                    prediction = "REAL"
                    confidence = round(score * 100, 2)
                else:
                    prediction = "FAKE"
                    confidence = round((1 - score) * 100, 2)
                    
                image_url = url_for('static', filename='uploads/' + filename)
            else:
                flash('Error processing image.', 'danger')

    return render_template('predict.html', prediction=prediction, confidence=confidence, image_url=image_url)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
