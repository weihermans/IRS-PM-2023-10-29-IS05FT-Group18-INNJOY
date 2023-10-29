import sqlite3
import time
from flask import Flask, request, jsonify, render_template, session, redirect, url_for
from flask_cors import CORS  # 导入CORS模块
from question_to_feature import ask_question_getInput,extract_new_feature, generate_userFrame, update_userFrame, exstract_feature
import subprocess
from recommend import process_data_and_get_recommendations
from db_process import get_bnb_info
from flask_socketio import SocketIO
from map_data import get_random_list
from datetime import timedelta

app = Flask(__name__)
CORS(app, resources={r"/socket.io/*": {"origins": "http://127.0.0.1:8000"}})

socketio = SocketIO(app, cors_allowed_origins="*")  # Socket.IO 跨域设置

app.static_folder = 'static'
app.secret_key = 'Innjoy'
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=5)
userfeature={}
userfeature['ask_round']=0

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/recommended_homestays_map')
def recommended_homestays_map():
    return render_template('recommended_homestays_map.html')

@app.route('/recommends')
def recommends():
    # 检查用户的登录状态
    if not session.get('logged_in'):
        # 用户未登录，重定向到登录页面
        return redirect(url_for('login'))
    else:
        rendered_template = render_template('recommends.html')
        userfeature['try_most'] = False
        userfeature['user_f'] = generate_userFrame(session['user_id'])
        userfeature['none_feature'] = []
        userfeature['current_feature'] = None
        userfeature['user_f']['reviewer_id'] = 44250196
        #userfeature['user_f']['reviewer_id'] = session['user_id']
        userfeature['ask_round'] = 0
        print("feature:", userfeature['user_f'])
        return rendered_template


@app.route('/login', methods=['POST'])
def loginOrRegister():
    data = request.json
    action = data.get('action')
    print(action)
    if action == "login":
        emailOrUsername = data.get('emailOrUsername')
        password = data.get('password')
        print(emailOrUsername, password)
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()

        #check username
        cursor.execute('SELECT * FROM users WHERE username=? OR email=?', (emailOrUsername,emailOrUsername))
        user = cursor.fetchone()
        if user:
            db_password = user[2]
            if password == db_password:
                response = {'message': 'Login success'}
                session['logged_in'] = True
                session.permanent = False
                session['user_id'] = user[0]
                #print("Login: ",user[0])
                return jsonify(response)
            else:
                response = {'message': 'Wrong password!'}
        else:
            response = {'message': ' Unknown user name!'}
        print(response)
        conn.close()
        return jsonify(response)
    if action == "signup":
        data = request.json
        email = data.get('email')
        username = data.get('username')
        password = data.get('password')
        repassword = data.get('repassword')
        print(email, username, username, password)
        # con to db
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()

        # 获取当前已注册用户的数量
        cursor.execute('SELECT COUNT(*) FROM users')
        user_count = cursor.fetchone()[0]

        # 为新用户分配新的UserID（当前用户数量 + 1）
        user_id = user_count + 1

        # check username
        cursor.execute('SELECT username, email FROM users WHERE username=? OR email=?', (username, email))
        user = cursor.fetchone()

        if user:
            username_from_db, email_from_db = user
            print(user)
            if username_from_db == username:
                response = {'message': 'Username already exists!'}
            else:
                response = {'message': 'Email already exists!'}
        else:
            if repassword != password:
                response = {'message': 'The two passwords entered are not the same! '}
            else:
                cursor.execute('INSERT INTO users (id, email, username, password) VALUES (?, ?, ?, ?)', (user_id, email, username, password))
                conn.commit()
                session['logged_in'] = True
                session.permanent = False
                session['user_id'] = user_id
                response = {'message': 'Registration successful!'}
        print(response)
        conn.close()
        return jsonify(response)

@app.route('/chatbot', methods=['POST'])
def get_and_send():
    data = request.json
    message = data.get('message')
    print(message)
    if userfeature['try_most'] == False :
        user_feature = extract_new_feature(message,"Try_most")
        userfeature['user_f'], none_columns = update_userFrame(userfeature['user_f'], user_feature)
        print("feature:", userfeature['user_f'])
        listing_ids = process_data_and_get_recommendations(userfeature['user_f'])[:8]
        send_update_to_f(listing_ids)
        userfeature['try_most'] = True
        userfeature['none_feature'] = none_columns
        print("wait: ",none_columns)
        if len(userfeature['none_feature']):
            question = ask_question_getInput(userfeature['none_feature'][0])
            userfeature['current_feature'] = userfeature['none_feature'][0]
            response = {'message': question}
        else:
            userfeature['current_feature'] = None
            response = {'message': "Please select the most suitable one from left."}
    else:
        if userfeature['current_feature']!=None and userfeature['ask_round'] < 5:
            user_feature1 = exstract_feature(message, userfeature['current_feature'])
            user_feature2 = extract_new_feature(message, "Try_potential")
            userfeature['user_f'], none_columns = update_userFrame(userfeature['user_f'], user_feature1, user_feature2)
            print("feature:",userfeature['user_f'])
            listing_ids = process_data_and_get_recommendations(userfeature['user_f'])[:8]
            send_update_to_f(listing_ids)
            print("wait: ",none_columns)
            userfeature['none_feature'] = none_columns
            if len(userfeature['none_feature']):
                question = ask_question_getInput(userfeature['none_feature'][0])
                userfeature['current_feature'] = userfeature['none_feature'][0]
                response = {'message': question}
                userfeature['ask_round']+=1
            else:
                userfeature['current_feature'] = None
                response = {'message': "Please select the most suitable one from left."}
        else:
            response = {'message': "Please select the most suitable one from left."}
    return jsonify(response)

@app.route('/get-initial')
def get_init():
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM features WHERE reviewer_id=?', (int(userfeature['user_f']['reviewer_id'].iloc[0]),))
    user_history = cursor.fetchone()
    if user_history:
        listing_ids = process_data_and_get_recommendations(userfeature['user_f'])[:8]
        time.sleep(0.5)
        send_update_to_f(listing_ids)
    else:
        listing_df = get_random_list()
        time.sleep(1)
        listing_ids = [url.split("/")[-1] for url in listing_df["listing_url"].head(8)]
        send_update_to_f(listing_ids)
    conn.close()
    return jsonify({'message': 'Initial data sent successfully!'})

@app.route('/run-python')
def run_python_script():
    script_name = request.args.get('script', '')
    if script_name:
        try:
            # 使用subprocess来执行Python脚本
            result = subprocess.run(["python", script_name], capture_output=True, text=True, shell=True)
            if result.returncode == 0:
                # 执行成功
                return "Python脚本执行成功"
            else:
                # 执行失败
                return "Python脚本执行失败：" + result.stderr
        except Exception as e:
            return "执行错误：" + str(e)
    else:
        return "没有指定要执行的Python脚本"

def send_update_to_f(listing_ids):
    room_data = []
    for listing_id in listing_ids:
        room_info = get_bnb_info(listing_id)
        if room_info:
            room_data.append(room_info)
            print("sent to F:",room_info)
    socketio.emit('update',{"room_data": room_data})

if __name__ == '__main__':
    print(1)
    socketio.run(app, host='localhost', port=8000, debug=True, allow_unsafe_werkzeug=True)
