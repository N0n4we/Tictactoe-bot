from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret'
socketio = SocketIO(app)

player_mv = None

@app.route('/')
def index():
    # 棋盘
    return render_template('index.html')

@app.route('/player_make_move', methods=['GET', 'POST'])
def player_make_move():
    # 获取前端发送的动作数据
    data = request.get_json()
    cell_index = data.get('index')

    global player_mv
    player_mv = cell_index

    print(f"Player make move: {data}")
    return jsonify({'index': cell_index})

@app.route('/server_make_move', methods=['GET', 'POST'])
def server_make_move():
    # 通过socketweb向前端发送数据
    if request.method == 'POST':
        socketio.emit('ai_make_move', request.form['cell_index'])
        return f"已发出数据: {request.form['cell_index']}"
    return "不支持的方法!"

@app.route('/get_player_move', methods=['GET'])
def get_player_move():
    # 返回玩家的下子
    global player_mv
    rsp = player_mv
    player_mv = None
    return jsonify({'cell_index': rsp})
    

