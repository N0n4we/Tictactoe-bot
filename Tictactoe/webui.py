from app import app
import requests
import json
import _thread
import time

"""

WebUi类提供两个接口:
    get_player_move() -> list or None
    make_move(cell: list[2]) -> PostResponds

"""


class WebUi:
    def __init__(self, host="127.0.0.1", port=8080):
        self.host, self.port = host, port
        self.url = f"http://{host}:{port}"
        self.app = app

    def run(self, debug=False):
        print("服务器, 启动!")
        try:
            _thread.start_new_thread( self.server_thread, (debug, self.host, self.port))
        except:
            print ("Error: 无法启动服务器线程")

    def server_thread(self, debug, host, port):
        app.run(debug=debug, host=host, port=port)

    def get_player_move(self) -> list or None:
        """获取玩家的下子位置坐标, 只返回一次, 其他返回都为None"""
        response = requests.get(self.url + '/get_player_move')
        cell_index = json.loads(response.text)["cell_index"]
        if cell_index is not None:
            cell_index = int(cell_index)
            cell = [cell_index//3, cell_index%3]
            return cell
        else:
            return None

    def make_move(self, cell: list[2]):
        """发出下子的指令, cell为下子位置的坐标"""
        print(f"送出：{cell}")
        cell_index = 3*cell[0] + cell[1]
        return requests.post(self.url + '/server_make_move', data={'cell_index': cell_index})
    
    def test(self):
        print("Running WebUi.test")
        while True:
            cell = self.get_player_move()
            if cell is not None:
                print(cell)
                self.make_move([0,0])
            time.sleep(1)


if __name__ == '__main__':
    webui = WebUi()
    webui.run()
    # webui.test()
