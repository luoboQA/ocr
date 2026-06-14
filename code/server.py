import http.server
import json
from ocr import OCRNeuralNetwork
import numpy as np
from sklearn.model_selection import train_test_split

# 服务器配置
HOST_NAME = 'localhost'  # 服务器主机地址
PORT_NUMBER = 8000       # 服务器端口
HIDDEN_NODE_COUNT = 15   # 隐藏层节点数（可通过neural_network_design.py优化得到）

# 加载数据样本和标签
data_matrix = np.loadtxt(open('data.csv', 'rb'), delimiter = ',')
data_labels = np.loadtxt(open('dataLabels.csv', 'rb'))

# 转换为Python列表格式
data_matrix = data_matrix.tolist()
data_labels = data_labels.tolist()

# 分割训练集和测试集（80%训练，20%测试）
train_indices, test_indices = train_test_split(
    list(range(5000)), test_size=0.2, random_state=42
)
print(f"Training with {len(train_indices)} samples, testing with {len(test_indices)} samples")

# 创建神经网络实例（使用训练集进行初始训练）
nn = OCRNeuralNetwork(HIDDEN_NODE_COUNT, data_matrix, data_labels, train_indices)

class JSONHandler(http.server.BaseHTTPRequestHandler):
    """
    处理HTTP请求的处理器
    支持POST请求：训练网络或预测数字
    """
    
    def do_POST(self):
        """处理POST请求"""
        response_code = 200
        response = ""
        
        # 读取请求体内容
        var_len = int(self.headers.get('Content-Length'))
        content = self.rfile.read(var_len)
        payload = json.loads(content.decode('utf-8'))

        # 处理训练请求
        if payload.get('train'):
            train_array = payload['trainArray']
            # 格式化训练数据
            formatted_train = []
            for item in train_array:
                formatted_train.append({
                    'y0': item['y0'],
                    'label': item['label']
                })
            # 训练神经网络
            nn.train(formatted_train)
            nn.save()  # 保存更新后的权重
            print(f"Trained with {len(formatted_train)} samples")
            response = {"type": "train", "status": "success"}
        
        # 处理预测请求
        elif payload.get('predict'):
            try:
                result = nn.predict(payload['image'])
                response = {"type": "test", "result": result}
                print(f"Predicted: {result}")
            except Exception as e:
                print(f"Prediction error: {e}")
                response_code = 500
        
        # 无效请求
        else:
            response_code = 400

        # 发送响应
        self.send_response(response_code)
        self.send_header("Content-type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")  # 允许跨域请求
        self.end_headers()
        if response:
            self.wfile.write(json.dumps(response).encode('utf-8'))

    def do_OPTIONS(self):
        """处理OPTIONS请求（用于CORS预检）"""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

# 启动服务器
if __name__ == '__main__':
    server_class = http.server.HTTPServer
    httpd = server_class((HOST_NAME, PORT_NUMBER), JSONHandler)

    try:
        print(f"Server started at http://{HOST_NAME}:{PORT_NUMBER}")
        print("Open http://localhost:8000/ocr.html in your browser")
        httpd.serve_forever()  # 持续监听请求
    except KeyboardInterrupt:
        print("\nServer stopped.")
    finally:
        httpd.server_close()