/**
 * OCR前端模块
 * 
 * 功能：
 * 1. 创建200x200像素的画布供用户绘制数字
 * 2. 将绘制的数字转换为神经网络可处理的格式
 * 3. 与后端服务器通信进行训练或预测
 * 
 * 简化计算：200x200px画布被转换为20x20px的网格
 * 每个20x20网格对应一个10x10px的画布区域
 * 最终形成400个像素的输入数组（1表示白色，0表示黑色）
 */
var ocrDemo = {
    // 画布配置
    CANVAS_WIDTH: 200,
    TRANSLATED_WIDTH: 20,    // 转换后的网格尺寸（20x20）
    PIXEL_WIDTH: 10,         // 每个网格像素的实际尺寸（200/20=10）
    BATCH_SIZE: 1,           // 批量训练时的批次大小

    // 服务器配置
    PORT: "8000",
    HOST: "http://localhost",

    // 颜色定义
    BLACK: "#000000",
    BLUE: "#0000ff",

    // 训练相关变量
    trainArray: [],          // 存储待训练的数据
    trainingRequestCount: 0, // 已收集的训练样本数

    /**
     * 页面加载时的初始化函数
     */
    onLoadFunction: function() {
        this.resetCanvas();
    },

    /**
     * 重置画布（清空所有绘制内容）
     */
    resetCanvas: function() {
        var canvas = document.getElementById('canvas');
        var ctx = canvas.getContext('2d');

        this.data = [];  // 存储20x20网格数据（0=黑，1=白）
        // 用黑色填充整个画布
        ctx.fillStyle = this.BLACK;
        ctx.fillRect(0, 0, this.CANVAS_WIDTH, this.CANVAS_WIDTH);
        // 初始化数据数组（400个0）
        var matrixSize = 400;
        while (matrixSize--) this.data.push(0);
        // 绘制蓝色网格线
        this.drawGrid(ctx);

        // 绑定鼠标事件
        canvas.onmousemove = function(e) { this.onMouseMove(e, ctx, canvas) }.bind(this);
        canvas.onmousedown = function(e) { this.onMouseDown(e, ctx, canvas) }.bind(this);
        canvas.onmouseup = function(e) { this.onMouseUp(e, ctx) }.bind(this);
    },

    /**
     * 绘制网格线
     */
    drawGrid: function(ctx) {
        for (var x = this.PIXEL_WIDTH, y = this.PIXEL_WIDTH; 
             x < this.CANVAS_WIDTH; 
             x += this.PIXEL_WIDTH, y += this.PIXEL_WIDTH) {
            ctx.strokeStyle = this.BLUE;
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, this.CANVAS_WIDTH);
            ctx.stroke();

            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(this.CANVAS_WIDTH, y);
            ctx.stroke();
        }
    },

    /**
     * 鼠标移动事件处理（绘制轨迹）
     */
    onMouseMove: function(e, ctx, canvas) {
        if (!canvas.isDrawing) {
            return;
        }
        this.fillSquare(ctx, e.clientX - canvas.offsetLeft, e.clientY - canvas.offsetTop);
    },

    /**
     * 鼠标按下事件处理（开始绘制）
     */
    onMouseDown: function(e, ctx, canvas) {
        canvas.isDrawing = true;
        this.fillSquare(ctx, e.clientX - canvas.offsetLeft, e.clientY - canvas.offsetTop);
    },

    /**
     * 鼠标松开事件处理（停止绘制）
     */
    onMouseUp: function(e) {
        canvas.isDrawing = false;
    },

    /**
     * 填充一个网格方块
     * 将画布坐标转换为网格坐标，并更新数据数组
     */
    fillSquare: function(ctx, x, y) {
        var xPixel = Math.floor(x / this.PIXEL_WIDTH);
        var yPixel = Math.floor(y / this.PIXEL_WIDTH);
        // 更新数据数组（将对应位置设为1表示白色）
        this.data[((xPixel - 1) * this.TRANSLATED_WIDTH + yPixel) - 1] = 1;

        // 在画布上绘制白色方块
        ctx.fillStyle = '#ffffff';
        ctx.fillRect(xPixel * this.PIXEL_WIDTH, yPixel * this.PIXEL_WIDTH, 
                     this.PIXEL_WIDTH, this.PIXEL_WIDTH);
    },

    /**
     * 训练神经网络
     * 收集用户绘制的数字和输入的数字标签，批量发送给服务器
     */
    train: function() {
        var digitVal = document.getElementById("digit").value;
        if (!digitVal || this.data.indexOf(1) < 0) {
            alert("请先绘制数字并输入数字值后再训练网络");
            return;
        }
        // 添加到训练队列
        this.trainArray.push({"y0": this.data, "label": parseInt(digitVal)});
        this.trainingRequestCount++;

        // 达到批次大小时发送训练数据
        if (this.trainingRequestCount == this.BATCH_SIZE) {
            alert("正在发送训练数据到服务器...");
            var json = {
                trainArray: this.trainArray,
                train: true
            };
            this.sendData(json);
            this.trainingRequestCount = 0;
            this.trainArray = [];  // 清空队列
        }
    },

    /**
     * 测试神经网络
     * 发送当前绘制的数字给服务器进行预测
     */
    test: function() {
        if (this.data.indexOf(1) < 0) {
            alert("请先绘制一个数字后再测试网络");
            return;
        }
        var json = {
            image: this.data,
            predict: true
        };
        this.sendData(json);
    },

    /**
     * 处理服务器响应
     */
    receiveResponse: function(xmlHttp) {
        if (xmlHttp.status != 200) {
            alert("服务器返回状态码 " + xmlHttp.status);
            return;
        }
        var responseJSON = JSON.parse(xmlHttp.responseText);
        if (xmlHttp.responseText && responseJSON.type == "test") {
            alert("神经网络预测您绘制的是 '" + responseJSON.result + "'");
        }
    },

    /**
     * 请求错误处理
     */
    onError: function(e) {
        alert("连接服务器时出错: " + e.target.statusText);
    },

    /**
     * 发送数据到服务器
     */
    sendData: function(json) {
        var xmlHttp = new XMLHttpRequest();
        xmlHttp.open('POST', this.HOST + ":" + this.PORT, false);
        xmlHttp.onload = function() { this.receiveResponse(xmlHttp); }.bind(this);
        xmlHttp.onerror = function() { this.onError(xmlHttp) }.bind(this);
        var msg = JSON.stringify(json);
        xmlHttp.setRequestHeader('Content-length', msg.length);
        xmlHttp.setRequestHeader("Connection", "close");
        xmlHttp.send(msg);
    }
}