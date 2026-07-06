# Gesture Recognition

## 使用指南
1. 复制`dataset.tar.gz`复制到本目录
2. 将压缩包解压到`rawdata`文件夹
    ```sh
    tar -xzvf dataset.tar.gz -C rawdata
    ```
3. 运行`preprocess.py` (使用`python preprocess.py --help`查看具体用法)
4. 运行`logistics/lstm_cnn/train_3dcnn.py`以训练模型。