import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# 讀取訓練和測試數據
train_data = pd.read_csv('training.csv')
test_data = pd.read_csv('test_X.csv')

# 將字母標籤轉換為二元形式
train_data['lettr'] = train_data['lettr'].apply(lambda x: 1 if x in ['B','H','P','W','R','M'] else -1)

# 切分訓練數據的特徵和標籤
X_train = train_data.drop('lettr', axis=1)
y_train = train_data['lettr']

# 將資料轉換為 torch.Tensor
X_train_tensor = torch.Tensor(X_train.values)
X_test_tensor = torch.Tensor(test_data.values)

# 建立 DataLoader
train_loader = DataLoader(TensorDataset(X_train_tensor), batch_size=32, shuffle=True)

# 定義自編碼器模型
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(X_train.shape[1], 128), nn.ReLU(True), nn.Linear(128, 64))
        self.decoder = nn.Sequential(nn.Linear(64, 128), nn.ReLU(True), nn.Linear(128, X_train.shape[1]))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# 初始化模型和優化器
model = AutoEncoder()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 訓練模型
for epoch in range(100):
    for data in train_loader:
        X = data[0]
        output = model(X)
        loss = criterion(output, X)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('epoch [{}/{}], loss:{:.4f}'.format(epoch+1, 100, loss.item()))

# 在測試數據上進行異常檢測
model.eval()
with torch.no_grad():
    test_output = model(X_test_tensor)
    mse_loss = nn.MSELoss(reduction='none')
    losses = mse_loss(test_output, X_test_tensor)
    anomaly_scores = torch.mean(losses, dim=1)

# 輸出結果
result = pd.DataFrame(list(range(len(anomaly_scores))), columns=['id'])
result['outliers'] = anomaly_scores.numpy()
result.to_csv('submission.csv', index=False)