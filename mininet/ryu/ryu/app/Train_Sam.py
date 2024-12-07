import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from packet_manager import PacketManager
import logging

class SwitchModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SwitchModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model():
    # 모델 초기화
    model = SwitchModel(input_size=2, hidden_size=64, num_classes=10).to(device)
    
    # 데이터 로더 설정
    packet_manager = PacketManager('packet_data.csv',initialize=False)
    batch_data = packet_manager.load_batch(100)
    if not batch_data:
        print("데이터를 로드할 수 없습니다.")
        return
    
    inputs = torch.tensor([[x[0], x[1]] for x in batch_data], dtype=torch.float32)
    labels = torch.tensor([x[2] for x in batch_data], dtype=torch.long)
    
    dataset = TensorDataset(inputs, labels)
    loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
    
    # 손실 함수와 옵티마이저 설정
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 모델 학습
    model.train()
    for epoch in range(10):
        epoch_loss = 0.0
        for batch_inputs, batch_labels in loader:
            optimizer.zero_grad()
            outputs = model(batch_inputs.to(device))
            loss = criterion(outputs, batch_labels.to(device))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print("Epoch {}, Loss: {:.4f}".format(epoch+1, epoch_loss))
    
    # 모델 저장
    torch.save(model.state_dict(), 'switch_model.pth')
    print("모델 학습 완료 및 저장됨.")

if __name__ == "__main__":
    train_model()