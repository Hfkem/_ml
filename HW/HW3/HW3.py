import random
from nn0 import Value, Adam, linear, softmax

# --- 輔助函數 ---
def init_weights(rows, cols):
    """隨機初始化權重矩陣"""
    return [[Value(random.uniform(-1.0, 1.0)) for _ in range(cols)] for _ in range(rows)]

# --- 1. 準備訓練資料 ---
# 輸入 x: [pitch, roll] (簡化的無人機姿態特徵)
# 目標 y: 狀態類別 (0: 穩定, 1: 移動中, 2: 不穩定)
data = [
    ([0.1, 0.0],  0),
    ([0.8, 0.2],  1),
    ([-0.1, 0.9], 2),
    ([0.0, -0.1], 0)
]

# --- 2. 定義神經網路參數 ---
# 輸入維度: 2, 輸出維度 (類別數): 3
# 注意：nn0.py 的 linear 函數實作了 W @ x，沒有加上偏差值 (bias)，這裡保持簡單使用純矩陣乘法。
W_out = init_weights(3, 2)

# 攤平所有權重以交給優化器
params = [w for row in W_out for w in row]

# --- 3. 初始化優化器 ---
optimizer = Adam(params, lr=0.1)

# --- 4. 訓練迴圈 ---
epochs = 50
print("開始訓練...")

for epoch in range(epochs):
    total_loss = Value(0.0)
    
    for x_val, y_true in data:
        # 將浮點數轉換為計算圖中的 Value 節點
        x = [Value(xi) for xi in x_val]
        
        # 前向傳播 (Forward Pass)
        # 1. 線性轉換: z = W @ x
        logits = linear(x, W_out)
        
        # 2. 轉換為機率分佈
        probs = softmax(logits)
        
        # 3. 計算交叉熵損失 (Cross-Entropy Loss): -log(P(y_true))
        loss = -probs[y_true].log()
        total_loss = total_loss + loss

    # 計算平均損失
    avg_loss = total_loss / Value(len(data))
    
    # 反向傳播 (Backward Pass): 計算所有參預運算的 Value 的梯度
    avg_loss.backward()
    
    # 更新參數並清空梯度
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:2d} | Loss: {avg_loss.data:.4f}")

print("\n訓練完成！來測試一下網路的預測能力：")
for x_val, y_true in data:
    x = [Value(xi) for xi in x_val]
    logits = linear(x, W_out)
    probs = softmax(logits)
    pred_class = max(range(len(probs)), key=lambda i: probs[i].data)
    print(f"輸入: {x_val} | 真實標籤: {y_true} | 預測標籤: {pred_class} | 預測機率: {probs[pred_class].data:.2f}")
