# 此作業為透過AI對話設計nn0.py的學習範例
## 實作項目:微型自動微分nn0.py
[AI對話連結](https://gemini.google.com/share/b58e9448e8dd)  
下為AI統整的README
# nn0.py: 輕量級深度學習引擎解析

`nn0.py` 是一個極簡主義的深度學習實作，旨在去魔術化（Demystify）現代框架（如 PyTorch）的底層邏輯。它展示了神經網路最核心的兩大支柱：**自動微分 (Autograd)** 與 **梯度下降優化 (Gradient Descent)**。

---

## 🚀 核心元件介紹

### 1. Value 類別：具有記憶能力的數值
`Value` 是整個引擎的心臟。當數字被包裝成 `Value(x)` 時，它不再只是個純量，而是一個運算節點。

* **前向傳播 (Forward)**：對 `Value` 進行運算（如 `a + b`）時，它會記錄運算關係並儲存於 `_children` 中，自動構建出一張 **有向無環圖 (DAG)**。
* **反向傳播 (Backward)**：呼叫 `loss.backward()` 時，引擎會利用 **微積分連鎖律 (Chain Rule)**，從 Loss 開始反向遍歷計算圖，自動計算每個變數對最終結果的影響力（即梯度 `grad`）。

### 2. linear(x, w)：特徵轉換
執行基礎的矩陣運算 $y = Wx$。
* 這是神經網路的「骨架」。
* 透過調整權重矩陣 $W$，模型學習如何將原始輸入（如感測器數據）映射到正確的特徵空間。

### 3. softmax(logits)：數值機率化
將神經網路輸出的原始數值（Logits）轉化為正規化的機率分佈：

$$P(y_i) = \frac{e^{z_i}}{\sum_{j} e^{z_j}}$$

> [!IMPORTANT]
> **數值穩定性**：`nn0.py` 在實作時會預先減去 `max_val`（即 $z_i - \max(z)$），這能有效避免計算指數時產生浮點數溢位（Overflow），是工業界處理 Softmax 的標準做法。

---

## 🛠️ 進階優化與正規化

### 4. Adam：智慧型步伐更新者
**Adam (Adaptive Moment Estimation)** 決定了模型更新權重的「步伐」。
* **動量 (Momentum)**：它不只看當下的梯度，還會參考過去梯度的移動平均。
* **自適應**：能讓訓練過程更平穩，有效過濾單一數據產生的雜訊，並在複雜的誤差地形中快速收斂。

### 5. RMSNorm (Root Mean Square Layer Normalization)
現代大語言模型（如 Llama 系列）常用的正規化技術。
* 相對於傳統 LayerNorm，RMSNorm 捨棄了平移（Mean Subtraction），僅保留縮放。
* **優勢**：計算量更小，且實驗證明能顯著穩定 Transformer 結構的訓練過程。

### 6. gd 函數：語言模型訓練步
這是一個專為序列預測（如文字接龍）設計的訓練邏輯。
* 預期接收一系列的 tokens。
* 在每個時間步長（Timesteps）上進行預測與誤差回傳，非常適合用來實作小型 Transformer 雛形。

---

## 💡 快速上手範例

```python
from nn0 import Value

# 1. 定義輸入與權重
x = Value(0.5)
w = Value(-1.0)

# 2. 前向傳播 (自動建構計算圖)
y = x * w
loss = y**2

# 3. 反向傳播 (自動計算梯度)
loss.backward()

print(f"Loss: {loss.data}")
print(f"Weight Gradient (dL/dw): {w.grad}")
