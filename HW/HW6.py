import numpy as np

class SemanticResonanceField:
    def __init__(self, vocab, wave_dim=100, decay_rate=0.5, learning_rate=0.01):
        # [機制 C] decay_rate 預設調降為 0.5，讓舊波形更快消散，專注當下
        self.vocab = vocab
        self.word_to_id = {word: i for i, word in enumerate(vocab)}
        self.id_to_word = {i: word for i, word in enumerate(vocab)}
        self.vocab_size = len(vocab)
        self.wave_dim = wave_dim
        self.decay_rate = decay_rate
        self.lr = learning_rate
        
        # 初始化每個字的頻率特徵
        self.vocab_waves = np.random.uniform(-1, 1, (self.vocab_size, self.wave_dim))
        self.current_pool_wave = np.zeros(wave_dim)

    def input_word(self, word):
        if word not in self.word_to_id: return
        word_id = self.word_to_id[word]
        
        # 衰減與疊加
        self.current_pool_wave = (self.current_pool_wave * self.decay_rate) + np.sin(self.vocab_waves[word_id])

    def train(self, corpus, epochs=5):
        print(f"🌊 開始共鳴訓練... (文本長度: {len(corpus)} 字, 訓練輪數: {epochs})")
        for epoch in range(epochs):
            self.current_pool_wave = np.zeros(self.wave_dim) 
            for i in range(len(corpus) - 1):
                curr_word = corpus[i]
                next_word = corpus[i+1]
                
                if curr_word in self.word_to_id and next_word in self.word_to_id:
                    self.input_word(curr_word)
                    next_id = self.word_to_id[next_word]
                    
                    self.vocab_waves[next_id] += self.lr * self.current_pool_wave
                    self.vocab_waves[next_id] /= np.linalg.norm(self.vocab_waves[next_id])

    def predict_next(self, recent_ids=None, all_generated_ids=None, top_k=3, temperature=1.0, presence_penalty=0.2):
        # 1. 計算原始共鳴
        resonances = np.dot(self.vocab_waves, self.current_pool_wave)
        
        # 2. [機制 D] 存在懲罰 (Presence Penalty)：針對已生成的字輕微扣減能量
        if all_generated_ids:
            for wid in all_generated_ids:
                resonances[wid] -= presence_penalty
                
        # 3. [機制 A] 最近記憶過濾 (N-Gram 阻斷)：針對最近 N 個字施加極大懲罰
        if recent_ids:
            for wid in recent_ids:
                resonances[wid] -= 999.0 
            
        energy_stability = 1.0 / (np.std(resonances) + 1e-6)
        
        # 4. [機制 B] 機率採樣 (Temperature Sampling)
        if temperature > 0:
            # 防溢位處理：先減去最大值
            res_shifted = resonances - np.max(resonances)
            exp_res = np.exp(res_shifted / temperature)
            probs = exp_res / np.sum(exp_res)
            
            # 根據機率隨機抽取 top_k 的候選
            best_indices = np.random.choice(self.vocab_size, size=top_k, p=probs, replace=False)
            # 將抽出的字依照原始共鳴值由高到低排序
            best_indices = sorted(best_indices, key=lambda idx: resonances[idx], reverse=True)
        else:
            # 退回傳統的貪婪搜尋 (Top-1)
            best_indices = np.argsort(resonances)[-top_k:][::-1]
            
        results = [(self.id_to_word[idx], resonances[idx]) for idx in best_indices]
        return results, energy_stability

    def generate_sentence(self, start_word, max_length=15, temperature=0.6, presence_penalty=0.5, block_n=2):
        """整合四大機制的自動接龍生成器"""
        if start_word not in self.word_to_id:
            return f"字典裡沒有 '{start_word}' 這個字。"

        print(f"\n🚀 開始生成接龍，初始輸入：【{start_word}】(溫度: {temperature})")
        self.current_pool_wave = np.zeros(self.wave_dim)
        self.input_word(start_word)
        
        sentence = [start_word]
        generated_ids = [self.word_to_id[start_word]]
        current_word = start_word

        for _ in range(max_length):
            # 取出最近 block_n 個字的 ID
            recent_ids = generated_ids[-block_n:] if block_n > 0 else []
            
            # 帶入所有懲罰與溫度機制
            predictions, _ = self.predict_next(
                recent_ids=recent_ids, 
                all_generated_ids=generated_ids,
                top_k=1, 
                temperature=temperature,
                presence_penalty=presence_penalty
            )
            next_word = predictions[0][0]
            
            sentence.append(next_word)
            generated_ids.append(self.word_to_id[next_word])
            
            if next_word == "。":
                break
                
            self.input_word(next_word)
            current_word = next_word
            
        return " ".join(sentence)


# --- 1. 準備擴充學習文本 (保留原始內容) ---
training_text = """
小貓 坐 在 桌上 。 小狗 跑 在 路上 。 小鳥 飛 在 天上 。 
小魚 游 在 水中 。 大貓 坐 在 地上 。 大狗 跑 在 地上 。 
大鳥 飛 在 山上 。 小貓 吃 了 魚 。 小狗 吃 了 肉 。 
小鳥 吃 了 蟲 。 小魚 吃 了 草 。 大貓 吃 了 魚 。 
大狗 吃 了 肉 。 大鳥 吃 了 蟲 。 我 看到 一隻 小貓 。 
我 看到 一隻 小狗 。 我 看到 一隻 小鳥 。 我 看到 一隻 小魚 。 
你 看到 一隻 大貓 。 你 看到 一隻 大狗 。 你 看到 一隻 大鳥 。 
他 看到 一隻 大魚 。 我 喜歡 小貓 。 我 喜歡 小狗 。 
我 喜歡 小鳥 。 你 喜歡 大貓 。 你 喜歡 大狗 。 
他 喜歡 小魚 。 她 喜歡 大鳥 。 天上 有 白雲 。 
天上 有 太陽 。 天上 有 月亮 。 天上 有 星星 。 
山上 有 大樹 。 山上 有 小花 。 
山上 有 小草 。 水中 有 小魚 。 
水中 有 大魚 。 地上 有 小貓 。 
地上 有 大狗 。 今天 天氣 好 。 
今天 太陽 大 。 今天 風 很大 。 
今天 雨 很大 。 小貓 很 可愛 。 
小狗 很 可愛 。 小鳥 很 可愛 。 
大貓 很 漂亮 。 大狗 很 漂亮 。 
大鳥 很 漂亮 。 我 去 學校 上課 。 
你 去 學校 上課 。 他 去 學校 上課 。 
她 去 學校 上課 。 我 在 家裡 吃飯 。 
你 在 家裡 吃飯 。 他 在 家裡 看書 。 
她 在 家裡 看書 。 我 和 你 是 好朋友 。 
你 和 他 是 好朋友 。 他 和 她 是 好朋友 。 
我們 都 是 好朋友 。 早上 太陽 出來 了 。 
中午 太陽 很大 。 下午 風 很大 。 
晚上 月亮 出來 了 。 晚上 星星 很多 。 
小花 很 漂亮 。 小草 很多 。 
大樹 很高 。 白雲 很 白 。 
太陽 很大 。 月亮 很 亮 。 
星星 很 小 。 我 愛 吃飯 。 
你 愛 看書 。 他 愛 跑步 。 
她 愛 唱歌 。 我 會 寫字 。 
你 會 看書 。 他 會 跑步 。 
她 會 唱歌 。 我 想 吃 魚 。 
你 想 吃 肉 。 他 想 看書 。 
她 想 唱歌 。 一隻 小貓 在 桌上 。 
一隻 小狗 在 地上 。 一隻 小鳥 在 天上 。 
一隻 小魚 在 水中 。 兩隻 小貓 在 地上 。 
兩隻 小狗 在 路上 。 三隻 小鳥 在 天上 。 
很多 小魚 在 水中 。 春天 小花 很多 。 
夏天 太陽 很大 。 秋天 風 很大 。 
冬天 雨 很大 。 春天 很 美 。 
夏天 很 熱 。 秋天 很 涼 。 
冬天 很 冷 。 我 在 春天 看 花 。 
你 在 夏天 游水 。 他 在 秋天 看書 。 
她 在 冬天 唱歌 。 小貓 和 小狗 是 好朋友 。 
小鳥 和 小魚 是 好朋友 。 大貓 和 大狗 是 好朋友 。 
我 和 小貓 是 好朋友 。 你 和 小狗 是 好朋友 。 
他 喜歡 在 山上 跑步 。 她 喜歡 在 水中 游水 。 
我 喜歡 在 家裡 看書 。 你 喜歡 在 學校 上課 。 
天上 的 白雲 很 漂亮 。 山上 的 大樹 很高 。 
水中 的 小魚 很 可愛 。 地上 的 小花 很 美 。 
早上 我 去 學校 上課 。 中午 我 在 學校 吃飯 。 
下午 我 在 家裡 看書 。 晚上 我 在 家裡 吃飯 。 
早上 你 去 學校 上課 。 中午 你 在 學校 吃飯 。 
下午 你 在 家裡 看書 。 晚上 你 在 家裡 吃飯 。 
小貓 在 家裡 。 小狗 在 路上 。 
小鳥 在 山上 。 小魚 在 水中 。 
大樹 在 山上 。 小花 在 地上 。 
太陽 在 天上 。 月亮 在 天上 。 
星星 在 天上 。 白雲 在 天上 。 
我 很 開心 。 你 很 開心 。 
他 很 開心 。 她 很 開心 。 
我們 很 開心 。 大家 都 很 開心 。 
今天 很 開心 。 我 今天 很 開心 。 
你 今天 很 開心 。 他 今天 很 開心 。
"""
corpus = training_text.split()
vocab = list(set(corpus))

# --- 2. 初始化並訓練 ---
# 採用機制C：調降 decay_rate=0.5 讓共鳴更專注當下
srf = SemanticResonanceField(vocab, wave_dim=128, decay_rate=0.5, learning_rate=0.05)
srf.train(corpus, epochs=8) # 增加訓練輪數讓溫度採樣的特徵更鮮明

# --- 3. 測試進階自動接龍 ---
print("\n" + "="*40)
# 你可以調整 temperature 看看句子的變化 (0.0=絕對理性, 1.0=充滿想像力)
generated_1 = srf.generate_sentence("早上", temperature=0.5)
print(f"生成結果: {generated_1}")

generated_2 = srf.generate_sentence("早上", temperature=0.7)
print(f"生成結果: {generated_2}")

generated_3 = srf.generate_sentence("早上", temperature=0.8)
print(f"生成結果: {generated_3}")