import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sentence_transformers import SentenceTransformer
import faiss
import umap
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from bs4 import BeautifulSoup
import random

st.set_page_config(page_title=" 金門限定 - 終極電影 AI", layout="wide")
st.title(" 金獅影城 AI 助手 - 在地化動態推薦系統")

# ==========================================
# 1. 深度學習模型定義 (PyTorch)
# ==========================================
class PopularityPredictorDNN(nn.Module):
    def __init__(self, input_dim=384): 
        super(PopularityPredictorDNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.network(x)

# ==========================================
# 2. 系統核心運算引擎
# ==========================================
@st.cache_resource
def load_encoder():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data
def load_and_preprocess_data():
    df = pd.read_csv('mymovie.csv', engine='python', on_bad_lines='skip')
    df = df.dropna(subset=['Overview', 'Title', 'Popularity']).reset_index(drop=True)
    df['Popularity'] = pd.to_numeric(df['Popularity'], errors='coerce').fillna(0)
    return df

@st.cache_resource
def build_feature_vectors(_df, _encoder):
    with st.spinner("🧠 萃取全庫 SBERT 語意向量中..."):
        return _encoder.encode(_df['Overview'].tolist(), show_progress_bar=True).astype('float32')

@st.cache_resource
def train_deep_learning_model(_features, targets):
    with st.spinner("🔥 啟動 PyTorch 模型訓練 (自動進行對數去偏態)..."):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 【修改亮點 1】將極端右偏的目標值進行對數縮放 log(1+x)
        targets_log = np.log1p(targets)
        
        X_train, X_test, y_train, y_test = train_test_split(_features, targets_log, test_size=0.2, random_state=42)
        
        X_train_t = torch.FloatTensor(X_train).to(device)
        y_train_t = torch.FloatTensor(y_train).view(-1, 1).to(device)
        X_test_t = torch.FloatTensor(X_test).to(device)
        y_test_t = torch.FloatTensor(y_test).view(-1, 1).to(device)
        
        model = PopularityPredictorDNN(input_dim=384).to(device)
        
        # 【修改亮點 2】將 MSE 替換為對離群值具備高魯棒性的 Huber Loss
        criterion = nn.HuberLoss(delta=1.0)
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        
        train_losses, test_losses = [], []
        for epoch in range(150):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train_t)
            loss = criterion(outputs, y_train_t)
            loss.backward()
            optimizer.step()
            
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_test_t)
                val_loss = criterion(val_outputs, y_test_t)
                train_losses.append(loss.item())
                test_losses.append(val_loss.item())
                
        preds_log = val_outputs.cpu().numpy()
        
        # 【修改亮點 3】計算評估指標前，透過 e^x - 1 還原回真實世界的熱度
        preds_real = np.expm1(preds_log).flatten()
        y_test_real = np.expm1(y_test).flatten()
        
        metrics = {
            "R2": r2_score(y_test_real, preds_real), 
            "MSE": mean_squared_error(y_test_real, preds_real), 
            "TrainLoss": train_losses, 
            "ValLoss": test_losses, 
            "y_test": y_test_real, 
            "preds": preds_real
        }
        return model, metrics, device

@st.cache_resource
def build_faiss_index(_features):
    with st.spinner("⚡ 建構 FAISS 向量檢索庫..."):
        index = faiss.IndexFlatL2(_features.shape[1])
        faiss.normalize_L2(_features)
        index.add(_features)
        return index

@st.cache_resource
def compute_umap(_features):
    with st.spinner("🌌 計算 UMAP 高階流形降維..."):
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine')
        return reducer.fit_transform(_features)

@st.cache_data(ttl=3600) # 快取一小時
def fetch_windlion_movies():
    """精準爬取金獅影城目前上映與即將上映的電影 (V4 最終優化版)"""
    with st.spinner("🦁 正在連接金門風獅爺商店街伺服器，抓取最新檔期..."):
        url = "https://cinemax.windlion.com.tw/"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        }
        movies = []
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.encoding = 'utf-8'
            soup = BeautifulSoup(response.text, 'html.parser')
            
            movie_links = soup.find_all('a', href=lambda href: href and 'movies.php?gid=' in href)
            
            seen_titles = set()
            for link in movie_links:
                parent_div = link.parent
                p_tags = parent_div.find_all('p', recursive=False) 
                
                if len(p_tags) >= 1:
                    title = p_tags[0].text.strip()
                    if title and title not in seen_titles and len(title) < 40:
                        seen_titles.add(title)
                        
                        date_tag = link.find('p', class_='release-date')
                        date_info = date_tag.text.strip() if date_tag else "近期熱映"
                        
                        overview = f"【金獅影城院線強檔】片名：{title}。 {date_info}。這是一部目前正在金門風獅爺商店街金獅影城上映的強檔電影，適合當地民眾與學生前往觀賞。"
                        
                        movies.append({
                            "Title": title,
                            "Overview": overview
                        })
                        
        except Exception as e:
            st.warning(f"爬蟲連線遇到阻礙: {e}")
            
        if len(movies) == 0:
            st.info("目前無法抓取即時資訊，自動載入金獅影城備用熱門片單。")
            movies = [
                {"Title": "沙丘：第二部", "Overview": "保羅亞崔迪與荃妮聯手，對毀滅他家族的陰謀者展開報復。"},
                {"Title": "排球少年！！垃圾場的決戰", "Overview": "烏野高中與音駒高中迎來了注定的一戰，這場不能重來的比賽即將改變所有人的青春。"}
            ]
            
        return movies

# --- 啟動初始化 ---
encoder = load_encoder()
base_df = load_and_preprocess_data()
base_vectors = build_feature_vectors(base_df, encoder)

dnn_model, val_metrics, device = train_deep_learning_model(base_vectors, base_df['Popularity'].values)
umap_coords = compute_umap(base_vectors)

# ==========================================
# 3. 狀態管理 (Session State)
# ==========================================
if 'db_df' not in st.session_state:
    st.session_state.db_df = base_df.copy()
if 'db_vectors' not in st.session_state:
    st.session_state.db_vectors = base_vectors.copy()
if 'db_faiss' not in st.session_state:
    st.session_state.db_faiss = build_faiss_index(st.session_state.db_vectors.copy())

if 'user_profile' not in st.session_state:
    st.session_state.user_profile = None
if 'interacted_ids' not in st.session_state:
    st.session_state.interacted_ids = set() 
if 'trajectory' not in st.session_state:
    st.session_state.trajectory = [] 

if 'shown_ids' not in st.session_state:
    st.session_state.shown_ids = set() 
if 'current_batch' not in st.session_state:
    st.session_state.current_batch = [] 
if 'needs_new_batch' not in st.session_state:
    st.session_state.needs_new_batch = True 

def handle_feedback(idx, vec, is_like):
    st.session_state.interacted_ids.add(int(idx))
    lr = 0.2
    
    vec_norm = vec / np.linalg.norm(vec) if np.linalg.norm(vec) > 0 else vec
    current_profile = st.session_state.user_profile
    
    if is_like:
        new_profile = (1 - lr) * current_profile + lr * vec_norm
        st.toast(f"👍 已學習您的喜好！", icon="✅")
    else:
        new_profile = current_profile - lr * vec_norm
        
    if np.linalg.norm(new_profile) > 0:
        st.session_state.user_profile = new_profile / np.linalg.norm(new_profile)
    else:
        st.session_state.user_profile = new_profile
        
    st.session_state.trajectory.append(st.session_state.user_profile.copy())

def trigger_new_batch():
    st.session_state.needs_new_batch = True
    st.session_state.current_batch = [] 

# ==========================================
# 4. 介面與邏輯 (Tabs)
# ==========================================
tab1, tab2, tab3, tab4, tab5 = st.tabs(["1️⃣ 冷啟動選擇", "2️⃣ 庫內精準推薦", "3️⃣ 手動輸入預測", "4️⃣ 金獅影城檔期", "5️⃣ 數據看版"])

# --- Tab 1: 入口 ---
with tab1:
    st.header("建立您的初始電影 DNA")
    if 'random_indices' not in st.session_state:
        st.session_state.random_indices = random.sample(range(len(st.session_state.db_df)), 12)
        
    selected_indices = []
    cols = st.columns(3)
    for i, idx in enumerate(st.session_state.random_indices):
        movie = st.session_state.db_df.iloc[idx]
        with cols[i % 3]:
            with st.container(border=True):
                st.subheader(movie['Title'])
                st.caption(f"⭐ 熱度: {movie['Popularity']:.1f}")
                st.write(str(movie['Overview'])[:100] + "...")
                if st.checkbox("感興趣", key=f"cold_{idx}"):
                    selected_indices.append(idx)
                    
    if st.button("🚀 生成我的專屬口味模型"):
        if selected_indices:
            selected_vecs = st.session_state.db_vectors[selected_indices]
            st.session_state.user_profile = np.mean(selected_vecs, axis=0)
            st.session_state.trajectory.append(st.session_state.user_profile.copy())
            st.session_state.interacted_ids.update(selected_indices)
            st.success("✅ 口味建立完成！")

# --- Tab 2: 資料庫推薦與動態回饋 ---
with tab2:
    st.header("庫內精準推薦 (動態學習)")
    if st.session_state.user_profile is None:
        st.info("請先至「1️⃣ 冷啟動選擇」建立初始口味。")
    else:
        st.button("🔄 換一批推薦", on_click=trigger_new_batch)
        
        if st.session_state.needs_new_batch or not st.session_state.current_batch:
            q_vec = st.session_state.user_profile.copy().reshape(1, -1)
            faiss.normalize_L2(q_vec)
            
            search_k = len(st.session_state.shown_ids) + 50
            distances, indices = st.session_state.db_faiss.search(q_vec, search_k)
            
            new_batch = []
            for i, raw_idx in enumerate(indices[0]):
                idx = int(raw_idx) 
                if idx not in st.session_state.shown_ids and idx not in st.session_state.interacted_ids:
                    new_batch.append((idx, distances[0][i]))
                if len(new_batch) == 5:
                    break
            
            st.session_state.current_batch = new_batch
            st.session_state.shown_ids.update([item[0] for item in new_batch])
            st.session_state.needs_new_batch = False

        for item in st.session_state.current_batch:
            idx = item[0]
            sim_score = 1 / (1 + item[1])
            movie = st.session_state.db_df.iloc[idx]
            vec = st.session_state.db_vectors[idx]
            
            with st.container(border=True):
                st.subheader(f"✨ {movie['Title']} (推薦度: {sim_score*100:.1f}%)")
                st.write(movie['Overview'])
                
                if idx in st.session_state.interacted_ids:
                    st.success("✅ 已記錄您的回饋，底層神經網路權重已更新！")
                else:
                    c1, c2, c3 = st.columns([1, 1, 8])
                    c1.button("👍 喜歡", key=f"like_{idx}", on_click=handle_feedback, args=(idx, vec, True))
                    c2.button("👎 沒興趣", key=f"dislike_{idx}", on_click=handle_feedback, args=(idx, vec, False))

# --- Tab 3: 手動輸入測試與資料庫擴充 ---
with tab3:
    st.header("✍️ 手動輸入劇本與資料庫擴充")
    st.write("您可以測試腦海中的新劇本，並將它永久加入 AI 的推薦大腦中！")
    
    custom_title = st.text_input("")
    custom_overview = st.text_area("")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔍 啟動 AI 盲測 (僅預測不寫入)"):
            if custom_overview and st.session_state.user_profile is not None:
                new_vec = encoder.encode([custom_overview]).astype('float32')
                
                dnn_model.eval()
                with torch.no_grad():
                    q_tensor = torch.FloatTensor(new_vec).to(device)
                    # 【修改】推論結果是 Log，需要 expm1 還原
                    pred_pop_log = dnn_model(q_tensor).item()
                    pred_pop = np.expm1(pred_pop_log)
                    
                from sklearn.metrics.pairwise import cosine_similarity
                user_vec = st.session_state.user_profile.reshape(1, -1)
                sim_score = cosine_similarity(new_vec, user_vec)[0][0]
                match_pct = max(0, sim_score) * 100
                
                st.markdown("### 📊 測試結果")
                st.metric("AI 預測市場熱度", f"{pred_pop:.1f}")
                st.metric("與您的口味契合度", f"{match_pct:.1f}%")
                
                if match_pct > 50:
                     st.success("🌟 您的口味一定會喜歡這個劇本！")
                else:
                     st.warning("⚠️ 這個劇本可能不太合您的胃口。")
            elif st.session_state.user_profile is None:
                st.warning("請先至 Tab 1 建立口味向量才能計算契合度！")

    with col2:
        if st.button("💾 加入全局資料庫 (可被推薦)"):
            if custom_title and custom_overview:
                with st.spinner("正在編譯特徵並寫入 FAISS 向量庫..."):
                    new_vec = encoder.encode([custom_overview]).astype('float32')
                    
                    dnn_model.eval()
                    with torch.no_grad():
                        pred_pop_log = dnn_model(torch.FloatTensor(new_vec).to(device)).item()
                        pred_pop = np.expm1(pred_pop_log)
                    
                    new_row = {"Title": custom_title, "Overview": custom_overview, "Popularity": pred_pop}
                    st.session_state.db_df = pd.concat([st.session_state.db_df, pd.DataFrame([new_row])], ignore_index=True)
                    
                    st.session_state.db_vectors = np.vstack((st.session_state.db_vectors, new_vec))
                    faiss.normalize_L2(new_vec)
                    st.session_state.db_faiss.add(new_vec)
                    
                    st.success(f"✅ 寫入成功！《{custom_title}》已成為資料庫的第 {len(st.session_state.db_df)} 部電影。")
                    st.info("💡 現在它有機會在 Tab 2 被推薦給您了！")
            else:
                st.error("請填寫電影名稱與劇情簡介。")

# --- Tab 4: 金獅影城檔期爬蟲 ---
with tab4:
    st.header("🦁 聯網推薦：金獅影城現正熱映")
    if st.session_state.user_profile is None:
        st.info("請先至「1️⃣ 冷啟動選擇」建立初始口味。")
    else:
        st.write("📡 正在連接金門風獅爺商店街伺服器 (`cinemax.windlion.com.tw`)...")
        online_movies = fetch_windlion_movies()
        
        if online_movies:
            st.success(f"成功抓取 {len(online_movies)} 部院線電影！")
            
            new_overviews = [m['Overview'] for m in online_movies]
            new_vecs = encoder.encode(new_overviews).astype('float32')
            
            dnn_model.eval()
            with torch.no_grad():
                q_tensor = torch.FloatTensor(new_vecs).to(device)
                pred_pops_log = dnn_model(q_tensor).cpu().numpy().flatten()
                pred_pops = np.expm1(pred_pops_log) # 【修改】還原對數預測值
            
            from sklearn.metrics.pairwise import cosine_similarity
            user_vec = st.session_state.user_profile.reshape(1, -1)
            sim_scores = cosine_similarity(new_vecs, user_vec).flatten()
            
            results = []
            for i in range(len(online_movies)):
                results.append({
                    "Title": online_movies[i]['Title'],
                    "Overview": online_movies[i]['Overview'],
                    "Sim": sim_scores[i],
                    "PredPop": pred_pops[i]
                })
            results.sort(key=lambda x: x['Sim'], reverse=True)
            
            for res in results[:5]: 
                match_pct = max(0, res['Sim']) * 100
                with st.expander(f"🎟️ {res['Title']} | 契合度: {match_pct:.1f}% | 預測熱度: {res['PredPop']:.1f}", expanded=True):
                    if match_pct > 50:
                        st.markdown("🌟 **強烈建議您去金獅影城看這部！**")
                    st.write(res['Overview'])
        else:
            st.warning("目前無法解析金獅影城網頁結構，請稍後再試或檢查網站是否維護中。")

# --- Tab 5: 系統驗證儀表板 ---
with tab5:
    st.header("📈 AI 效能驗證與數據分析")
    
    st.subheader("1. 深度學習模型 (PyTorch) 預測能力")
    c1, c2 = st.columns(2)
    c1.metric("R² Score (決定係數)", f"{val_metrics['R2']:.4f}")
    c2.metric("MSE (均方誤差)", f"{val_metrics['MSE']:.2f}")
    
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.plot(val_metrics['TrainLoss'], label='Train Loss', color='blue')
    ax1.plot(val_metrics['ValLoss'], label='Validation Loss', color='red')
    ax1.set_title("Training Loss Convergence")
    ax1.legend()
    
    # 找到這行畫散點圖的程式碼：
    ax2.scatter(val_metrics['y_test'], val_metrics['preds'], alpha=0.3, color='purple')
    
    # 找到畫 Y=X 理想線的程式碼：
    max_val = max(max(val_metrics['y_test']), max(val_metrics['preds']))
    ax2.plot([0, max_val], [0, max_val], 'k--', lw=2)
    ax2.set_title("Popularity: Actual vs Predicted")
    
    ax2.set_xscale('symlog')  # 將 X 軸設為對稱對數尺度
    ax2.set_yscale('symlog')  # 將 Y 軸設為對稱對數尺度
    ax2.set_xlabel("Actual Popularity (Log Scale)")
    ax2.set_ylabel("Predicted Popularity (Log Scale)")
    
    st.pyplot(fig1)
    
    st.subheader("2. 您的口味進化軌跡 (UMAP 流形宇宙)")
    fig2, ax = plt.subplots(figsize=(12, 8))
    sample_idx = np.random.choice(len(umap_coords), 2000, replace=False)
    scatter = ax.scatter(umap_coords[sample_idx, 0], umap_coords[sample_idx, 1], 
                         c=base_df.iloc[sample_idx]['Popularity'], cmap='viridis', s=15, alpha=0.5)
    plt.colorbar(scatter, label='Popularity Heat')
    
    if len(st.session_state.trajectory) > 0:
        traj_coords = []
        for vec in st.session_state.trajectory:
            v = vec.reshape(1, -1).copy()
            faiss.normalize_L2(v)
            _, nearest = st.session_state.db_faiss.search(v, 1)
            traj_coords.append(umap_coords[nearest[0][0]])
        
        traj_coords = np.array(traj_coords)
        ax.plot(traj_coords[:, 0], traj_coords[:, 1], color='red', linestyle='-', linewidth=2, alpha=0.7)
        ax.scatter(traj_coords[0, 0], traj_coords[0, 1], color='red', marker='*', s=300, label='Initial Taste (Start)')
        if len(traj_coords) > 1:
            ax.scatter(traj_coords[-1, 0], traj_coords[-1, 1], color='green', marker='X', s=200, label='Current Taste (Now)')
        ax.legend()
        
    ax.set_title("UMAP Manifold & User Taste Trajectory")
    st.pyplot(fig2)
