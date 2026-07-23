import io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error
import streamlit as st
import torch
import torch.nn as nn

st.set_page_config(page_title="Hisse Senedi Fiyat Tahmini (LSTM)",layout="wide")

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LSTMModel(nn.Module):
    def __init__(self,input_size=5,hidden_size=64,num_layers=2,output_size=5,dropout=0.2):
        super(LSTMModel,self).__init__()
        self.lstm=nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers>1 else 0.0,
        )
        self.fc=nn.Linear(hidden_size,output_size)

    def forward(self,x):
        out,(hn,cn)=self.lstm(x)
        return self.fc(out[:,-1,:])

def create_sequences(data,seq_length):
    xs,ys=[],[]
    for i in range(len(data)-seq_length):
        xs.append(data[i:(i+seq_length)])
        ys.append(data[i+seq_length])
    return torch.stack(xs),torch.stack(ys)

def inverse_transform(scaled_data,min_vals,max_vals):
    return scaled_data*(max_vals-min_vals)+min_vals

st.title("📈 Hisse Senedi Fiyat Tahmini — LSTM")
st.caption("Geçmiş OHLCV verisiyle eğitilen bir LSTM modeliyle gelecekteki kapanış fiyatını tahmin edin.")

st.sidebar.header("1️⃣ Veri")
uploaded_file=st.sidebar.file_uploader("CSV dosyası yükle (Date, Open, High, Low, Close, Volume)",type=["csv"])

use_sample=False
if uploaded_file is None:
    use_sample=st.sidebar.checkbox("Örnek AABA verisini kullan",value=True)

st.sidebar.header("2️⃣ Model Ayarları")
seq_length=st.sidebar.slider("Sequence uzunluğu (gün)",min_value=5,max_value=60,value=30,step=1)
hidden_size=st.sidebar.select_slider("Gizli katman boyutu (hidden_size)",options=[16,32,64,128],value=64)
num_layers=st.sidebar.slider("LSTM katman sayısı",min_value=1,max_value=4,value=2)
num_epochs=st.sidebar.slider("Epoch sayısı",min_value=10,max_value=300,value=100,step=10)
learning_rate=st.sidebar.select_slider("Öğrenme oranı (lr)",options=[0.01,0.005,0.001,0.0005],value=0.001)

train_button=st.sidebar.button("🚀 Modeli Eğit",use_container_width=True)
feature_cols=["Open","High","Low","Close","Volume"]

@st.cache_data
def generate_sample_data():
    np.random.seed(42)
    dates=pd.date_range(start="2006-01-03",end="2017-12-29",freq="B")
    n=len(dates)
    returns=np.random.normal(loc=0.0003,scale=0.02,size=n)
    close_prices=40.0*np.exp(np.cumsum(returns))
    open_prices=close_prices*(1+np.random.uniform(-0.01,0.01,n))
    high_prices=np.maximum(open_prices,close_prices)*(1+np.random.uniform(0.001,0.015,n))
    low_prices=np.minimum(open_prices,close_prices)*(1-np.random.uniform(0.001,0.015,n))
    volume=np.random.randint(10000000,80000000,size=n)
    return pd.DataFrame({
        "Date":dates,
        "Open":np.round(open_prices,2),
        "High":np.round(high_prices,2),
        "Low":np.round(low_prices,2),
        "Close":np.round(close_prices,2),
        "Volume":volume,
        "Name":"AABA",
    })

if uploaded_file is not None:
    df=pd.read_csv(uploaded_file)
elif use_sample:
    df=generate_sample_data()
else:
    st.warning("Lütfen bir CSV dosyası yükleyin veya örnek veriyi etkinleştirin.")
    st.stop()

missing_cols=[c for c in ["Date"]+feature_cols if c not in df.columns]
if missing_cols:
    st.error(f"Veri setinde eksik sütun(lar) var: {missing_cols}")
    st.stop()

df["Date"]=pd.to_datetime(df["Date"])
df=df.sort_values("Date").reset_index(drop=True)

col1,col2=st.columns([2,1])

with col1:
    st.subheader("📊 Kapanış Fiyatı Grafiği")
    fig,ax=plt.subplots(figsize=(10,4))
    ax.plot(df["Date"],df["Close"],color="#1f77b4")
    ax.set_xlabel("Tarih")
    ax.set_ylabel("Close Fiyatı")
    ax.grid(alpha=0.3)
    st.pyplot(fig)

with col2:
    st.subheader("🔎 Veri Önizleme")
    st.dataframe(df.tail(10),use_container_width=True,height=250)
    st.metric("Toplam Gün Sayısı",len(df))
    st.metric("Son Kapanış Fiyatı",f"{df['Close'].iloc[-1]:.2f}")

st.divider()

def train_model(df,seq_length,hidden_size,num_layers,num_epochs,lr):
    data=df[feature_cols].to_numpy()
    tensor=torch.tensor(data,dtype=torch.float32)

    n=tensor.shape[0]
    train_end=int(n*0.7)
    val_end=int(n*0.85)

    train_data=tensor[:train_end]
    val_data=tensor[train_end:val_end]
    test_data=tensor[val_end:]

    train_min=train_data.min(dim=0,keepdim=True).values
    train_max=train_data.max(dim=0,keepdim=True).values
    diff=train_max-train_min
    diff[diff==0]=1

    train_scaled=(train_data-train_min)/diff
    val_scaled=(val_data-train_min)/diff
    test_scaled=(test_data-train_min)/diff

    X_train,y_train=create_sequences(train_scaled,seq_length)
    X_val,y_val=create_sequences(val_scaled,seq_length)
    X_test,y_test=create_sequences(test_scaled,seq_length)

    X_train,y_train=X_train.to(device),y_train.to(device)
    X_val,y_val=X_val.to(device),y_val.to(device)
    X_test,y_test=X_test.to(device),y_test.to(device)

    model=LSTMModel(input_size=5,hidden_size=hidden_size,num_layers=num_layers,output_size=5).to(device)
    criterion=nn.MSELoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=lr)

    best_val_loss=float("inf")
    best_state=None
    patience,patience_counter=15,0

    train_losses,val_losses=[],[]
    progress_bar=st.progress(0,text="Eğitim başlıyor...")

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        output=model(X_train)
        loss=criterion(output,y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_output=model(X_val)
            val_loss=criterion(val_output,y_val)

        train_losses.append(loss.item())
        val_losses.append(val_loss.item())

        if val_loss.item()<best_val_loss:
            best_val_loss=val_loss.item()
            best_state={k:v.clone() for k,v in model.state_dict().items()}
            patience_counter=0
        else:
            patience_counter+=1
            if patience_counter>=patience:
                progress_bar.progress(1.0,text=f"Early stopping: epoch {epoch+1}")
                break

        progress_bar.progress((epoch+1)/num_epochs,text=f"Epoch {epoch+1}/{num_epochs} — Val Loss: {val_loss.item():.5f}")

    model.load_state_dict(best_state)
    model.eval()

    return {
        "model":model,
        "train_losses":train_losses,
        "val_losses":val_losses,
        "X_test":X_test,
        "y_test":y_test,
        "test_scaled":test_scaled,
        "train_min":train_min,
        "train_max":train_max,
        "best_val_loss":best_val_loss,
    }

if train_button:
    with st.spinner("Model eğitiliyor, lütfen bekleyin..."):
        result=train_model(df,seq_length,hidden_size,num_layers,num_epochs,learning_rate)
    st.session_state["result"]=result
    st.session_state["seq_length"]=seq_length
    st.success(f"Eğitim tamamlandı ✅ — En iyi doğrulama kaybı: {result['best_val_loss']:.6f}")

if "result" in st.session_state:
    result=st.session_state["result"]
    model=result["model"]
    seq_length_used=st.session_state["seq_length"]

    st.subheader("📉 Eğitim / Doğrulama Kaybı")
    fig,ax=plt.subplots(figsize=(10,4))
    ax.plot(result["train_losses"],label="Train Loss")
    ax.plot(result["val_losses"],label="Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)

    model.eval()
    with torch.no_grad():
        test_output=model(result["X_test"])

    test_pred_real=inverse_transform(test_output.cpu(),result["train_min"].cpu(),result["train_max"].cpu())
    y_test_real=inverse_transform(result["y_test"].cpu(),result["train_min"].cpu(),result["train_max"].cpu())

    close_idx=feature_cols.index("Close")
    y_true_close=y_test_real[:,close_idx].detach().numpy()
    y_pred_close=test_pred_real[:,close_idx].detach().numpy()

    rmse=np.sqrt(mean_squared_error(y_true_close,y_pred_close))
    mae=mean_absolute_error(y_true_close,y_pred_close)
    mape=np.mean(np.abs((y_true_close-y_pred_close)/y_true_close))*100

    naive_pred=y_true_close[:-1]
    naive_true=y_true_close[1:]
    naive_rmse=np.sqrt(mean_squared_error(naive_true,naive_pred))
    naive_mae=mean_absolute_error(naive_true,naive_pred)
    naive_mape=np.mean(np.abs((naive_true-naive_pred)/naive_true))*100

    st.subheader("🎯 Test Performansı (Close Fiyatı)")
    m1,m2,m3=st.columns(3)
    m1.metric("MAE",f"{mae:.3f}",delta=f"{mae-naive_mae:+.3f} vs naive",delta_color="inverse")
    m2.metric("RMSE",f"{rmse:.3f}",delta=f"{rmse-naive_rmse:+.3f} vs naive",delta_color="inverse")
    m3.metric("MAPE",f"%{mape:.2f}",delta=f"{mape-naive_mape:+.2f} vs naive",delta_color="inverse")
    st.caption("Delta değerleri, 'bir önceki günü tahmin olarak kullanma' (naive baseline) yöntemine göre farkı gösterir. Negatif değer, LSTM modelinin naive'den daha iyi olduğu anlamına gelir.")

    st.subheader("📈 Gerçek vs Tahmin (Test Seti)")
    fig,ax=plt.subplots(figsize=(12,5))
    ax.plot(y_true_close,label="Gerçek Close")
    ax.plot(y_pred_close,label="Tahmin Close")
    ax.set_xlabel("Zaman (Test Seti İndeksi)")
    ax.set_ylabel("Fiyat")
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)

    st.divider()

    st.subheader("🔮 Kendi Tahminini Yap")
    st.write("Modelin son gördüğü veriyi kullanarak ileriye dönük tahmin üretebilirsin. **Not:** 1 günden fazlası için model kendi tahminini bir sonraki adıma girdi olarak kullanır (rekürsif tahmin), bu yüzden hata birikip büyüyebilir.")

    days_ahead=st.slider("Kaç gün ileriye tahmin yapılsın?",min_value=1,max_value=30,value=5)
    predict_button=st.button("Tahmin Et")

    if predict_button:
        test_scaled=result["test_scaled"]
        train_min,train_max=result["train_min"],result["train_max"]

        current_seq=test_scaled[-seq_length_used:].clone().unsqueeze(0).to(device)
        preds_scaled=[]

        model.eval()
        with torch.no_grad():
            for _ in range(days_ahead):
                next_pred=model(current_seq)
                preds_scaled.append(next_pred.squeeze(0))
                current_seq=torch.cat([current_seq[:,1:,:],next_pred.unsqueeze(1)],dim=1)

        preds_scaled=torch.stack(preds_scaled)
        preds_real=inverse_transform(preds_scaled.cpu(),train_min.cpu(),train_max.cpu())

        pred_df=pd.DataFrame(preds_real.numpy(),columns=feature_cols)
        pred_df.insert(0,"Gün",[f"+{i+1}" for i in range(days_ahead)])

        st.dataframe(pred_df.style.format({c:"{:.2f}" for c in feature_cols}),use_container_width=True)

        fig,ax=plt.subplots(figsize=(10,4))
        last_actual=df["Close"].iloc[-30:].values
        ax.plot(range(len(last_actual)),last_actual,label="Son veriler (gerçek)")
        ax.plot(
            range(len(last_actual)-1,len(last_actual)-1+days_ahead+1),
            [last_actual[-1]]+pred_df["Close"].tolist(),
            label="Tahmin",
            linestyle="--",
            marker="o",
            color="orange",
        )
        ax.legend()
        ax.set_ylabel("Close Fiyatı")
        ax.grid(alpha=0.3)
        st.pyplot(fig)
else:
    st.info("👈 Soldaki menüden ayarları yapıp **'Modeli Eğit'** butonuna basarak başlayabilirsin.")

st.divider()
st.caption("⚠️ Bu uygulama eğitim/demo amaçlıdır. Üretilen tahminler yatırım tavsiyesi niteliği taşımaz.")
