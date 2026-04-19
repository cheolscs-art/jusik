import FinanceDataReader as fdr
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import yfinance as yf
import platform
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# ==================== matplotlib 설정 ====================
try:
    matplotlib.use('TkAgg')
except Exception:
    matplotlib.use('Agg')

def set_korean_font():
    system = platform.system()
    if system == 'Windows':
        plt.rcParams['font.family'] = 'Malgun Gothic'
    elif system == 'Darwin':
        plt.rcParams['font.family'] = 'AppleGothic'
    else:
        plt.rcParams['font.family'] = 'NanumGothic'
    plt.rcParams['axes.unicode_minus'] = False

set_korean_font()

# ==================== 메인 클래스 ====================
class UltimateStockForecaster:
    def __init__(self):
        self.today = datetime.today().date()
        self.krx_list = None
        self.load_krx_list()

    # ---------------- KRX 리스트 ----------------
    def load_krx_list(self):
        try:
            self.krx_list = fdr.StockListing('KRX')[['Code', 'Name']]
            if self.krx_list.empty:
                raise ValueError("Empty list")
            print(f"✅ KRX 종목 리스트 로드 완료 ({len(self.krx_list)}개)")
        except Exception as e:
            print(f"⚠️ KRX 리스트 로드 실패: {e}")
            self.krx_list = pd.DataFrame(columns=['Code', 'Name'])

    # ---------------- 종목 찾기 ----------------
    def find_stock(self, input_str):
        input_str = input_str.strip()

        if input_str.isdigit():
            code = input_str.zfill(6)
            result = self.krx_list[self.krx_list['Code'] == code]
            if not result.empty:
                name = result.iloc[0]['Name']
            else:
                name = f"종목 {code}"
            return code, name

        if self.krx_list.empty:
            return None, None

        exact = self.krx_list[self.krx_list['Name'] == input_str]
        if not exact.empty:
            return exact.iloc[0]['Code'], exact.iloc[0]['Name']

        matches = self.krx_list[self.krx_list['Name'].str.contains(input_str, na=False)]
        if not matches.empty:
            return matches.iloc[0]['Code'], matches.iloc[0]['Name']

        return None, None

    # ---------------- 다음 거래일 ----------------
    def get_next_trading_day(self):
        candidate = self.today + timedelta(days=1)
        while candidate.weekday() >= 5:
            candidate += timedelta(days=1)
        return candidate

    # ---------------- 글로벌 지표 ----------------
    def get_global_factors(self):
        print("\n🌍 글로벌 시장")
        tickers = {'^GSPC': 'S&P500', '^IXIC': '나스닥', '^DJI': '다우',
                   'GC=F': '금', 'CL=F': '원유', '^VIX': 'VIX'}

        global_data = {}
        failed = []

        for ticker, name in tickers.items():
            try:
                data = yf.download(ticker, period='5d', progress=False)

                if data.empty or 'Close' not in data:
                    failed.append(name)
                    continue

                close = data['Close']
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]

                if len(close) >= 2:
                    last_val = float(close.iloc[-1])
                    prev_val = float(close.iloc[-2])
                    chg = ((last_val / prev_val) - 1) * 100
                    global_data[name] = round(chg, 2)
                else:
                    failed.append(name)

            except Exception:
                failed.append(name)

        for name, chg in global_data.items():
            color = "🟢" if chg > 0 else "🔴"
            print(f"{color} {name}: {chg:+.2f}%")

        if failed:
            print(f"⚠️ 실패: {', '.join(failed)}")

        vix_chg = global_data.get('VIX', 0)
        risk_adj = -6 if vix_chg > 6 else 6 if vix_chg < -4 else 0

        return global_data, risk_adj

    # ---------------- 주가 데이터 ----------------
    def get_data(self, code):
        try:
            end = datetime.today().strftime('%Y-%m-%d')
            start = (datetime.today() - timedelta(days=1095)).strftime('%Y-%m-%d')

            df = fdr.DataReader(code, start, end)

            if df is None or df.empty:
                print(f"⚠️ 데이터 없음: {code}")
                return pd.DataFrame()

            required = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in df.columns for col in required):
                print(f"⚠️ 컬럼 부족: {code}")
                return pd.DataFrame()

            return df[required].copy()

        except Exception:
            return pd.DataFrame()

    # ---------------- 지표 추가 ----------------
    def add_indicators(self, df):
        if len(df) < 60:
            return df

        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume']

        df['MA5'] = close.rolling(5).mean()
        df['MA20'] = close.rolling(20).mean()

        # MACD
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        df['MACD'] = ema12 - ema26
        df['Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Hist'] = df['MACD'] - df['Signal']

        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        loss = loss.replace(0, np.nan)
        rs = gain / loss
        df['RSI'] = (100 - (100 / (1 + rs))).fillna(50)

        # 볼린저
        df['BB_Mid'] = close.rolling(20).mean()
        std = close.rolling(20).std()
        df['BB_Upper'] = df['BB_Mid'] + 2 * std
        df['BB_Lower'] = df['BB_Mid'] - 2 * std

        # 거래량
        df['Vol_MA20'] = volume.rolling(20).mean()

        # Stochastic
        low14 = low.rolling(14).min()
        high14 = high.rolling(14).max()
        denom = (high14 - low14).replace(0, np.nan)
        df['Stoch_K'] = (100 * (close - low14) / denom).fillna(50)

        # ATR
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(14).mean()

        return df.ffill().fillna(0)

    # ---------------- 시그널 ----------------
    def detect_signals(self, df):
        latest = df.iloc[-1]
        prev = df.iloc[-2]

        signals = []

        if latest['MA5'] > latest['MA20'] and prev['MA5'] <= prev['MA20']:
            signals.append(12)

        if latest['MACD_Hist'] > 0 and prev['MACD_Hist'] <= 0:
            signals.append(10)

        if latest['RSI'] < 30:
            signals.append(8)
        elif latest['RSI'] > 70:
            signals.append(-8)

        return sum(signals)

    # ---------------- 확률 계산 ----------------
    def next_day_probability(self, df, global_factors, risk_adj):
        latest = df.iloc[-1]

        score = self.detect_signals(df)

        us = global_factors.get('S&P500', 0) + global_factors.get('나스닥', 0)
        score += int(us * 1.5) + risk_adj

        buy_prob = max(20, min(80, 50 + score))

        return {
            "오전": round(buy_prob + us, 1),
            "오후": round(buy_prob - 2, 1)
        }

    # ---------------- 실행 ----------------
    def run(self):
        next_day = self.get_next_trading_day()
        print(f"\n🔥 {next_day} 예측")

        global_factors, risk_adj = self.get_global_factors()

        input_str = input("\n종목 입력 → ").strip()
        if not input_str:
            print("❌ 입력 없음")
            return

        for item in input_str.split(','):
            code, name = self.find_stock(item.strip())

            if not code:
                print(f"❌ {item} 없음")
                continue

            df = self.get_data(code)
            if df.empty:
                continue

            df = self.add_indicators(df)

            prob = self.next_day_probability(df, global_factors, risk_adj)

            print(f"\n📊 {name} ({code})")
            print(f"➡️ 오전: {prob['오전']}% / 오후: {prob['오후']}%")

# ==================== 실행 ====================
if __name__ == "__main__":
    UltimateStockForecaster().run()