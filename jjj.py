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

# ==================== 백엔드 & 폰트 설정 ====================
matplotlib.use('TkAgg')

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

class UltimateStockForecaster:
    def __init__(self):
        self.today = datetime.today().date()
        self.krx_list = None
        self.load_krx_list()

    def load_krx_list(self):
        try:
            self.krx_list = fdr.StockListing('KRX')[['Code', 'Name']]
            print(f"✅ KRX 종목 리스트 로드 완료 ({len(self.krx_list)}개)")
        except:
            self.krx_list = pd.DataFrame()
            print("⚠️ KRX 리스트 로드 실패")

    def find_stock(self, input_str):
        input_str = input_str.strip()
        if input_str.isdigit() and len(input_str) in (5, 6):
            code = input_str.zfill(6)
            try:
                name = self.krx_list[self.krx_list['Code'] == code]['Name'].iloc[0]
                return code, name
            except:
                return code, f"종목 {code}"
        
        if self.krx_list.empty:
            return None, None
        
        exact = self.krx_list[self.krx_list['Name'] == input_str]
        if not exact.empty:
            return exact.iloc[0]['Code'], exact.iloc[0]['Name']
        
        matches = self.krx_list[self.krx_list['Name'].str.contains(input_str, na=False)]
        if not matches.empty:
            if len(matches) > 1:
                print(f"   여러 종목 검색됨: {matches['Name'].tolist()[:5]} ...")
            return matches.iloc[0]['Code'], matches.iloc[0]['Name']
        
        return None, None

    def get_next_trading_day(self):
        candidate = self.today + timedelta(days=1)
        while candidate.weekday() >= 5:
            candidate += timedelta(days=1)
        return candidate

    def get_global_factors(self):
        print("\n🌍 세계 금융시장 요약 (최근 변화율)")
        try:
            tickers = {
                '^GSPC': 'S&P500', '^IXIC': '나스닥', '^DJI': '다우',
                'GC=F': '금', 'CL=F': '원유', '^VIX': 'VIX'
            }
            global_data = {}
            for ticker, name in tickers.items():
                data = yf.download(ticker, period='5d', progress=False)['Close']
                if len(data) >= 2:
                    chg = (data.iloc[-1] / data.iloc[-2] - 1) * 100
                    global_data[name] = round(chg, 2)
            
            for name, chg in global_data.items():
                color = "🟢" if chg > 0 else "🔴"
                print(f"   {color} {name}: {chg:+.2f}%")
            
            vix_chg = global_data.get('VIX', 0)
            risk_adj = -5 if vix_chg > 5 else 5 if vix_chg < -3 else 0
            print(f"   ⚠️ 시장 리스크 조정: {'상승' if risk_adj > 0 else '하락' if risk_adj < 0 else '중립'} 영향 예상")
            return global_data, risk_adj
        except:
            print("   ⚠️ 글로벌 데이터 가져오기 실패")
            return {}, 0

    def get_data(self, code):
        try:
            end = datetime.today().strftime('%Y-%m-%d')
            start = (datetime.today() - timedelta(days=1095)).strftime('%Y-%m-%d')  # 3년 데이터로 확대
            df = fdr.DataReader(code, start, end)
            return df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        except:
            return pd.DataFrame()

    def add_indicators(self, df):
        if len(df) < 60:
            return df
        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume']

        # 기존 지표
        df['MA5'] = close.rolling(5).mean()
        df['MA20'] = close.rolling(20).mean()
        df['MA60'] = close.rolling(60).mean()
        
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['Signal']
        
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        df['BB_Mid'] = close.rolling(20).mean()
        df['BB_Std'] = close.rolling(20).std()
        df['BB_Upper'] = df['BB_Mid'] + 2 * df['BB_Std']
        df['BB_Lower'] = df['BB_Mid'] - 2 * df['BB_Std']
        
        df['Vol_MA20'] = volume.rolling(20).mean()

        # === 추가 정밀 지표 ===
        # Stochastic Oscillator
        low14 = low.rolling(14).min()
        high14 = high.rolling(14).max()
        df['Stoch_K'] = 100 * (close - low14) / (high14 - low14)
        df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()
        
        # CCI (Commodity Channel Index)
        tp = (high + low + close) / 3
        df['CCI'] = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).std())
        
        # ADX (Trend Strength)
        plus_dm = high.diff()
        minus_dm = low.diff()
        tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        df['ADX'] = 100 * abs(plus_dm.rolling(14).mean() - minus_dm.rolling(14).mean()) / atr.rolling(14).mean()  # 간단 버전
        
        # ATR (변동성)
        df['ATR'] = atr
        
        # OBV (On-Balance Volume)
        df['OBV'] = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        df['OBV_MA'] = df['OBV'].rolling(20).mean()
        
        # ROC (Rate of Change)
        df['ROC'] = close.pct_change(10) * 100

        return df

    def detect_signals(self, df):
        if len(df) < 5:
            return []
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        signals = []

        # 골든/데드 크로스 강화
        if latest['MA5'] > latest['MA20'] and prev['MA5'] <= prev['MA20']:
            signals.append(('MA 골든크로스', '매수', 12))
        if latest['MACD_Hist'] > 0 and prev['MACD_Hist'] <= 0:
            signals.append(('MACD 골든크로스', '매수', 10))
        
        # 과매수/과매도
        if latest['RSI'] < 32:
            signals.append(('RSI 과매도', '매수', 8))
        elif latest['RSI'] > 68:
            signals.append(('RSI 과매수', '매도', -8))
        
        # Bollinger + Stochastic
        if latest['Close'] < latest['BB_Lower'] and latest['Stoch_K'] < 25:
            signals.append(('볼린저 하단 + Stoch 저점', '매수', 15))
        if latest['Close'] > latest['BB_Upper'] and latest['Stoch_K'] > 75:
            signals.append(('볼린저 상단 + Stoch 고점', '매도', -12))
        
        # Volume + OBV
        if latest['Volume'] > latest['Vol_MA20'] * 2.8 and latest['OBV'] > latest['OBV_MA']:
            signals.append(('거래량 폭발 + OBV 상승', '매수', 10))
        
        # CCI 극단
        if latest['CCI'] < -100:
            signals.append(('CCI 과매도', '매수', 7))
        elif latest['CCI'] > 100:
            signals.append(('CCI 과매수', '매도', -7))

        return signals

    def next_day_probability(self, df, global_factors, risk_adj):
        if len(df) < 120:
            return {"오전": 50, "오후": 50, "방향": "데이터 부족 - 중립", "예상변동폭": "±0원"}
        
        latest = df.iloc[-1]
        signals = self.detect_signals(df)
        
        score = 0
        # 신호 가중치 합산
        for _, _, weight in signals:
            score += weight
        
        # 모멘텀 지표
        if latest['RSI'] < 40:
            score += 12
        elif latest['RSI'] > 65:
            score -= 12
        
        if latest['MACD_Hist'] > 0:
            score += 9
        if latest['Stoch_K'] < 30 and latest['Stoch_D'] < latest['Stoch_K']:
            score += 8
        
        if latest['CCI'] < -80:
            score += 6
        
        # 최근 추세 (최근 5일 vs 20일)
        recent_return = (df['Close'].iloc[-1] / df['Close'].iloc[-6] - 1) * 100
        if recent_return > 3:
            score += 5
        elif recent_return < -3:
            score -= 6
        
        # 글로벌 팩터
        us_chg = global_factors.get('S&P500', 0) + global_factors.get('나스닥', 0)
        score += int(us_chg * 1.5)
        score += risk_adj
        
        # 변동성 페널티 (ATR 높으면 확률 보수적으로)
        volatility_penalty = min(10, int(latest.get('ATR', 0) / latest['Close'] * 1000))
        score -= volatility_penalty // 2
        
        # 최종 확률 (과도한 편향 방지)
        buy_prob = max(18, min(82, 50 + score * 0.75))  # 가중치 조정으로 과적합 완화
        
        # 오전/오후 분리 (글로벌 영향 강하게 오전 반영)
        morning_prob = round(buy_prob + (us_chg * 1.1), 1)
        afternoon_prob = round(buy_prob - 2.5, 1)
        
        direction = ("강한 상승" if buy_prob > 68 else 
                    "상승 우세" if buy_prob > 57 else 
                    "중립" if abs(buy_prob - 50) < 7 else 
                    "하락 우세" if buy_prob < 40 else "약한 하락")
        
        expected_range = round(latest['Close'] * (latest.get('ATR', latest['Close']*0.015) / latest['Close']) * 1.1, 0)
        
        return {
            "오전_매수확률": min(88, max(12, morning_prob)),
            "오후_매수확률": min(85, max(15, afternoon_prob)),
            "방향": direction,
            "예상변동폭": f"±{expected_range:,.0f}원",
            "종합스코어": round(score, 1)
        }

    def plot_next_day_forecast(self, df, name, code, prob, next_trading_day):
        if len(df) < 10:
            return
        latest_price = df.iloc[-1]['Close']
        next_date_str = next_trading_day.strftime('%Y-%m-%d')
        
        # 더 자연스러운 경로 생성 (모닝/애프터눈 + 약간의 노이즈)
        morning_boost = (prob["오전_매수확률"] - 50) * 0.014
        afternoon_boost = (prob["오후_매수확률"] - 50) * 0.019
        
        base_time = datetime.combine(next_trading_day, datetime.min.time())
        times = [base_time + timedelta(hours=h) for h in [0,9,10,11,12,13,14,15]]
        
        prices = [latest_price]
        for i in range(1, len(times)):
            if i <= 4:  # 오전
                progress = i / 4
                change = morning_boost * progress
            else:  # 오후
                change = morning_boost + afternoon_boost * ((i-4)/4)
            # 약간의 랜덤 워크 느낌 추가 (재현성을 위해 seed 고정 가능)
            noise = np.random.normal(0, latest_price * 0.003)
            new_price = prices[-1] * (1 + change/100) + noise
            prices.append(new_price)
        
        fig = plt.figure(figsize=(8, 5.5), dpi=110)
        fig.canvas.manager.set_window_title(f'{name} ({code}) - {next_date_str} 정밀 예상')
        
        ax = fig.add_subplot(111)
        ax.plot(times, prices, label='예상 가격 경로', color='#d62728', linestyle='--', 
                linewidth=3.5, marker='o', markersize=7, markerfacecolor='#ffeb3b')
        
        # 동적 신뢰 구간 (ATR 기반)
        atr_val = df.iloc[-1].get('ATR', latest_price * 0.018)
        band_width = atr_val * 1.35
        color_band = '#2ca02c' if prob["오전_매수확률"] > 58 else '#d62728'
        
        ax.fill_between(times, [p - band_width for p in prices], 
                       [p + band_width for p in prices],
                       color=color_band, alpha=0.22, label='예상 범위 (±ATR 기반)')
        
        ax.axhline(latest_price, color='gray', linestyle='-', alpha=0.85, 
                   linewidth=2, label=f'오늘 종가 {latest_price:,.0f}원')
        
        title = f'📈 {name} ({code}) {next_date_str} 정밀 예측\n'
        title += f'오전 {prob["오전_매수확률"]}% │ 오후 {prob["오후_매수확률"]}% │ {prob["방향"]} │ 변동폭 {prob["예상변동폭"]}'
        
        ax.set_title(title, fontsize=13.5, fontweight='bold', pad=25)
        ax.set_ylabel('가격 (원)', fontsize=12)
        ax.set_xlabel('시간대', fontsize=12)
        
        ax.legend(fontsize=10.5, loc='upper left')
        ax.grid(True, alpha=0.4)
        
        ax.set_xticks(times)
        ax.set_xticklabels(['종가', '09:00', '10:00', '11:00', '12:00', '13:00', '14:00', '15:00'], rotation=40)
        
        plt.tight_layout()
        plt.show()

    def run(self):
        next_trading_day = self.get_next_trading_day()
        print(f"🔥 Ultimate Stock Forecaster v2 - {next_trading_day.strftime('%Y-%m-%d')} 정밀 예측")
        print("   입력 예시: 삼성전자, 005930, 카카오 등 (쉼표로 여러 개 가능)\n")
        
        global_factors, risk_adj = self.get_global_factors()
        
        input_str = input("분석할 주식 이름 또는 코드 입력 → ").strip()
        if not input_str:
            return
        
        items = [x.strip() for x in input_str.split(',')]
        
        for item in items:
            print(f"\n{'='*100}")
            code, real_name = self.find_stock(item)
            if not code:
                print(f"   ❌ '{item}' 종목을 찾을 수 없습니다.")
                continue
            
            df = self.get_data(code)
            if df.empty:
                print(f"   ❌ {real_name} 데이터 로드 실패")
                continue
                
            df = self.add_indicators(df)
            if len(df) < 60:
                print(f"   ⚠️ {real_name} 데이터가 부족합니다.")
                continue
                
            latest = df.iloc[-1]
            print(f"✅ {real_name} ({code}) | 현재가: {latest['Close']:,.0f}원 | RSI: {latest.get('RSI',0):.1f} | CCI: {latest.get('CCI',0):.1f}")
            
            signals = self.detect_signals(df)
            prob = self.next_day_probability(df, global_factors, risk_adj)
            
            print(f"🔥 {next_trading_day.strftime('%Y-%m-%d')} 정밀 예상 → 오전 {prob['오전_매수확률']}% | 오후 {prob['오후_매수확률']}% | {prob['방향']}")
            if signals:
                print("   📢 감지된 신호:", ", ".join([f"{s[0]}" for s in signals]))
            if '종합스코어' in prob:
                print(f"   📊 종합 스코어: {prob['종합스코어']}")
            
            self.plot_next_day_forecast(df, real_name, code, prob, next_trading_day)
            print("   📊 그래프 창을 닫고 다음 종목을 분석하세요.\n")

if __name__ == "__main__":
    analyzer = UltimateStockForecaster()
    analyzer.run()