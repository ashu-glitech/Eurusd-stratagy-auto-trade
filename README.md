# 🎯 6EZ6 Live Vacuum Sniper | CME Euro FX Engine

Institutional-grade High-Frequency Trading (HFT) and orderbook microstructure engine for CME Euro FX (`6EZ6`) and Spot EUR/USD.

## 🚀 System Architecture
- **Live 10-Level Orderbook Microstructure**: Real-time Level 2 depth streaming via Tradovate WebSocket.
- **Dual-Engine LightGBM Models**: Call and Put barrier breakout prediction models.
- **Dynamic Vacuum Engine**: Classifies market into Neutral Chop, Medium Vacuum, and Big Vacuum.
- **Orderbook Wall Absorption & Highway Scanner**: Real-time ratio analysis of near depth (L1-L3) vs deep liquidity (L4-L10).
- **Automated MetaTrader 5 Bridge**: `CME_Sniper_Bridge.mq5` for zero-latency execution on Spot EUR/USD (Exness / any broker).

## 📊 Live Dashboard
- Real-time WebSockets streaming orderbook ladder, spread, vacuum score, runner trailing floors, and live PnL.
- Run via:
  ```bash
  python3 live_dashboard/server.py
  ```
  Open browser at: `http://localhost:8000`
