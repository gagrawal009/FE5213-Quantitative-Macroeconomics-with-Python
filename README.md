# Quantitative Macroeconomics with Python

This repository contains two quantitative research projects completed as part of the MSc Financial Engineering coursework at the National University of Singapore.

## 📁 Part 1: Asset Return Forecasting using Goyal-Welch Data

**Objective:**  
Predict excess returns of SPY, TLT, and GLD ETFs using 14 macroeconomic indicators from the Goyal-Welch dataset.

**Methods:**  
- OLS and Ridge Regression with Fourier-transformed features  
- Deep Neural Networks for non-linear prediction  
- Model tuning via Sharpe Ratio optimization and regularization

**Key Results:**  
- Ridge and deep learning models outperformed baselines  
- Best Sharpe ratio: **1.03** on SPY with deep learning  
- Analysis of signal importance and normalization impact

## 📁 Part 2: Inflation Forecasting – US and UK

**Objective:**  
Forecast quarterly inflation (CPI) for the US and UK from 1988 to 2025 using time series and machine learning models.

**Models Used:**  
- Autoregressive (AR)  
- Autoregressive Distributed Lag (ADL)  
- XGBoost with lagged features

**Key Findings:**  
- AR(3) for US and AR(5) for UK delivered lowest RMSE  
- Multi-step forecasts closely matched Fed and Statista projections for 2025  
- XGBoost performed well in stable regimes but struggled with recent volatility


## 📊 Authors

- Sai Kiran Reddy Poreddy  
- **Gaurav Agrawal**  
- Raditya  
- Jiya Dutta  
- Choo Jin Yi  

## 🏫 Course

FE5213 Quantitative Macroeconomics & Finance  
National University of Singapore – 2025

---



