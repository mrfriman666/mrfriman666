# 🤖 Binance AI Scalping Trading Bot

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-red.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.8-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Binance](https://img.shields.io/badge/Binance-Futures-yellow.svg)](https://www.binance.com/en/futures)

Продвинутый торговый бот для фьючерсной скальпинговой торговли на Binance с использованием глубокого обучения и трех популярных скальпинговых стратегий.

## 📋 Содержание
- [Особенности](#особенности)
- [Стратегии](#стратегии)
- [Установка](#установка)
- [Быстрый старт](#быстрый-старт)
- [Конфигурация](#конфигурация)
- [Использование](#использование)
- [Архитектура](#архитектура)
- [Результаты](#результаты)
- [Лицензия](#лицензия)

## ✨ Особенности

- 🧠 **Мультивалютная AI модель** на основе Transformer/Attention
- 📊 **Три скальпинговые стратегии**: Order Flow, Volume Profile, Market Microstructure
- ⚡ **Оптимизация под GPU** (RTX 2060) с ограничением ресурсов до 75%
- 🔄 **Автоматическое переобучение** каждые 24 часа или при просадке
- 💰 **Продвинутый риск-менеджмент** с динамическими стоп-лоссами
- 📈 **Бэктестинг** с расчетом всех метрик
- 🎯 **Paper trading** для безопасного тестирования
- 🔧 **Auto-оптимизация** гиперпараметров через Optuna
- 📝 **Полное логирование** всех действий

## 🎯 Стратегии

### 1. Order Flow Strategy
- Анализ дисбаланса Bid/Ask
- Кумулятивная дельта объема
- Паттерны абсорбции

### 2. Volume Profile Strategy
- Point of Control (POC)
- Value Area High/Low
- High Volume Nodes (HVN) и Low Volume Nodes (LVN)

### 3. Market Microstructure Strategy
- Volume-Synchronized Probability of Informed Trading (VPIN)
- Эффективный спред
- Adverse selection анализ

## 🚀 Установка

### Предварительные требования
- Python 3.8 или выше
- NVIDIA GPU с CUDA 11.8 (рекомендуется RTX 2060+)
- 8GB+ RAM
- 20GB+ свободного места

### Windows
```bash
# Клонирование репозитория
git clone https://github.com/yourusername/binance-scalping-bot.git
cd binance-scalping-bot

# Запуск установки
install.bat

# Настройка API ключей
notepad .env

# Запуск бота
python main.py╔══════════════════════════════════════════════════════════╗
║     Binance AI Scalping Trading Bot - Main Menu         ║
╠══════════════════════════════════════════════════════════╣
║  1. 📊 Collect & Prepare Data                          ║
║  2. 🤖 Train New Model                                 ║
║  3. 📈 Run Backtest                                    ║
║  4. 💹 Start Live Trading                              ║
║  5. ⚙️  Auto-optimize Parameters                       ║
║  6. 🔄 Retrain Model                                   ║
║  7. 📋 View Logs                                       ║
║  8. ⚡ System Settings                                 ║
║  9. 🚪 Exit                                            ║
╚══════════════════════════════════════════════════════════╝
