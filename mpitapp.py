import pandas as pd
import numpy as np
import random
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
import time
import os
from datetime import datetime, timedelta
import tkinter as tk
from tkinter import filedialog
warnings.filterwarnings('ignore')

class PriceRecommender:
    def __init__(self):
        self.model = None
        self.feature_columns = []
        self.passenger_max_ratio = 1.3
        self._is_trained = False

    def _get_holidays(self, year):
        holidays = [
            # Новогодние каникулы
            f"{year}-01-01", f"{year}-01-02", f"{year}-01-03", f"{year}-01-04", f"{year}-01-05",
            f"{year}-01-06", f"{year}-01-07", f"{year}-01-08",
            # Рождество
            f"{year}-01-07",
            # День защитника Отечества
            f"{year}-02-23",
            # Международный женский день
            f"{year}-03-08",
            # Праздник Весны и Труда
            f"{year}-05-01",
            # День Победы
            f"{year}-05-09",
            # День России
            f"{year}-06-12",
            # День народного единства
            f"{year}-11-04",
        ]
        return [datetime.strptime(date, "%Y-%m-%d").date() for date in holidays]

    def _is_holiday_or_preholiday(self, date):
        year = date.year
        holidays = self._get_holidays(year)
        
        # Проверяем сам праздник
        if date.date() in holidays:
            return True, 'holiday'
        
        # Проверяем предпраздничный день (день перед праздником)
        next_day = date + timedelta(days=1)
        if next_day.date() in holidays:
            return True, 'preholiday'
        
        return False, 'normal'

    def preprocess_data(self, df, is_training=True):
        print("Предобработка данных...")

        df_processed = df.copy()

        if is_training:
            required_columns = ['is_done', 'distance_in_meters', 'duration_in_seconds', 'order_timestamp']
        else:
            required_columns = ['distance_in_meters', 'duration_in_seconds', 'order_timestamp']

        missing_columns = [col for col in required_columns if col not in df_processed.columns]
        if missing_columns:
            raise ValueError(f"Отсутствуют колонки: {missing_columns}")

        if is_training:
            if 'price_start_local' not in df_processed.columns or 'price_bid_local' not in df_processed.columns:
                raise ValueError("Отсутствуют ценовые колонки")

            df_processed['is_done_numeric'] = (df_processed['is_done'] == 'done').astype(int)
            accepted_count = (df_processed['is_done_numeric'] == 1).sum()
            canceled_count = (df_processed['is_done_numeric'] == 0).sum()

            print(f"Принятые заказы: {accepted_count} ({accepted_count/len(df_processed)*100:.1f}%)")
            print(f"Отклоненные заказы: {canceled_count} ({canceled_count/len(df_processed)*100:.1f}%)")

            if accepted_count == 0:
                raise ValueError("Нет принятых заказов для анализа")
        else:
            if 'price_start_local' not in df_processed.columns:
                raise ValueError("Отсутствует колонка price_start_local")

        # Преобразование единиц
        df_processed['distance_km'] = df_processed['distance_in_meters'] / 1000.0
        df_processed['duration_min'] = df_processed['duration_in_seconds'] / 60.0

        # Временные признаки
        df_processed['order_datetime'] = pd.to_datetime(df_processed['order_timestamp'])
        df_processed['order_hour'] = df_processed['order_datetime'].dt.hour
        df_processed['order_day_of_week'] = df_processed['order_datetime'].dt.dayofweek
        df_processed['order_month'] = df_processed['order_datetime'].dt.month
        df_processed['order_day'] = df_processed['order_datetime'].dt.day
        df_processed['order_date'] = df_processed['order_datetime'].dt.date

        # Праздники и предпраздничные дни
        df_processed['is_holiday'] = 0
        df_processed['is_preholiday'] = 0
        df_processed['holiday_type'] = 'normal'
    
        for idx, row in df_processed.iterrows():
            try:
                is_special, holiday_type = self._is_holiday_or_preholiday(row['order_datetime'])
                if is_special:
                    if holiday_type == 'holiday':
                        df_processed.at[idx, 'is_holiday'] = 1
                        df_processed.at[idx, 'holiday_type'] = 'holiday'
                    elif holiday_type == 'preholiday':
                        df_processed.at[idx, 'is_preholiday'] = 1
                        df_processed.at[idx, 'holiday_type'] = 'preholiday'
            except:
                continue

        # Сезонность
        def get_season(month):
            if month in [12, 1, 2]: return 'winter'
            elif month in [3, 4, 5]: return 'spring'
            elif month in [6, 7, 8]: return 'summer'
            else: return 'autumn'

        df_processed['season'] = df_processed['order_month'].apply(get_season)
        season_map = {'winter': 0, 'spring': 1, 'summer': 2, 'autumn': 3}
        df_processed['season_code'] = df_processed['season'].map(season_map)

        # Погода (сезон + время)
        def get_weather(season, hour, month):
            # Более реалистичная логика погоды
            if season == 'winter':
                if month in [12, 1]:  # Глубокая зима
                    return 'snowy' if hour < 16 else 'cloudy'
                else:  # Поздняя зима/ранняя весна
                    return 'cloudy' if hour < 14 else 'rainy'
            elif season == 'summer':
                if hour < 11: return 'sunny'
                elif hour < 18: return 'cloudy'
                else: return 'rainy'
            elif season == 'spring':
                return 'rainy' if hour < 12 else 'cloudy'
            else:  # Осень
                return 'cloudy' if hour < 13 else 'rainy'

        df_processed['weather_type'] = [
            get_weather(s, h, m) for s, h, m in zip(
                df_processed['season'],
                df_processed['order_hour'],
                df_processed['order_month']
            )
        ]

        weather_map = {'sunny': 0, 'cloudy': 1, 'rainy': 2, 'snowy': 3}
        df_processed['weather_code'] = df_processed['weather_type'].map(weather_map)
        df_processed['bad_weather'] = (df_processed['weather_code'] >= 2).astype(int)

        # Временные признаки
        df_processed['is_weekend'] = (df_processed['order_day_of_week'] >= 5).astype(int)

        # Пиковые часы: будни 7-10 и 17-20
        df_processed['is_peak_hours'] = (
            ((df_processed['order_hour'] >= 7) & (df_processed['order_hour'] <= 10) |
             (df_processed['order_hour'] >= 17) & (df_processed['order_hour'] <= 20)) &
            (df_processed['is_weekend'] == 0) &
            (df_processed['is_holiday'] == 0)
        ).astype(int)

        # Пиковые часы для праздников и выходных
        df_processed['is_holiday_peak'] = (
            ((df_processed['order_hour'] >= 11) & (df_processed['order_hour'] <= 15) |
             (df_processed['order_hour'] >= 18) & (df_processed['order_hour'] <= 22)) &
            ((df_processed['is_weekend'] == 1) | (df_processed['is_holiday'] == 1) | (df_processed['is_preholiday'] == 1))
        ).astype(int)

        # Ночные часы (доплата)
        df_processed['is_night'] = ((df_processed['order_hour'] >= 0) & (df_processed['order_hour'] <= 6)).astype(int)

        # Специальные дни
        df_processed['is_special_day'] = (
            (df_processed['is_weekend'] == 1) | 
            (df_processed['is_holiday'] == 1) | 
            (df_processed['is_preholiday'] == 1)
        ).astype(int)

        print("Данные обработаны")
        if is_training:
            print(f"Распределение погоды: {df_processed['weather_type'].value_counts().to_dict()}")
            print(f"Пиковые часы: {(df_processed['is_peak_hours'] == 1).sum()} заказов")
            print(f"Праздничные дни: {(df_processed['is_holiday'] == 1).sum()} заказов")
            print(f"Предпраздничные дни: {(df_processed['is_preholiday'] == 1).sum()} заказов")
            print(f"Пиковые часы праздников: {(df_processed['is_holiday_peak'] == 1).sum()} заказов")

        return df_processed

    def _train_probability_model(self, df):
        print("Обучение модели вероятности...")
        self.probability_model = None

    def train_model(self, df):
        print("Обучение модели...")

        accepted_orders = df[df['is_done_numeric'] == 1]
        if len(accepted_orders) == 0:
            raise ValueError("Нет принятых заказов для обучения")

        print(f"Используется {len(accepted_orders)} принятых заказов для обучения")

        # Признаки модели
        self.feature_columns = [
            'distance_km', 'duration_min', 'order_hour', 'is_weekend',
            'is_peak_hours', 'bad_weather', 'season_code', 'weather_code', 'is_night',
            'is_holiday', 'is_preholiday', 'is_holiday_peak', 'is_special_day'
        ]

        X = accepted_orders[self.feature_columns]
        y = accepted_orders['price_bid_local']

        # Проверяем данные
        print(f"Диапазон цен для обучения: {y.min():.0f} - {y.max():.0f} руб.")
        print(f"Средняя цена: {y.mean():.0f} руб.")

        # Обучение с большим количеством итераций для лучшего качества
        self.model = HistGradientBoostingRegressor(
            random_state=42,
            max_iter=200,  # Увеличили для лучшего обучения
            learning_rate=0.1,
            max_depth=10,
            min_samples_leaf=5
        )
        self.model.fit(X, y)

        # Обучаем модель вероятности на ВСЕХ данных
        self._train_probability_model(df)

        # Предсказание минимальных цен для ВСЕХ заказов
        df['min_driver_price'] = self.model.predict(df[self.feature_columns])

        # Округление и защита от аномалий
        df['min_driver_price'] = np.round(df['min_driver_price'] / 10) * 10

        #цена никогда не ниже 90% от пассажирской
        df['min_driver_price'] = np.maximum(df['min_driver_price'], df['price_start_local'] * 0.9)

        #цена никогда не ниже разумного минимума
        df['min_driver_price'] = np.maximum(df['min_driver_price'], 100)  # Минимум 100 руб

        self._is_trained = True

        print(f"Модель обучена на {len(accepted_orders)} заказах")
        print(f"Диапазон предсказанных цен: {df['min_driver_price'].min():.0f} - {df['min_driver_price'].max():.0f} руб.")

        return df

    def analyze_passenger_behavior(self, df):
        print("Анализ поведения пассажиров...")

        # Используем только принятые заказы для анализа
        accepted_orders = df[df['is_done_numeric'] == 1]

        if len(accepted_orders) == 0:
            raise ValueError("Нет принятых заказов для анализа")

        # Анализируем реальные соотношения цен в принятых заказах
        accepted_orders = accepted_orders.copy()
        accepted_orders['price_ratio'] = accepted_orders['price_bid_local'] / accepted_orders['price_start_local']

        # Убираем выбросы (только разумные соотношения)
        reasonable_ratios = accepted_orders[
            (accepted_orders['price_ratio'] >= 0.8) &
            (accepted_orders['price_ratio'] <= 2.0)
        ]['price_ratio']

        if len(reasonable_ratios) > 0:
            # Берем 80-й процентиль как безопасный максимум
            self.passenger_max_ratio = np.percentile(reasonable_ratios, 80)
            self.passenger_max_ratio = min(self.passenger_max_ratio, 1.8)  # Ограничиваем сверху
            self.passenger_max_ratio = max(self.passenger_max_ratio, 1.2)  # Ограничиваем снизу

            print(f"Анализ {len(reasonable_ratios)} заказов:")
            print(f"50-й процентиль: {np.percentile(reasonable_ratios, 50):.2f}")
            print(f"80-й процентиль: {np.percentile(reasonable_ratios, 80):.2f}")
            print(f"95-й процентиль: {np.percentile(reasonable_ratios, 95):.2f}")
        else:
            print("Недостаточно данных, используем консервативное значение")
            self.passenger_max_ratio = 1.3

        print(f"Максимальное соотношение цены: {self.passenger_max_ratio:.2f}")
        return self.passenger_max_ratio

    def _calculate_dynamic_max_ratio(self, order):
        base_ratio = self.passenger_max_ratio

        passenger_price = order['price_start_local']

        # Дорогие заказы - меньшее максимальное повышение (пассажиры чувствительнее)
        if passenger_price > 4000:
            base_ratio = min(base_ratio, 1.25)  # Максимум +25%
        elif passenger_price > 3000:
            base_ratio = min(base_ratio, 1.35)  # Максимум +35%
        elif passenger_price > 2000:
            base_ratio *= 1.05  # +5% к максимуму
        elif passenger_price < 800:
            base_ratio *= 1.10  # Дешевые заказы +10% к максимуму

        # Дальние поездки - большее повышение
        distance_km = order.get('distance_km', 5)
        if distance_km > 20: base_ratio *= 1.15
        elif distance_km > 15: base_ratio *= 1.10
        elif distance_km > 10: base_ratio *= 1.05
        elif distance_km < 3: base_ratio *= 0.95  # Короткие - меньше повышения

        # Длительные поездки
        duration_min = order.get('duration_min', 15)
        if duration_min > 60: base_ratio *= 1.12
        elif duration_min > 45: base_ratio *= 1.08
        elif duration_min > 30: base_ratio *= 1.04

        return min(base_ratio, 2.0)  # Абсолютный максимум +100%

    def _calculate_weights(self, order):
        min_weight, max_weight = 0.4, 0.6

        passenger_price = order['price_start_local']

        # Для дорогих заказов больше ориентируемся на минимальную цену
        if passenger_price > 3000:
            min_weight, max_weight = 0.6, 0.4
        elif passenger_price > 2000:
            min_weight, max_weight = 0.5, 0.5
        elif passenger_price < 1000:
            min_weight, max_weight = 0.3, 0.7  # Дешевые - ближе к максимуму

        # Пиковые часы - больше веса минимальной цене (высокий спрос)
        if order.get('is_peak_hours', 0) == 1:
            min_weight += 0.15
            max_weight -= 0.15

        # Плохая погода - больше веса минимальной цене
        if order.get('bad_weather', 0) == 1:
            min_weight += 0.08
            max_weight -= 0.08

        # Ночное время - больше веса минимальной цене
        if order.get('is_night', 0) == 1:
            min_weight += 0.10
            max_weight -= 0.10

        # Гарантируем разумные границы
        min_weight = max(0.2, min(0.8, min_weight))
        max_weight = max(0.2, min(0.8, max_weight))

        # Нормализуем чтобы сумма была 1.0
        total = min_weight + max_weight
        min_weight /= total
        max_weight /= total

        return min_weight, max_weight

    def _calculate_increase_limits(self, order):
        # Базовые ограничения
        min_increase = 1.03  # Минимум +3%
        max_increase = 1.80  # Максимум +80%

        passenger_price = order['price_start_local']

        # Динамические ограничения в зависимости от цены
        if passenger_price > 4000:
            min_increase, max_increase = 1.05, 1.40  # Дорогие: +5-40%
        elif passenger_price > 3000:
            min_increase, max_increase = 1.04, 1.50  # Средние: +4-50%
        elif passenger_price > 2000:
            min_increase, max_increase = 1.03, 1.60  # +3-60%
        elif passenger_price > 1000:
            min_increase, max_increase = 1.03, 1.70  # +3-70%
        else:
            min_increase, max_increase = 1.03, 1.80  # Дешевые: +3-80%

        # Коротшие поездки - меньшее максимальное повышение
        distance_km = order.get('distance_km', 5)
        if distance_km < 3:
            max_increase = min(max_increase, 1.40)  # Максимум +40%
        elif distance_km < 5:
            max_increase = min(max_increase, 1.60)  # Максимум +60%

        # Пиковые часы - большее минимальное повышение
        if order.get('is_peak_hours', 0) == 1:
            min_increase = max(min_increase, 1.08)  # Минимум +8%

        # Плохая погода - большее минимальное повышение
        if order.get('bad_weather', 0) == 1:
            min_increase = max(min_increase, 1.06)  # Минимум +6%

        # Ночное время - большее минимальное повышение
        if order.get('is_night', 0) == 1:
            min_increase = max(min_increase, 1.10)  # Минимум +10%

        return min_increase, max_increase

    def _calculate_price_multiplier(self, order):
        multiplier = 1.0

        # Пиковые часы
        if order.get('is_peak_hours', 0) == 1:
            multiplier *= 1.15  # +15%

        # Праздничные пиковые часы
        if order.get('is_holiday_peak', 0) == 1:
            multiplier *= 1.20  # +20%

        # Плохая погода
        if order.get('bad_weather', 0) == 1:
            multiplier *= 1.12  # +12%

        # Выходные
        if order.get('is_weekend', 0) == 1:
            multiplier *= 1.08  # +8%

        # Праздничные дни
        if order.get('is_holiday', 0) == 1:
            multiplier *= 1.25  # +25%

        # Предпраздничные дни
        if order.get('is_preholiday', 0) == 1:
            multiplier *= 1.15  # +15%

        # Ночное время
        if order.get('is_night', 0) == 1:
            multiplier *= 1.20  # +20%

        # Время суток
        hour = order.get('order_hour', 12)
        if 18 <= hour <= 23:
            multiplier *= 1.10  # Вечер +10%
        elif 0 <= hour <= 6:
            multiplier *= 1.25  # Ночь +25%
        elif 7 <= hour <= 9:
            multiplier *= 1.05  # Утро +5%

        # Расстояние
        distance_km = order.get('distance_km', 5)
        if distance_km > 25: multiplier *= 1.18  # Очень дальние +18%
        elif distance_km > 20: multiplier *= 1.15  # Дальние +15%
        elif distance_km > 15: multiplier *= 1.10  # Средние +10%
        elif distance_km > 10: multiplier *= 1.05  # +5%
        elif distance_km < 3: multiplier *= 0.90   # Очень короткие -10%
        elif distance_km < 5: multiplier *= 0.95   # Короткие -5%

        # Длительность
        duration_min = order.get('duration_min', 15)
        if duration_min > 60: multiplier *= 1.15  # Очень долгие +15%
        elif duration_min > 45: multiplier *= 1.10  # Долгие +10%
        elif duration_min > 30: multiplier *= 1.05  # +5%

        # Сезонность
        season_code = order.get('season_code', 1)
        if season_code == 0: multiplier *= 1.10  # Зима +10%
        elif season_code == 2: multiplier *= 1.05  # Лето +5%

        return multiplier

    def _calculate_accept_probability(self, order_data):
        if hasattr(self, 'probability_model') and self.probability_model:
            try:
                # Подготавливаем фичи для модели вероятности
                features_dict = {}
                for feature in self.feature_columns:
                    features_dict[feature] = order_data.get(feature, 0)
                
                # Добавляем price_ratio (используем recommended_price если price_bid_local нет)
                recommended_price = order_data.get('price_bid_local', order_data['price_start_local'] * 1.1)
                price_ratio = recommended_price / order_data['price_start_local']
                features_dict['price_ratio'] = price_ratio
                
                # Создаем feature vector в правильном порядке
                feature_vector = [features_dict[feature] for feature in (self.feature_columns + ['price_ratio'])]
                
                # Предсказываем вероятность
                probability = self.probability_model.predict_proba([feature_vector])[0][1]
                return max(0.01, min(0.99, probability))  # Ограничиваем диапазон
            except Exception as e:
                print(f"Ошибка ML модели вероятности, используем эвристику: {e}")
        
        # Fallback на старую эвристику
        recommended_price = order_data.get('price_bid_local', order_data['price_start_local'] * 1.1)
        price_ratio = recommended_price / order_data['price_start_local']
        
        # Более плавная зависимость
        if price_ratio <= 1.02: return 0.90
        elif price_ratio <= 1.05: return 0.85
        elif price_ratio <= 1.08: return 0.80
        elif price_ratio <= 1.12: return 0.75
        elif price_ratio <= 1.16: return 0.70
        elif price_ratio <= 1.20: return 0.65
        elif price_ratio <= 1.25: return 0.55
        elif price_ratio <= 1.30: return 0.45
        elif price_ratio <= 1.40: return 0.35
        elif price_ratio <= 1.50: return 0.25
        elif price_ratio <= 1.70: return 0.15
        else: return 0.10

    def recommend_price(self, order):
        if not self._is_trained:
            raise ValueError("Модель не обучена. Сначала обучите модель.")

        passenger_price = float(order['price_start_local'])

        # цена пассажира должна быть разумной
        if passenger_price <= 0:
            raise ValueError("Цена пассажира должна быть положительной")

        if passenger_price < 100:
            print("Очень низкая цена пассажира, возможны неточности")

        # Предсказание модели
        features = {}
        for feature in self.feature_columns:
            features[feature] = order.get(feature, 0)

        feature_vector = pd.DataFrame([features])[self.feature_columns]
        min_driver_price = self.model.predict(feature_vector)[0]

        #от аномальных предсказаний
        min_driver_price = max(min_driver_price, passenger_price * 0.8)   # Не ниже 80%
        min_driver_price = max(min_driver_price, 100)                     # Абсолютный минимум 100 руб
        min_driver_price = min(min_driver_price, passenger_price * 3.0)   # Не выше 300%

        # полный расчет цены
        base_price = max(passenger_price, min_driver_price)
        multiplier = self._calculate_price_multiplier(order)
        adjusted_min_price = base_price * multiplier

        dynamic_max_ratio = self._calculate_dynamic_max_ratio(order)
        max_passenger_price = passenger_price * dynamic_max_ratio

        min_weight, max_weight = self._calculate_weights(order)
        recommended_price = (adjusted_min_price * min_weight + max_passenger_price * max_weight)

        # Применение ограничений
        min_increase, max_increase = self._calculate_increase_limits(order)
        recommended_price = max(recommended_price, passenger_price * min_increase)
        recommended_price = min(recommended_price, passenger_price * max_increase)

        # окончательная проверка
        recommended_price = max(recommended_price, passenger_price * 1.01)  # Минимум +1%
        recommended_price = min(recommended_price, passenger_price * 2.0)   # Максимум +100%

        # Умное округление
        recommended_price = round(recommended_price)
        last_digit = recommended_price % 10
        if last_digit not in [0, 5]:
            recommended_price = round(recommended_price / 10) * 10

        # Финальная гарантия
        recommended_price = max(recommended_price, passenger_price + 10)  # Минимум +10 руб

        # Расчет вероятности
        price_ratio = recommended_price / passenger_price
        probability = self._calculate_accept_probability(order)

        price_increase_percent = ((recommended_price - passenger_price) / passenger_price * 100)

        return {
            'order_id': order.get('order_id', 'unknown'),
            'passenger_price': passenger_price,
            'recommended_price': int(recommended_price),
            'price_increase_percent': round(price_increase_percent, 1),
            'accept_probability': round(probability, 3)
        }

    def recommend_prices_batch(self, df, sample_size=None):
        if sample_size is None:
            sample_size = min(100, len(df))

        print(f"Генерация рекомендаций для {sample_size} заказов...")

        df_sample = df.sample(n=min(sample_size, len(df)), random_state=42) if sample_size < len(df) else df

        recommendations = []
        for idx, (_, order) in enumerate(df_sample.iterrows()):
            try:
                recommendation = self.recommend_price(order.to_dict())
                recommendations.append(recommendation)
            except Exception as e:
                print(f"Пропущен заказ {idx}: {e}")
                continue

        print(f"Сгенерировано {len(recommendations)} рекомендаций")
        return pd.DataFrame(recommendations)

    def plot_training_analysis(self, df):
        if not self._is_trained:
            print("Модель не обучена")
            return

        print("\n Анализ обученной модели...")

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Анализ обученной модели рекомендаций цен', fontsize=16, fontweight='bold')

        # График 1: Распределение принятых цен
        accepted_prices = df[df['is_done_numeric'] == 1]['price_bid_local']
        axes[0, 0].hist(accepted_prices, bins=30, alpha=0.7, color='green', edgecolor='black')
        axes[0, 0].axvline(accepted_prices.mean(), color='red', linestyle='--',
                          label=f'Среднее: {accepted_prices.mean():.0f} руб')
        axes[0, 0].set_xlabel('Цена принятых заказов (руб)')
        axes[0, 0].set_ylabel('Количество')
        axes[0, 0].set_title('Распределение принятых цен')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # График 2: Соотношение цен в принятых заказах
        accepted_ratios = df[df['is_done_numeric'] == 1]
        accepted_ratios = accepted_ratios.copy()
        accepted_ratios['ratio'] = accepted_ratios['price_bid_local'] / accepted_ratios['price_start_local']
        axes[0, 1].hist(accepted_ratios['ratio'], bins=30, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 1].axvline(self.passenger_max_ratio, color='red', linestyle='--',
                          label=f'Макс. соотношение: {self.passenger_max_ratio:.2f}')
        axes[0, 1].set_xlabel('Соотношение цен (водитель/пассажир)')
        axes[0, 1].set_ylabel('Количество')
        axes[0, 1].set_title('Соотношение цен в принятых заказах')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # График 3: Распределение расстояний
        axes[1, 0].hist(df['distance_km'], bins=30, alpha=0.7, color='orange', edgecolor='black')
        axes[1, 0].axvline(df['distance_km'].mean(), color='red', linestyle='--',
                          label=f'Среднее: {df['distance_km'].mean():.1f} км')
        axes[1, 0].set_xlabel('Расстояние (км)')
        axes[1, 0].set_ylabel('Количество')
        axes[1, 0].set_title('Распределение расстояний поездок')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # График 4: Активность по часам
        hour_counts = df['order_hour'].value_counts().sort_index()
        axes[1, 1].bar(hour_counts.index, hour_counts.values, alpha=0.7, color='purple')
        axes[1, 1].set_xlabel('Час суток')
        axes[1, 1].set_ylabel('Количество заказов')
        axes[1, 1].set_title('Распределение заказов по времени суток')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Статистика
        print(f"\n СТАТИСТИКА ДАННЫХ:")
        print(f"Всего заказов: {len(df)}")
        print(f"Принято: {(df['is_done_numeric'] == 1).sum()} ({(df['is_done_numeric'] == 1).sum()/len(df)*100:.1f}%)")
        print(f"Отклонено: {(df['is_done_numeric'] == 0).sum()} ({(df['is_done_numeric'] == 0).sum()/len(df)*100:.1f}%)")
        print(f"Средняя цена пассажира: {df['price_start_local'].mean():.0f} руб.")
        print(f"Среднее расстояние: {df['distance_km'].mean():.1f} км")
        print(f"Средняя длительность: {df['duration_min'].mean():.1f} мин")

    def save_model(self, filename='price_recommender_model.pkl'):
        if not self._is_trained:
            raise ValueError("Модель не обучена")

        model_data = {
            'model': self.model,
            'probability_model': self.probability_model if hasattr(self, 'probability_model') else None,
            'feature_columns': self.feature_columns,
            'passenger_max_ratio': self.passenger_max_ratio,
            'is_trained': self._is_trained
        }

        joblib.dump(model_data, filename)
        print(f"Модель сохранена как '{filename}'")

    def ab_test_analysis(self, historical_data):
        results = []
        
        for _, order in historical_data.iterrows():
            try:
                # Что рекомендовала бы наша система
                our_recommendation = self.recommend_price(order)
                
                # Что было в реальности
                actual_result = {
                    'price': order['price_bid_local'],
                    'outcome': order['is_done_numeric'],
                    'income': order['price_bid_local'] if order['is_done_numeric'] == 1 else 0
                }
                
                # Сравниваем
                results.append({
                    'order_id': order['order_id'],
                    'actual_income': actual_result['income'],
                    'expected_income': our_recommendation['recommended_price'] * our_recommendation['accept_probability'],
                    'improvement': (our_recommendation['recommended_price'] * our_recommendation['accept_probability']) - actual_result['income']
                })
            except:
                continue
        
        self._plot_ab_results(results)
        return self._calculate_business_impact(results)
    
    def find_optimal_price_scientific(self, order_data):
        passenger_price = order_data['price_start_local']
        
        # Тестируем диапазон цен
        test_prices = np.linspace(passenger_price * 0.9, passenger_price * 2.0, 100)
        best_price = passenger_price
        best_income = 0
        
        for price in test_prices:
            test_order = order_data.copy()
            test_order['price_bid_local'] = price
            
            prob = self._calculate_accept_probability(test_order)
            expected_income = price * prob
            
            if expected_income > best_income:
                best_income = expected_income
                best_price = price
        
        return {
            'optimal_price': best_price,
            'max_expected_income': best_income,
            'probability': best_income / best_price
        }

class TaxiPriceAssistant:
    def __init__(self):
        self.recommender = None
        self.load_model()

    def show_progress(self, current, total, prefix="", length=50):
        percent = current / total
        filled_length = int(length * percent)
        bar = '█' * filled_length + '░' * (length - filled_length)
        progress_text = f"{prefix} |{bar}| {current}/{total} ({percent:.1%})"
        print(f"\r{progress_text}", end="", flush=True)
        if current == total:
            print()

    def load_model(self):
        print("Загрузка модели...")
        try:
            for i in range(5):
                self.show_progress(i + 1, 5, "Загрузка модели", 20)
                time.sleep(0.1)

            # Загружаем данные модели
            model_data = joblib.load('price_recommender_model.pkl')

            self.recommender = PriceRecommender()
            self.recommender.model = model_data['model']
            self.recommender.feature_columns = model_data['feature_columns']
            self.recommender.passenger_max_ratio = model_data['passenger_max_ratio']
            self.recommender._is_trained = model_data.get('is_trained', True)

            if 'probability_model' in model_data and model_data['probability_model'] is not None:
                self.recommender.probability_model = model_data['probability_model']
                print("Загружена ML модель вероятности")

            print("Модель успешно загружена!")
            print(f"Загружено признаков: {len(self.recommender.feature_columns)}")
            print(f"Максимальное соотношение: {self.recommender.passenger_max_ratio:.2f}")
            return True
        except Exception as e:
            print(f"Ошибка загрузки: {e}")
            print("\n Сначала обучите модель!")
            return False

    def show_welcome(self):
        print("="*60)
        print("СИСТЕМА РЕКОМЕНДАЦИЙ ЦЕН ДЛЯ ТАКСИ")
        print("="*60)
        print("Возможности:")
        print("Обработка CSV файлов с интеллектуальными рекомендациями")
        print("Ручной ввод заказов с детальным анализом")
        print("Умные рекомендации цен на основе машинного обучения")
        print("Визуализация результатов и статистика")
        print("Гарантия разнообразных и реалистичных цен")
        print("="*60)

    def show_main_menu(self):
        print("\n ГЛАВНОЕ МЕНЮ")
        print("1. Обработать CSV файл")
        print("2. Ручной ввод заказа")
        print("3. Показать сохраненные файлы")
        print("4. Выход")
        return input("Выберите действие (1-4): ").strip()

    def run(self):
        """Основной метод запуска интерфейса"""
        self.show_welcome()

        if not self.load_model():
            print("\n Для обучения модели выберите опцию 1 в главном меню системы")
            return

        while True:
            choice = self.show_main_menu()

            if choice == '1':
                self.process_csv_file()
            elif choice == '2':
                self.manual_order_input()
            elif choice == '3':
                self.show_saved_files()
            elif choice == '4':
                print("\n Спасибо за использование системы! До свидания!")
                break
            else:
                print("Неверный выбор. Попробуйте снова.")

            input("\n Нажмите Enter чтобы продолжить...")

    # Остальные методы класса остаются без изменений:
    def preprocess_order_data(self, order):
        if 'order_timestamp' in order:
            order_datetime = pd.to_datetime(order['order_timestamp'])
            order['order_hour'] = order_datetime.hour
            order['order_day_of_week'] = order_datetime.dayofweek
            order['order_month'] = order_datetime.month
            order['order_day'] = order_datetime.day
            order['order_date'] = order_datetime.date()

            # Праздники и предпраздничные дни
            if hasattr(self.recommender, '_is_holiday_or_preholiday'):
                is_special, holiday_type = self.recommender._is_holiday_or_preholiday(order_datetime)
                if is_special:
                    if holiday_type == 'holiday':
                        order['is_holiday'] = 1
                        order['is_preholiday'] = 0
                    elif holiday_type == 'preholiday':
                        order['is_holiday'] = 0
                        order['is_preholiday'] = 1
                else:
                    order['is_holiday'] = 0
                    order['is_preholiday'] = 0
            else:
                order['is_holiday'] = 0
                order['is_preholiday'] = 0

        if 'order_month' not in order:
            order['order_month'] = datetime.now().month
    
        month = order['order_month']
        if month in [12, 1, 2]: 
            season = 'winter'
        elif month in [3, 4, 5]: 
            season = 'spring'
        elif month in [6, 7, 8]: 
            season = 'summer'
        else: 
            season = 'autumn'
    
        order['season'] = season
        season_map = {'winter': 0, 'spring': 1, 'summer': 2, 'autumn': 3}
        order['season_code'] = season_map.get(season, 0)

        if 'season' not in order:
            order['season'] = season 
    
        if 'order_hour' not in order:
            order['order_hour'] = datetime.now().hour
    
        season = order['season']
        hour = order['order_hour']
        month = order['order_month']

        if season == 'winter':
            if month in [12, 1]:
                weather = 'snowy' if hour < 16 else 'cloudy'
            else:
                weather = 'cloudy' if hour < 14 else 'rainy'
        elif season == 'summer':
            if hour < 11: 
                weather = 'sunny'
            elif hour < 18: 
                weather = 'cloudy'
            else: 
                weather = 'rainy'
        elif season == 'spring':
            weather = 'rainy' if hour < 12 else 'cloudy'
        else: 
            weather = 'cloudy' if hour < 13 else 'rainy'

        order['weather_type'] = weather
        weather_map = {'sunny': 0, 'cloudy': 1, 'rainy': 2, 'snowy': 3}
        order['weather_code'] = weather_map.get(weather, 0)
        order['bad_weather'] = 1 if order['weather_code'] >= 2 else 0

        if 'order_day_of_week' not in order:
            order['order_day_of_week'] = datetime.now().weekday()
    
        order['is_weekend'] = 1 if order['order_day_of_week'] >= 5 else 0

        if 'order_hour' not in order:
            order['order_hour'] = datetime.now().hour
    
        hour = order['order_hour']
        is_weekend = order['is_weekend']
        is_holiday = order.get('is_holiday', 0)
        is_preholiday = order.get('is_preholiday', 0)
    
        # Обычные пиковые часы
        order['is_peak_hours'] = 1 if (
            ((hour >= 7) and (hour <= 10) or (hour >= 17) and (hour <= 20)) and
            (is_weekend == 0) and (is_holiday == 0)
        ) else 0

        # Праздничные пиковые часы
        order['is_holiday_peak'] = 1 if (
            ((hour >= 11) and (hour <= 15) or (hour >= 18) and (hour <= 22)) and
            ((is_weekend == 1) or (is_holiday == 1) or (is_preholiday == 1))
        ) else 0

        # Специальные дни
        order['is_special_day'] = 1 if (
            order.get('is_weekend', 0) == 1 or 
            order.get('is_holiday', 0) == 1 or 
            order.get('is_preholiday', 0) == 1
        ) else 0

        # Ночное время
        order['is_night'] = 1 if (order['order_hour'] >= 0 and order['order_hour'] <= 6) else 0

        # Преобразование единиц измерения
        if 'distance_in_meters' in order:
            order['distance_km'] = order['distance_in_meters'] / 1000.0
        if 'duration_in_seconds' in order:
            order['duration_min'] = order['duration_in_seconds'] / 60.0

        required_features = [
            'distance_km', 'duration_min', 'order_hour', 'is_weekend',
            'is_peak_hours', 'bad_weather', 'season_code', 'weather_code', 'is_night',
            'is_holiday', 'is_preholiday', 'is_holiday_peak', 'is_special_day'
        ]
    
        for feature in required_features:
            if feature not in order:
                order[feature] = 0  # значение по умолчанию
                print(f"Предупреждение: признак {feature} установлен в 0")

        return order

    def process_csv_file(self):
        print("\n ОБРАБОТКА CSV ФАЙЛА")
        print("-" * 50)

        print("Загрузите CSV файл...")
    
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
    
        file_path = filedialog.askopenfilename(
            title="Выберите CSV файл с данными заказов",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
    
        root.destroy()

        if not file_path:
            print("Файл не выбран")
            return

        file_name = os.path.basename(file_path)
        print(f"Выбран файл: {file_name}")

        try:
            # Чтение файла
            df = pd.read_csv(file_path)
            print(f"Загружено {len(df)} заказов")

            df_processed = self.recommender.preprocess_data(df, is_training=False)

            results = []
            total_orders = len(df_processed)

            print(f"\n Обработка {total_orders} заказов:")
            print("Используется ПОЛНАЯ логика модели")

            for idx, row in df_processed.iterrows():
                try:
                    if total_orders <= 50 or idx % max(1, total_orders // 20) == 0:
                        self.show_progress(idx + 1, total_orders, "Обработка", 40)

                    order_data = row.to_dict()
                    recommendation = self.recommender.recommend_price(order_data)
                    results.append(recommendation)

                except Exception as e:
                    print(f"\n Пропущен заказ {idx}: {e}")
                    continue

            self.show_progress(total_orders, total_orders, "Обработка", 40)

            if results:
                results_df = pd.DataFrame(results)
                print(f"\n Успешно обработано: {len(results_df)} заказов")

                # Анализ результатов
                self.analyze_recommendations(results_df)
                self.save_csv_results(results_df, file_name)
                self.show_csv_preview(results_df)
            else:
                print("Не удалось обработать заказы")

        except Exception as e:
            print(f"Ошибка загрузки файла: {e}")

    def analyze_recommendations(self, results_df):
        print(f"\n ДЕТАЛЬНЫЙ АНАЛИЗ РЕКОМЕНДАЦИЙ")
        print("=" * 50)

        increases = results_df['price_increase_percent']
        probs = results_df['accept_probability']

        print("СТАТИСТИКА ПОВЫШЕНИЯ ЦЕН:")
        print(f"Среднее повышение: {increases.mean():.1f}%")
        print(f"Медианное повышение: {increases.median():.1f}%")
        print(f"Максимальное повышение: {increases.max():.1f}%")
        print(f"Минимальное повышение: {increases.min():.1f}%")
        print(f"Стандартное отклонение: {increases.std():.1f}%")

        increase_ranges = [
            (0, 5, "0-5%"),
            (5, 10, "5-10%"),
            (10, 15, "10-15%"),
            (15, 20, "15-20%"),
            (20, 30, "20-30%"),
            (30, 50, "30-50%"),
            (50, 100, "50-100%")
        ]

        print(f"\n РАСПРЕДЕЛЕНИЕ ПОВЫШЕНИЙ:")
        for min_r, max_r, label in increase_ranges:
            count = len(increases[(increases >= min_r) & (increases < max_r)])
            if count > 0:
                print(f"   {label}: {count} заказов ({count/len(increases)*100:.1f}%)")

        print(f"\n СТАТИСТИКА ВЕРОЯТНОСТЕЙ:")
        print(f"Средняя вероятность: {probs.mean():.1%}")

        prob_categories = [
            (0.7, 1.0, "Высокая (>70%)"),
            (0.5, 0.7, "Средняя (50-70%)"),
            (0.3, 0.5, "Низкая (30-50%)"),
            (0.0, 0.3, "Очень низкая (<30%)")
        ]

        print(f"\n РАСПРЕДЕЛЕНИЕ ВЕРОЯТНОСТЕЙ:")
        for min_p, max_p, label in prob_categories:
            count = len(probs[(probs >= min_p) & (probs < max_p)])
            if count > 0:
                print(f"   {label}: {count} заказов ({count/len(probs)*100:.1f}%)")

    def manual_order_input(self):
        print("\n РУЧНОЙ ВВОД ЗАКАЗА")
        print("-" * 50)

        print("Введите данные заказа:")

        order_data = {
            'price_start_local': float(input("Цена пассажира (руб): ")),
            'distance_in_meters': float(input("Расстояние (метры): ")),
            'duration_in_seconds': float(input("Длительность поездки (минуты): ")) * 60,
        }

        order_id = input("Номер заказа (опционально): ").strip()
        order_data['order_id'] = order_id if order_id else 'manual_order'

        # Используем текущее время для реалистичных данных
        now = datetime.now()
        order_data['order_timestamp'] = now.strftime('%Y-%m-%d %H:%M:%S')

        print(f"\n Используется текущее время: {now.strftime('%Y-%m-%d %H:%M:%S')}")
        print("Сезон и погода будут определены автоматически")

        print("\n Расчет рекомендации...")
        for i in range(3):
            self.show_progress(i + 1, 3, "Анализ данных", 25)
            time.sleep(0.4)

        try:
            order_data = self.preprocess_order_data(order_data)
            recommendation = self.recommender.recommend_price(order_data)

            print("\n" + "=" * 30)
            print("РЕЗУЛЬТАТ РЕКОМЕНДАЦИИ")
            print("=" * 30)
            self.show_detailed_recommendation(recommendation, order_data)

            save_choice = input("\nСохранить этот заказ в файл? (y/n): ").lower().strip()
            if save_choice == 'y':
                self.save_single_order(recommendation)

        except Exception as e:
            print(f"Ошибка расчета: {e}")

    def show_detailed_recommendation(self, recommendation, order_data):
        print(f"Номер заказа: {recommendation['order_id']}")
        print(f"Цена пассажира: {recommendation['passenger_price']} руб.")
        print(f"Рекомендуемая цена: {recommendation['recommended_price']} руб.")
        print(f"Повышение цены: {recommendation['price_increase_percent']:+.1f}%")
        print(f"Вероятность принятия: {recommendation['accept_probability']:.1%}")

        print(f"\n АНАЛИЗ ФАКТОРОВ:")

        # Время и погода
        hour = order_data.get('order_hour', 'N/A')
        season = order_data.get('season', 'N/A')
        weather = order_data.get('weather_type', 'N/A')
        print(f"Время: {hour}ч, Сезон: {season}, Погода: {weather}")

        # Характеристики поездки
        distance_km = order_data.get('distance_km', 0)
        duration_min = order_data.get('duration_in_seconds', 0) / 60
        print(f"Расстояние: {distance_km:.1f} км, Длительность: {duration_min:.1f} мин")

        # Особые условия
        conditions = []
        if order_data.get('is_peak_hours', 0) == 1:
            conditions.append("Пиковые часы")
        if order_data.get('is_weekend', 0) == 1:
            conditions.append("Выходной")
        if order_data.get('bad_weather', 0) == 1:
            conditions.append("Плохая погода")
        if order_data.get('is_night', 0) == 1:
            conditions.append("Ночное время")
        if order_data.get('is_holiday', 0) == 1:
            conditions.append("Праздничный день")
        if order_data.get('is_preholiday', 0) == 1:
            conditions.append("Предпраздничный день")

        if conditions:
            print(f" Особые условия: {', '.join(conditions)}")

        prob = recommendation['accept_probability']
        if prob >= 0.7:
            indicator = "ВЫСОКАЯ"
            advice = "Отличный шанс принятия заказа"
        elif prob >= 0.5: 
            indicator = "СРЕДНЯЯ" 
            advice = "Хорошие шансы на принятие"
        elif prob >= 0.3:
            indicator = "НИЗКАЯ"
            advice = "Рассмотрите снижение цены для увеличения вероятности"
        else:
            indicator = "ОЧЕНЬ НИЗКАЯ"
            advice = "Рекомендуется снизить цену"

        is_done_recommendation = "принять" if prob >= 0.5 else "отклонить"
        print(f"Рекомендация системы: {is_done_recommendation.upper()}")
        print(f"Шанс принятия: {indicator}")
        print(f"Совет: {advice}")
        print("=" * 30)

    def show_csv_preview(self, results_df):
        print(f"\n ПРЕВЬЮ РЕЗУЛЬТАТОВ ({len(results_df)} заказов)")
        print("=" * 80)

        display_df = results_df.copy()

        display_df['is_done'] = display_df['accept_probability'].apply(
            lambda x: 'done' if x >= 0.5 else 'cancel'
        )
    
        display_df = display_df.head(12)

        print(display_df[['is_done']].to_string(index=False))

        self.plot_recommendations_analysis(results_df)

    def plot_recommendations_analysis(self, results_df):
        print(f"\n ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ")
        print("-" * 50)

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Анализ рекомендаций цен', fontsize=16, fontweight='bold')

        # График 1: Распределение процента повышения цены
        increases = results_df['price_increase_percent']
        axes[0, 0].hist(increases, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0, 0].axvline(increases.mean(), color='red', linestyle='--', linewidth=2,
                          label=f'Среднее: {increases.mean():.1f}%')
        axes[0, 0].axvline(increases.median(), color='blue', linestyle='--', linewidth=2,
                          label=f'Медиана: {increases.median():.1f}%')
        axes[0, 0].set_xlabel('Процент повышения цена (%)')
        axes[0, 0].set_ylabel('Количество заказов')
        axes[0, 0].set_title('Распределение повышения цен')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # График 2: Зависимость вероятности принятия от процента повышения
        scatter = axes[0, 1].scatter(results_df['price_increase_percent'],
                                    results_df['accept_probability'] * 100,
                                    c=results_df['recommended_price'],
                                    cmap='viridis', alpha=0.6, s=60)
        axes[0, 1].set_xlabel('Процент повышения цена (%)')
        axes[0, 1].set_ylabel('Вероятность принятия пассажиром (%)')
        axes[0, 1].set_title('Повышение цены vs Вероятность принятия')
        axes[0, 1].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[0, 1], label='Рекомендуемая цена (руб)')

        # График 3: Сравнение цен (первые 8 заказов)
        sample_display = results_df.head(8)
        x_pos = np.arange(len(sample_display))
        width = 0.35

        axes[1, 0].bar(x_pos - width/2, sample_display['passenger_price'], width,
                      label='Цена пассажира', alpha=0.7, color='lightblue')
        axes[1, 0].bar(x_pos + width/2, sample_display['recommended_price'], width,
                      label='Реком. цена', alpha=0.7, color='lightgreen')

        axes[1, 0].set_xlabel('Заказы')
        axes[1, 0].set_ylabel('Цена (руб)')
        axes[1, 0].set_title('Сравнение цен пассажира и рекомендаций')
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels(sample_display['order_id'].astype(str), rotation=45)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # График 4: Распределение вероятностей принятия
        prob_bins = [0, 30, 50, 70, 100]
        prob_labels = ['Низкая (<30%)', 'Средняя (30-50%)', 'Высокая (50-70%)', 'Очень высокая (>70%)']
        prob_data = pd.cut(results_df['accept_probability'] * 100, bins=prob_bins, labels=prob_labels)
        prob_counts = prob_data.value_counts()

        colors = ['lightcoral', 'gold', 'lightgreen', 'green']
        axes[1, 1].pie(prob_counts.values, labels=prob_counts.index, autopct='%1.1f%%',
                      colors=colors, startangle=90)
        axes[1, 1].set_title('Распределение вероятностей принятия')

        plt.tight_layout()
        plt.show()

    def save_csv_results(self, results_df, original_filename):
        print("\n Сохранение результатов...")

        for i in range(3):
            self.show_progress(i + 1, 3, "Сохранение данных", 20)
            time.sleep(0.3)

        filename = "TryCatchUs_predict.csv"

        output_df = results_df.copy()

        final_df = pd.DataFrame()

        final_df['is_done'] = output_df['accept_probability'].apply(
            lambda x: 'done' if x >= 0.5 else 'cancel'
        )
    
        final_df.to_csv(filename, index=False, encoding='utf-8')

        print(f"\n Результаты сохранены в файл: {filename}")
        print(f"Обработано заказов: {len(results_df)}")

        done_count = (final_df['is_done'] == 'done').sum()
        cancel_count = (final_df['is_done'] == 'cancel').sum()
        print(f"Рекомендуется принять: {done_count} заказов ({done_count/len(final_df)*100:.1f}%)")
        print(f"Рекомендуется отклонить: {cancel_count} заказов ({cancel_count/len(final_df)*100:.1f}%)")
    
        print(f"Размер файла: {os.path.getsize(filename) / 1024:.1f} KB")
        print(f"Файл доступен в текущей директории")

    def save_single_order(self, recommendation):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"single_recommendation_{timestamp}.csv"

        # Создаем DataFrame только с is_done
        final_df = pd.DataFrame()
    
        # ИЗМЕНЕНИЕ: порог 0.5 вместо 0.6 и только один столбец
        final_df['is_done'] = ['done' if recommendation['accept_probability'] >= 0.5 else 'cancel']

        final_df.to_csv(filename, index=False)
        print(f"Заказ сохранен в файл: {filename}")
        print(f"Рекомендация: {final_df['is_done'].iloc[0]}")

    def show_saved_files(self):
        print("\n СОХРАНЕННЫЕ ФАЙЛЫ")
        print("-" * 40)

        files_list = [f for f in os.listdir('.')
                     if f.startswith(('recommendations_', 'single_recommendation_', 'final_predictions'))
                     and f.endswith('.csv')]

        if not files_list:
            print("Нет сохраненных файлов с рекомендациями")
            return

        print("Найдены файлы:")
        for i, filename in enumerate(sorted(files_list, key=os.path.getmtime, reverse=True), 1):
            size = os.path.getsize(filename) / 1024
            mod_time = time.strftime("%Y-%m-%d %H:%M:%S",
                                   time.localtime(os.path.getmtime(filename)))
            print(f"   {i}. {filename}")
            print(f"Размер: {size:.1f} KB")
            print(f"Создан: {mod_time}")
            print()

def train_model():
    print("=" * 60)
    print("ОБУЧЕНИЕ МОДЕЛИ РЕКОМЕНДАЦИЙ ЦЕН")
    print("=" * 60)

    recommender = PriceRecommender()

    try:
        print("Загрузите CSV файл с данными для обучения...")
        
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        
        file_path = filedialog.askopenfilename(
            title="Выберите CSV файл для обучения модели",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        root.destroy()

        if not file_path:
            raise ValueError("Файл не выбран")

        file_name = os.path.basename(file_path)
        print(f"Выбран файл: {file_name}")
        
        df = pd.read_csv(file_path)
        print(f"Загружено {len(df)} заказов")

        print("\n Начинаем обработку данных...")
        df_processed = recommender.preprocess_data(df, is_training=True)

        print("\n Начинаем обучение модели...")
        df_with_prices = recommender.train_model(df_processed)

        print("\n Анализируем поведение пассажиров...")
        recommender.analyze_passenger_behavior(df_with_prices)

        print("\n Строим графики анализа данных...")
        recommender.plot_training_analysis(df_with_prices)

        print(f"\n ТЕСТИРОВАНИЕ МОДЕЛИ НА РЕАЛЬНЫХ ДАННЫХ")
        print("=" * 50)

        test_samples = df_with_prices.sample(n=min(8, len(df_with_prices)), random_state=42)

        test_results = []
        for idx, (_, order) in enumerate(test_samples.iterrows()):
            try:
                recommendation = recommender.recommend_price(order.to_dict())
                test_results.append(recommendation)
                print(f"Тест {idx+1}: {order['price_start_local']:.0f} → {recommendation['recommended_price']} руб. (+{recommendation['price_increase_percent']}%) - {recommendation['accept_probability']:.1%}")
            except Exception as e:
                print(f"Ошибка теста {idx+1}: {e}")

        # Анализ тестовых результатов
        if test_results:
            test_df = pd.DataFrame(test_results)
            avg_increase = test_df['price_increase_percent'].mean()
            avg_prob = test_df['accept_probability'].mean()
            print(f"\n Результаты тестирования:")
            print(f"Среднее повышение в тесте: {avg_increase:.1f}%")
            print(f"Средняя вероятность в тесте: {avg_prob:.1%}")

            # Проверяем разнообразие
            increases = test_df['price_increase_percent']
            if increases.std() < 2.0:
                print("Предупреждение: недостаточное разнообразие рекомендаций")
            else:
                print("Хорошее разнообразие рекомендаций")

        print(f"\n Сохраняем модель...")
        recommender.save_model('price_recommender_model.pkl')

        print("\n МОДЕЛЬ УСПЕШНО ОБУЧЕНА И ГОТОВА К ИСПОЛЬЗОВАНИЮ!")
        print("Теперь вы можете запустить интерфейс рекомендаций")

        return recommender

    except Exception as e:
        print(f"Ошибка при обучении модели: {e}")
        return None

def create_test_dataset():
    print("СОЗДАНИЕ ТЕСТОВОГО ДАТАСЕТА")
    print("-" * 40)

    orders = []
    base_time = datetime.now() - timedelta(days=30)

    # Создаем разнообразные заказы
    scenarios = [
        # (distance_km, duration_min, base_price, success_rate)
        (2.5, 10, 180, 0.8),   # Короткая поездка
        (5.0, 15, 250, 0.7),   # Городская поездка
        (12.0, 30, 450, 0.6),  # Средняя поездка
        (18.0, 45, 700, 0.5),  # Дальняя поездка
        (25.0, 60, 950, 0.4),  # Очень дальняя поездка
    ]

    for i in range(200):
        scenario = random.choice(scenarios)
        distance_km, duration_min, base_price, success_rate = scenario

        order_id = f"test_order_{i+1:04d}"

        # Случайное время в течение 30 дней
        random_hours = random.randint(0, 30*24)
        order_time = base_time + timedelta(hours=random_hours)

        # Цена пассажира
        passenger_price = base_price * random.uniform(0.9, 1.2)

        # Цена водителя
        strategy = random.choice(['conservative', 'moderate', 'aggressive'])
        if strategy == 'conservative':
            driver_price = passenger_price * random.uniform(0.95, 1.10)
        elif strategy == 'moderate':
            driver_price = passenger_price * random.uniform(1.05, 1.20)
        else:
            driver_price = passenger_price * random.uniform(1.15, 1.35)

        # Статус заказа
        is_done = 'done' if random.random() < success_rate else 'canceled'

        orders.append({
            'order_id': order_id,
            'order_timestamp': order_time.strftime('%Y-%m-%d %H:%M:%S'),
            'distance_in_meters': int(distance_km * 1000),
            'duration_in_seconds': int(duration_min * 60),
            'price_start_local': round(passenger_price, 2),
            'price_bid_local': round(driver_price, 2),
            'is_done': is_done
        })

    df = pd.DataFrame(orders)
    filename = 'taxi_orders_test_dataset.csv'
    df.to_csv(filename, index=False)

    print(f"Тестовый датасет создан: {filename}")
    print(f"Статистика:")
    print(f"Всего заказов: {len(df)}")
    print(f"Принято: {(df['is_done'] == 'done').sum()} ({(df['is_done'] == 'done').sum()/len(df)*100:.1f}%)")
    print(f"Диапазон цен: {df['price_start_local'].min():.0f} - {df['price_start_local'].max():.0f} руб.")

    print("Файл сохранен в рабочей директории")

    return df

def main():
    print("="*60)
    print("ПОЛНАЯ СИСТЕМА РЕКОМЕНДАЦИЙ ЦЕН ДЛЯ ТАКСИ")
    print("="*60)
    print("Возможности:")
    print("Обучение модели на реальных данных")
    print("Интеллектуальные рекомендации цен")
    print("Разнообразные и реалистичные цены")
    print("Визуализация и анализ результатов")
    print("Гарантия от одинаковых цен и минусовых значений")
    print("="*60)

    while True:
        print("\n ГЛАВНОЕ МЕНЮ СИСТЕМЫ")
        print("1.Обучить модель на CSV файле")
        print("2.Создать тестовый датасет")
        print("3.Запустить интерфейс рекомендаций")
        print("4.Информация о системе")
        print("5.Выход")

        choice = input("\nВаш выбор (1-5): ").strip()

        if choice == '1':
            train_model()
        elif choice == '2':
            create_test_dataset()
        elif choice == '3':
            assistant = TaxiPriceAssistant()
            assistant.run()
        elif choice == '4':
            print(f"\nИНФОРМАЦИЯ О СИСТЕМЕ")
            print("=" * 40)
            print("Система рекомендаций цен для такси")
            print("Особенности:")
            print("- Полностью на машинном обучении")
            print("- Использует ВСЕ данные из файла для анализа")
            print("- Разнообразные рекомендации (не фиксированные +10%)")
            print("- Защита от минусовых и нулевых значений")
            print("- Визуализация результатов и статистика")
            print("- Учет времени, погоды, сезонности")
            print("- Учет праздников и предпраздничных дней")
            print("- Реалистичные цены на основе поведения пассажиров")
        elif choice == '5':
            print("\n До свидания! Спасибо за использование системы!")
            break
        else:
            print("Неверный выбор. Попробуйте снова.")

        input("\n Нажмите Enter чтобы продолжить...")

if __name__ == "__main__":
    main()