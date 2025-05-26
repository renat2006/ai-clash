#!/usr/bin/env python3
"""
🚀 Yandex Cloud решение для соревнования Clash Royale
Оптимизировано для Yandex DataSphere и Compute Cloud
"""

import os
import sys
import subprocess
import requests
import zipfile
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# Yandex Cloud специфичные настройки
YANDEX_ENV = '/home/jupyter' in os.getcwd() or 'DATASPHERE' in os.environ

def install_dependencies():
    """
    Устанавливает зависимости для Yandex Cloud
    """
    print("🔧 УСТАНОВКА ЗАВИСИМОСТЕЙ ДЛЯ YANDEX CLOUD")
    print("=" * 45)
    
    dependencies = [
        'polars>=0.20.0',
        'requests>=2.25.0',
        'catboost>=1.2.0',
        'scikit-learn>=1.3.0'
    ]
    
    # Проверяем установленные пакеты
    try:
        import pandas, numpy
        print("✅ Pandas и NumPy установлены")
    except ImportError:
        print("⚠️  Устанавливаем базовые пакеты...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pandas', 'numpy', '--quiet'])
    
    for dep in dependencies:
        try:
            print(f"📦 Устанавливаем {dep}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', dep, '--quiet'])
        except subprocess.CalledProcessError as e:
            print(f"⚠️  Ошибка установки {dep}: {e}")
            print("Продолжаем без этой зависимости...")

def check_yandex_gpu():
    """
    Проверяет доступность GPU в Yandex Cloud
    """
    print("🎮 ПРОВЕРКА GPU В YANDEX CLOUD")
    print("-" * 35)
    
    # Проверяем переменные окружения
    if 'DATASPHERE' in os.environ:
        print("✅ Yandex DataSphere окружение обнаружено")
    
    # Проверяем NVIDIA GPU
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ NVIDIA GPU доступен!")
            # Показываем информацию о GPU
            lines = result.stdout.split('\n')
            for line in lines:
                if any(gpu in line for gpu in ['Tesla', 'GeForce', 'Quadro', 'V100', 'T4', 'A100']):
                    print(f"🚀 GPU: {line.strip()}")
                    break
            return True
        else:
            print("⚠️  nvidia-smi недоступен")
    except FileNotFoundError:
        print("⚠️  nvidia-smi не найден")
    
    # Проверяем через CatBoost
    try:
        import catboost
        test_model = catboost.CatBoostRegressor(
            iterations=1,
            task_type='GPU',
            devices='0',
            verbose=False
        )
        print("✅ CatBoost может использовать GPU")
        return True
    except Exception as e:
        print(f"⚠️  CatBoost GPU недоступен: {e}")
        print("💻 Будем использовать CPU")
        return False

def download_data_yandex():
    """
    Загружает данные для Yandex Cloud
    """
    print("📥 ЗАГРУЗКА ДАННЫХ")
    print("-" * 20)
    
    required_files = ['train.csv', 'test.csv', 'submission_example.csv']
    
    # Проверяем, есть ли файлы уже
    files_exist = all(os.path.exists(file) for file in required_files)
    
    if files_exist:
        print("✅ Файлы уже существуют")
        return
    
    # Проверяем возможные пути в Yandex Cloud
    yandex_paths = [
        '/home/jupyter/work/resources',
        '/home/jupyter/work',
        './data',
        './input',
        '.'
    ]
    
    for path in yandex_paths:
        if os.path.exists(path):
            print(f"🔍 Проверяем {path}...")
            for file in required_files:
                file_path = os.path.join(path, file)
                if os.path.exists(file_path):
                    print(f"✅ Найден {file} в {path}")
                    # Копируем в рабочую директорию
                    import shutil
                    shutil.copy2(file_path, file)
    
    # Проверяем еще раз
    files_exist = all(os.path.exists(file) for file in required_files)
    
    if not files_exist:
        print("📥 Загружаем данные из интернета...")
        url = "http://devopn.ru:8000/cu-base-project.zip"
        
        try:
            print("🌐 Подключаемся к серверу...")
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            
            print("📦 Распаковываем архив...")
            with zipfile.ZipFile(BytesIO(response.content)) as zip_ref:
                zip_ref.extractall()
            
            print("✅ Данные успешно загружены!")
            
        except Exception as e:
            print(f"❌ Ошибка загрузки: {e}")
            print("💡 Попробуйте загрузить файлы вручную:")
            print("1. Скачайте http://devopn.ru:8000/cu-base-project.zip")
            print("2. Распакуйте в рабочую папку")
            print("3. Перезапустите скрипт")
            raise

def main():
    """
    Основная функция для Yandex Cloud
    """
    print("🚀 YANDEX CLOUD CLASH ROYALE SOLUTION")
    print("=" * 50)
    
    # Информация о среде
    print(f"🐍 Python: {sys.version}")
    print(f"📁 Рабочая директория: {os.getcwd()}")
    
    # Проверяем доступное место
    try:
        statvfs = os.statvfs('.')
        free_space = statvfs.f_bavail * statvfs.f_frsize // (1024**3)
        print(f"💾 Свободное место: {free_space} GB")
    except:
        print("💾 Информация о диске недоступна")
    
    if YANDEX_ENV:
        print("✅ Запуск в Yandex Cloud")
    else:
        print("⚠️  Запуск вне Yandex Cloud")
    
    # Установка зависимостей
    install_dependencies()
    
    # Импорты
    try:
        import polars as pl
        import numpy as np
        import pandas as pd
        from catboost import CatBoostRegressor
        print("✅ Все библиотеки импортированы успешно")
    except ImportError as e:
        print(f"❌ Ошибка импорта: {e}")
        raise
    
    # Проверка GPU
    use_gpu = check_yandex_gpu()
    
    # Загрузка данных
    download_data_yandex()
    
    # Глобальные переменные для частот игроков
    global_p1_freq_map = {}
    global_p2_freq_map = {}
    
    print("\n📊 ЗАГРУЗКА И АНАЛИЗ ДАННЫХ")
    print("-" * 35)
    
    # Загружаем данные
    df_train = pl.read_csv('train.csv')
    df_test = pl.read_csv('test.csv')
    submission = pd.read_csv('submission_example.csv')
    
    print(f"📈 Train shape: {df_train.shape}")
    print(f"📉 Test shape: {df_test.shape}")
    print(f"📋 Submission shape: {submission.shape}")
    
    # Показываем примеры данных
    print(f"\n🔍 Пример train данных:")
    print(df_train.head(3).to_pandas())
    
    def create_yandex_features(df, is_train=True):
        """
        Создание признаков, оптимизированных для Yandex Cloud
        """
        print(f"🔧 Создание признаков ({'train' if is_train else 'test'})...")
        
        # 1. БАЗОВЫЕ ТРОФЕЙНЫЕ ПРИЗНАКИ
        df = df.with_columns([
            (pl.col('player_1_trophies') - pl.col('player_2_trophies')).alias('trophy_diff'),
            (pl.col('player_1_trophies') + pl.col('player_2_trophies')).alias('trophy_sum'),
            (pl.col('player_1_trophies') / (pl.col('player_2_trophies') + 1)).alias('trophy_ratio'),
            (pl.col('player_1_trophies') - pl.col('player_2_trophies')).abs().alias('abs_trophy_diff'),
            (pl.col('player_1_trophies') * pl.col('player_2_trophies')).alias('trophy_product'),
            ((pl.col('player_1_trophies') + pl.col('player_2_trophies')) / 2).alias('trophy_mean')
        ])
        
        # 2. ПРИЗНАКИ КАРТ
        card_cols_p1 = [f'player_1_card_{i}' for i in range(1, 9)]
        card_cols_p2 = [f'player_2_card_{i}' for i in range(1, 9)]
        
        df = df.with_columns([
            pl.mean_horizontal([pl.col(f'player_1_card_{i}') for i in range(1, 9)]).alias('player_1_avg_card'),
            pl.mean_horizontal([pl.col(f'player_2_card_{i}') for i in range(1, 9)]).alias('player_2_avg_card'),
            pl.min_horizontal([pl.col(f'player_1_card_{i}') for i in range(1, 9)]).alias('player_1_min_card'),
            pl.min_horizontal([pl.col(f'player_2_card_{i}') for i in range(1, 9)]).alias('player_2_min_card'),
            pl.max_horizontal([pl.col(f'player_1_card_{i}') for i in range(1, 9)]).alias('player_1_max_card'),
            pl.max_horizontal([pl.col(f'player_2_card_{i}') for i in range(1, 9)]).alias('player_2_max_card'),
            pl.concat_list(card_cols_p1).list.median().alias('player_1_median_card'),
            pl.concat_list(card_cols_p2).list.median().alias('player_2_median_card'),
            pl.concat_list(card_cols_p1).list.std().alias('p1_card_std'),
            pl.concat_list(card_cols_p2).list.std().alias('p2_card_std')
        ])
        
        # Разности карт
        df = df.with_columns([
            (pl.col('player_1_avg_card') - pl.col('player_2_avg_card')).alias('avg_card_diff'),
            (pl.col('player_1_min_card') - pl.col('player_2_min_card')).alias('min_card_diff'),
            (pl.col('player_1_max_card') - pl.col('player_2_max_card')).alias('max_card_diff'),
            (pl.col('player_1_median_card') - pl.col('player_2_median_card')).alias('median_card_diff'),
            (pl.col('p1_card_std') - pl.col('p2_card_std')).alias('card_std_diff')
        ])
        
        # 3. ОБЩИЕ КАРТЫ
        common_cards_expr = pl.lit(0)
        for i in range(1, 9):
            for j in range(1, 9):
                common_cards_expr = common_cards_expr + (
                    pl.col(f'player_1_card_{i}') == pl.col(f'player_2_card_{j}')
                ).cast(pl.Int32)
        
        df = df.with_columns([
            common_cards_expr.alias('common_cards_count'),
            (common_cards_expr / 8.0).alias('common_cards_ratio')
        ])
        
        # 4. ВРЕМЕННЫЕ ПРИЗНАКИ
        df = df.with_columns([
            pl.col('datetime').str.strptime(pl.Datetime, format='%Y%m%dT%H%M%S%.fZ').alias('parsed_datetime')
        ])
        
        df = df.with_columns([
            pl.col('parsed_datetime').dt.hour().alias('hour'),
            pl.col('parsed_datetime').dt.day().alias('day'),
            pl.col('parsed_datetime').dt.month().alias('month'),
            pl.col('parsed_datetime').dt.weekday().alias('weekday'),
            # Циклические признаки для времени
            (2 * np.pi * pl.col('parsed_datetime').dt.hour() / 24).sin().alias('hour_sin'),
            (2 * np.pi * pl.col('parsed_datetime').dt.hour() / 24).cos().alias('hour_cos'),
            (2 * np.pi * pl.col('parsed_datetime').dt.weekday() / 7).sin().alias('weekday_sin'),
            (2 * np.pi * pl.col('parsed_datetime').dt.weekday() / 7).cos().alias('weekday_cos')
        ])
        
        # 5. КАТЕГОРИАЛЬНЫЕ ПРИЗНАКИ
        df = df.with_columns([
            pl.when(pl.col('player_1_trophies') < 1000).then(pl.lit('beginner'))
            .when(pl.col('player_1_trophies') < 3000).then(pl.lit('intermediate'))
            .when(pl.col('player_1_trophies') < 5000).then(pl.lit('advanced'))
            .when(pl.col('player_1_trophies') < 7000).then(pl.lit('expert'))
            .otherwise(pl.lit('master')).alias('player_1_skill_level'),
            
            pl.when(pl.col('player_2_trophies') < 1000).then(pl.lit('beginner'))
            .when(pl.col('player_2_trophies') < 3000).then(pl.lit('intermediate'))
            .when(pl.col('player_2_trophies') < 5000).then(pl.lit('advanced'))
            .when(pl.col('player_2_trophies') < 7000).then(pl.lit('expert'))
            .otherwise(pl.lit('master')).alias('player_2_skill_level'),
            
            pl.when(pl.col('trophy_sum') < 2000).then(pl.lit('low_tier'))
            .when(pl.col('trophy_sum') < 6000).then(pl.lit('mid_tier'))
            .when(pl.col('trophy_sum') < 10000).then(pl.lit('high_tier'))
            .otherwise(pl.lit('top_tier')).alias('match_tier')
        ])
        
        # 6. ВЗАИМОДЕЙСТВИЯ
        df = df.with_columns([
            (pl.col('trophy_diff') * pl.col('avg_card_diff')).alias('trophy_card_interaction'),
            (pl.col('trophy_diff') * pl.col('gamemode')).alias('trophy_gamemode_interaction'),
            (pl.col('player_1_avg_card') / (pl.col('player_1_trophies') + 1)).alias('p1_card_trophy_ratio'),
            (pl.col('player_2_avg_card') / (pl.col('player_2_trophies') + 1)).alias('p2_card_trophy_ratio')
        ])
        
        # 7. ОПЫТ ИГРОКОВ
        if is_train:
            nonlocal global_p1_freq_map, global_p2_freq_map
            p1_freq = df.group_by('player_1_tag').agg(pl.len().alias('p1_freq'))
            p2_freq = df.group_by('player_2_tag').agg(pl.len().alias('p2_freq'))
            
            df = df.join(p1_freq, on='player_1_tag', how='left')
            df = df.join(p2_freq, on='player_2_tag', how='left')
            
            global_p1_freq_map = p1_freq.to_pandas().set_index('player_1_tag')['p1_freq'].to_dict()
            global_p2_freq_map = p2_freq.to_pandas().set_index('player_2_tag')['p2_freq'].to_dict()
        else:
            df = df.with_columns([
                pl.col('player_1_tag').map_elements(
                    lambda x: global_p1_freq_map.get(x, 1), return_dtype=pl.Int64
                ).alias('p1_freq'),
                pl.col('player_2_tag').map_elements(
                    lambda x: global_p2_freq_map.get(x, 1), return_dtype=pl.Int64
                ).alias('p2_freq')
            ])
        
        df = df.with_columns([
            (pl.col('p1_freq') - pl.col('p2_freq')).alias('experience_diff'),
            (pl.col('p1_freq') + pl.col('p2_freq')).alias('total_experience'),
            (pl.col('p1_freq') / (pl.col('p2_freq') + 1)).alias('experience_ratio')
        ])
        
        # 8. ПРОДВИНУТЫЕ МАТЕМАТИЧЕСКИЕ ПРИЗНАКИ
        df = df.with_columns([
            (pl.col('trophy_sum') + 1).log().alias('log_trophy_sum'),
            (pl.col('total_experience') + 1).log().alias('log_total_experience'),
            (pl.col('trophy_diff') ** 2).alias('trophy_diff_squared'),
            (pl.col('avg_card_diff') ** 2).alias('avg_card_diff_squared'),
            pl.col('trophy_diff').abs().sqrt().alias('sqrt_abs_trophy_diff'),
            (pl.col('trophy_diff') ** 3).alias('trophy_diff_cubed')
        ])
        
        # Удаляем временные колонки
        df = df.drop('parsed_datetime')
        
        print(f"✅ Создано признаков. Размерность: {df.shape}")
        return df
    
    # Создание признаков
    print("\n🔧 FEATURE ENGINEERING")
    print("-" * 25)
    
    df_train = create_yandex_features(df_train, is_train=True)
    df_test = create_yandex_features(df_test, is_train=False)
    
    # Подготовка к обучению
    print("\n🤖 ПОДГОТОВКА К ОБУЧЕНИЮ")
    print("-" * 30)
    
    # Категориальные признаки
    cat_features = [
        'gamemode', 'player_1_tag', 'player_2_tag',
        'player_1_skill_level', 'player_2_skill_level', 'match_tier',
        'hour', 'day', 'month', 'weekday'
    ] + [f'player_1_card_{i}' for i in range(1, 9)] + [f'player_2_card_{i}' for i in range(1, 9)]
    
    print(f"📊 Категориальных признаков: {len(cat_features)}")
    
    # Настройка модели для Yandex Cloud
    model_params = {
        'cat_features': cat_features,
        'random_state': 52,
        'verbose': 100,  # Подробный вывод для мониторинга
        'iterations': 2000 if use_gpu else 1200,  # Больше итераций для лучшего качества
        'learning_rate': 0.08,
        'depth': 9,
        'l2_leaf_reg': 3,
        'bootstrap_type': 'Bernoulli',
        'subsample': 0.8,
        'eval_metric': 'RMSE',
        'early_stopping_rounds': 100
    }
    
    if use_gpu:
        model_params.update({
            'task_type': 'GPU',
            'devices': '0',
            'gpu_ram_part': 0.8  # Используем больше GPU памяти
        })
        print("🚀 Используем GPU для обучения")
    else:
        model_params['task_type'] = 'CPU'
        print("💻 Используем CPU для обучения")
    
    model = CatBoostRegressor(**model_params)
    
    # Подготовка данных
    X_train = df_train.drop(['id', 'datetime', 'target']).to_pandas()
    y_train = df_train['target'].to_pandas()
    
    print(f"📏 Размерность: {X_train.shape}")
    print(f"🎯 Таргет распределение:")
    print(y_train.value_counts().sort_index())
    
    # Обучение с валидацией
    print(f"\n⏳ ОБУЧЕНИЕ МОДЕЛИ")
    print("-" * 20)
    print("🕐 Это может занять 10-30 минут в зависимости от железа...")
    
    # Разделяем на train/validation для early stopping
    from sklearn.model_selection import train_test_split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    model.fit(
        X_tr, y_tr,
        eval_set=(X_val, y_val),
        use_best_model=True,
        plot=False
    )
    
    # Предсказания
    print(f"\n🔮 ПРЕДСКАЗАНИЯ")
    print("-" * 15)
    
    X_test = df_test.drop(['id', 'datetime']).to_pandas()
    predictions = model.predict(X_test)
    
    # Постобработка
    print("🔧 Постобработка предсказаний...")
    predictions = np.clip(predictions, -3, 3)
    predictions = np.round(predictions)
    predictions = np.where(predictions == 0, 
                          np.where(predictions >= 0, 1, -1), 
                          predictions)
    predictions = np.clip(predictions, -3, 3)
    
    # Анализ результатов
    print(f"\n📊 РЕЗУЛЬТАТЫ")
    print("-" * 12)
    
    print("Распределение предсказаний:")
    unique, counts = np.unique(predictions, return_counts=True)
    for val, count in zip(unique, counts):
        print(f"  {val:2.0f}: {count:6d} ({count/len(predictions)*100:5.1f}%)")
    
    # Сохранение
    submission['target'] = predictions.astype(int)
    submission.to_csv('submission.csv', index=False)
    
    # Важность признаков
    print(f"\n🏆 ТОП-15 ВАЖНЫХ ПРИЗНАКОВ")
    print("-" * 30)
    
    feature_importance = model.get_feature_importance()
    feature_names = X_train.columns
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    for i, row in importance_df.head(15).iterrows():
        print(f"  {row.name+1:2d}. {row['feature']:25s}: {row['importance']:8.2f}")
    
    # Сохранение важности и модели
    importance_df.to_csv('feature_importance.csv', index=False)
    model.save_model('catboost_model.cbm')
    
    print(f"\n🎉 ГОТОВО!")
    print("=" * 20)
    print(f"✅ submission.csv сохранен")
    print(f"✅ feature_importance.csv сохранен")
    print(f"✅ catboost_model.cbm сохранен")
    print(f"📊 Признаков: {X_train.shape[1]}")
    print(f"🚀 GPU: {'Да' if use_gpu else 'Нет'}")
    print(f"🎯 Итераций обучения: {model.get_best_iteration()}")
    
    # Yandex Cloud специфичный вывод
    if YANDEX_ENV:
        print(f"\n📋 YANDEX CLOUD ИНСТРУКЦИИ:")
        print("1. Скачайте submission.csv")
        print("2. Отправьте в соревнование")
        print("3. Сохраните модель для повторного использования")
    
    print(f"\n🏆 Удачи в соревновании!")

if __name__ == "__main__":
    main() 