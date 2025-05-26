#!/usr/bin/env python3
"""
🚀 Yandex Cloud решение для соревнования Clash Royale (ИСПРАВЛЕННАЯ ВЕРСИЯ)
Оптимизировано для Yandex DataSphere и Compute Cloud
Улучшенная установка зависимостей и обработка ошибок
"""

import os
import sys
import subprocess
import time
import warnings
warnings.filterwarnings('ignore')

# Yandex Cloud специфичные настройки
YANDEX_ENV = '/home/jupyter' in os.getcwd() or 'DATASPHERE' in os.environ

def robust_install(package_name, alternative_names=None, pip_args=None):
    """
    Надежная установка пакета с несколькими попытками
    """
    if alternative_names is None:
        alternative_names = []
    if pip_args is None:
        pip_args = []
    
    packages_to_try = [package_name] + alternative_names
    
    for package in packages_to_try:
        try:
            print(f"🔄 Попытка установки {package}...")
            cmd = [sys.executable, '-m', 'pip', 'install', package] + pip_args
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print(f"✅ {package} установлен успешно")
                return True
            else:
                print(f"⚠️  Ошибка установки {package}: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            print(f"⏰ Timeout при установке {package}")
        except Exception as e:
            print(f"❌ Исключение при установке {package}: {e}")
    
    return False

def install_dependencies_robust():
    """
    Надежная установка зависимостей для Yandex Cloud
    """
    print("🔧 НАДЕЖНАЯ УСТАНОВКА ЗАВИСИМОСТЕЙ")
    print("=" * 45)
    
    # Обновляем pip сначала
    print("📦 Обновляем pip...")
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'], 
                      capture_output=True, timeout=120)
    except:
        print("⚠️  Не удалось обновить pip, продолжаем...")
    
    # Проверяем базовые пакеты
    try:
        import pandas, numpy
        print("✅ Pandas и NumPy уже установлены")
    except ImportError:
        print("📦 Устанавливаем базовые пакеты...")
        robust_install('pandas numpy', pip_args=['--quiet'])
    
    # Устанавливаем остальные пакеты
    packages = [
        ('polars', ['polars>=0.20.0', 'polars>=0.19.0', 'polars']),
        ('requests', ['requests>=2.25.0', 'requests']),
        ('scikit-learn', ['scikit-learn>=1.3.0', 'scikit-learn>=1.0.0', 'scikit-learn']),
        ('catboost', ['catboost>=1.2.0', 'catboost>=1.1.0', 'catboost'])
    ]
    
    failed_packages = []
    
    for package_name, alternatives in packages:
        print(f"\n📦 Устанавливаем {package_name}...")
        success = robust_install(alternatives[0], alternatives[1:], ['--quiet', '--no-cache-dir'])
        
        if not success:
            print(f"❌ Не удалось установить {package_name}")
            failed_packages.append(package_name)
        
        # Небольшая пауза между установками
        time.sleep(1)
    
    if failed_packages:
        print(f"\n⚠️  Не удалось установить: {', '.join(failed_packages)}")
        print("💡 Попробуем альтернативные методы...")
        
        # Альтернативная установка CatBoost
        if 'catboost' in failed_packages:
            print("🔄 Альтернативная установка CatBoost...")
            alternatives = [
                'catboost --no-deps',
                'catboost --force-reinstall',
                'https://files.pythonhosted.org/packages/source/c/catboost/catboost-1.2.tar.gz'
            ]
            
            for alt in alternatives:
                if robust_install(alt):
                    failed_packages.remove('catboost')
                    break
    
    return failed_packages

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
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
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
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("⚠️  nvidia-smi не найден или timeout")
    
    return False

def download_data_yandex():
    """
    Загружает данные для Yandex Cloud с улучшенной обработкой ошибок
    """
    print("📥 ЗАГРУЗКА ДАННЫХ")
    print("-" * 20)
    
    required_files = ['train.csv', 'test.csv', 'submission_example.csv']
    
    # Проверяем, есть ли файлы уже
    files_exist = all(os.path.exists(file) for file in required_files)
    
    if files_exist:
        print("✅ Файлы уже существуют")
        return True
    
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
                    try:
                        import shutil
                        shutil.copy2(file_path, file)
                    except Exception as e:
                        print(f"⚠️  Ошибка копирования {file}: {e}")
    
    # Проверяем еще раз
    files_exist = all(os.path.exists(file) for file in required_files)
    
    if not files_exist:
        print("📥 Загружаем данные из интернета...")
        
        # Пробуем разные методы загрузки
        urls = [
            "http://devopn.ru:8000/cu-base-project.zip",
            "https://github.com/renat2006/ai-clash/raw/main/cu-base-project.zip"
        ]
        
        for url in urls:
            try:
                print(f"🌐 Пробуем загрузить с {url}...")
                
                # Используем requests если доступен
                try:
                    import requests
                    response = requests.get(url, timeout=60)
                    response.raise_for_status()
                    
                    import zipfile
                    from io import BytesIO
                    
                    print("📦 Распаковываем архив...")
                    with zipfile.ZipFile(BytesIO(response.content)) as zip_ref:
                        zip_ref.extractall()
                    
                    print("✅ Данные успешно загружены!")
                    return True
                    
                except ImportError:
                    # Fallback на wget/curl
                    print("🔄 Используем wget...")
                    result = subprocess.run(['wget', '-O', 'data.zip', url], 
                                          capture_output=True, timeout=120)
                    if result.returncode == 0:
                        subprocess.run(['unzip', '-o', 'data.zip'], capture_output=True)
                        os.remove('data.zip')
                        print("✅ Данные загружены через wget!")
                        return True
                        
            except Exception as e:
                print(f"❌ Ошибка загрузки с {url}: {e}")
                continue
        
        # Если ничего не сработало, создаем демо файлы
        print("⚠️  Не удалось загрузить данные, создаем демо файлы...")
        create_demo_files()
        return True
    
    return True

def create_demo_files():
    """
    Создает демо файлы для тестирования
    """
    print("🔧 Создание демо файлов...")
    
    import pandas as pd
    import numpy as np
    
    # Создаем минимальный train.csv
    demo_data = {
        'id': range(1000),
        'datetime': ['20240101T120000.000Z'] * 1000,
        'gamemode': ['Classic'] * 500 + ['Tournament'] * 500,
        'player_1_tag': [f'#TAG{i}' for i in range(1000)],
        'player_2_tag': [f'#TAG{i+1000}' for i in range(1000)],
        'player_1_trophies': np.random.randint(1000, 8000, 1000),
        'player_2_trophies': np.random.randint(1000, 8000, 1000),
        'target': np.random.choice([-3, -2, -1, 1, 2, 3], 1000)
    }
    
    # Добавляем карты
    for i in range(1, 9):
        demo_data[f'player_1_card_{i}'] = np.random.randint(1, 15, 1000)
        demo_data[f'player_2_card_{i}'] = np.random.randint(1, 15, 1000)
    
    pd.DataFrame(demo_data).to_csv('train.csv', index=False)
    
    # Создаем test.csv (без target)
    test_data = demo_data.copy()
    del test_data['target']
    test_data['id'] = range(1000, 1500)
    pd.DataFrame(test_data).to_csv('test.csv', index=False)
    
    # Создаем submission_example.csv
    submission_data = {
        'id': range(1000, 1500),
        'target': [1] * 500
    }
    pd.DataFrame(submission_data).to_csv('submission_example.csv', index=False)
    
    print("✅ Демо файлы созданы")

def main():
    """
    Основная функция для Yandex Cloud
    """
    print("🚀 YANDEX CLOUD CLASH ROYALE SOLUTION (FIXED)")
    print("=" * 55)
    
    # Информация о среде
    print(f"🐍 Python: {sys.version}")
    print(f"📁 Рабочая директория: {os.getcwd()}")
    
    if YANDEX_ENV:
        print("✅ Запуск в Yandex Cloud")
    else:
        print("⚠️  Запуск вне Yandex Cloud")
    
    # Установка зависимостей
    failed_packages = install_dependencies_robust()
    
    # Проверяем критические пакеты
    critical_missing = []
    
    try:
        import pandas as pd
        import numpy as np
        print("✅ Pandas и NumPy импортированы")
    except ImportError as e:
        print(f"❌ Критическая ошибка: {e}")
        critical_missing.append('pandas/numpy')
    
    try:
        import polars as pl
        print("✅ Polars импортирован")
    except ImportError:
        print("⚠️  Polars недоступен, используем Pandas")
        # Fallback на pandas
        import pandas as pd
        pl = None
    
    try:
        from catboost import CatBoostRegressor
        print("✅ CatBoost импортирован")
        catboost_available = True
    except ImportError:
        print("❌ CatBoost недоступен")
        catboost_available = False
        
        # Пробуем альтернативы
        try:
            from sklearn.ensemble import GradientBoostingRegressor
            print("✅ Используем sklearn GradientBoosting как альтернативу")
        except ImportError:
            critical_missing.append('catboost/sklearn')
    
    if critical_missing:
        print(f"❌ Критические пакеты недоступны: {critical_missing}")
        print("💡 Попробуйте:")
        print("1. Перезапустить kernel")
        print("2. Использовать другую конфигурацию VM")
        print("3. Установить пакеты вручную")
        return
    
    # Проверка GPU
    use_gpu = check_yandex_gpu() and catboost_available
    
    # Загрузка данных
    if not download_data_yandex():
        print("❌ Не удалось загрузить данные")
        return
    
    print("\n📊 ЗАГРУЗКА И АНАЛИЗ ДАННЫХ")
    print("-" * 35)
    
    # Загружаем данные
    if pl is not None:
        try:
            df_train = pl.read_csv('train.csv')
            df_test = pl.read_csv('test.csv')
            submission = pd.read_csv('submission_example.csv')
            use_polars = True
        except Exception as e:
            print(f"⚠️  Ошибка Polars: {e}, используем Pandas")
            use_polars = False
    else:
        use_polars = False
    
    if not use_polars:
        df_train = pd.read_csv('train.csv')
        df_test = pd.read_csv('test.csv')
        submission = pd.read_csv('submission_example.csv')
    
    print(f"📈 Train shape: {df_train.shape}")
    print(f"📉 Test shape: {df_test.shape}")
    print(f"📋 Submission shape: {submission.shape}")
    
    # Упрощенный feature engineering для совместимости
    def create_simple_features(df, is_train=True):
        """
        Упрощенное создание признаков для максимальной совместимости
        """
        print(f"🔧 Создание признаков ({'train' if is_train else 'test'})...")
        
        if use_polars:
            # Polars версия
            df = df.with_columns([
                (pl.col('player_1_trophies') - pl.col('player_2_trophies')).alias('trophy_diff'),
                (pl.col('player_1_trophies') + pl.col('player_2_trophies')).alias('trophy_sum'),
                (pl.col('player_1_trophies') / (pl.col('player_2_trophies') + 1)).alias('trophy_ratio')
            ])
            
            # Карточные признаки
            card_cols_p1 = [f'player_1_card_{i}' for i in range(1, 9)]
            card_cols_p2 = [f'player_2_card_{i}' for i in range(1, 9)]
            
            df = df.with_columns([
                pl.mean_horizontal(card_cols_p1).alias('player_1_avg_card'),
                pl.mean_horizontal(card_cols_p2).alias('player_2_avg_card')
            ])
            
            df = df.with_columns([
                (pl.col('player_1_avg_card') - pl.col('player_2_avg_card')).alias('avg_card_diff')
            ])
            
        else:
            # Pandas версия
            df = df.copy()
            df['trophy_diff'] = df['player_1_trophies'] - df['player_2_trophies']
            df['trophy_sum'] = df['player_1_trophies'] + df['player_2_trophies']
            df['trophy_ratio'] = df['player_1_trophies'] / (df['player_2_trophies'] + 1)
            
            # Карточные признаки
            card_cols_p1 = [f'player_1_card_{i}' for i in range(1, 9)]
            card_cols_p2 = [f'player_2_card_{i}' for i in range(1, 9)]
            
            df['player_1_avg_card'] = df[card_cols_p1].mean(axis=1)
            df['player_2_avg_card'] = df[card_cols_p2].mean(axis=1)
            df['avg_card_diff'] = df['player_1_avg_card'] - df['player_2_avg_card']
        
        print(f"✅ Создано признаков. Размерность: {df.shape}")
        return df
    
    # Создание признаков
    print("\n🔧 FEATURE ENGINEERING")
    print("-" * 25)
    
    df_train = create_simple_features(df_train, is_train=True)
    df_test = create_simple_features(df_test, is_train=False)
    
    # Подготовка к обучению
    print("\n🤖 ПОДГОТОВКА К ОБУЧЕНИЮ")
    print("-" * 30)
    
    # Категориальные признаки
    cat_features = ['gamemode', 'player_1_tag', 'player_2_tag'] + \
                   [f'player_1_card_{i}' for i in range(1, 9)] + \
                   [f'player_2_card_{i}' for i in range(1, 9)]
    
    print(f"📊 Категориальных признаков: {len(cat_features)}")
    
    # Подготовка данных
    if use_polars:
        X_train = df_train.drop(['id', 'datetime', 'target']).to_pandas()
        y_train = df_train['target'].to_pandas()
        X_test = df_test.drop(['id', 'datetime']).to_pandas()
    else:
        X_train = df_train.drop(['id', 'datetime', 'target'], axis=1)
        y_train = df_train['target']
        X_test = df_test.drop(['id', 'datetime'], axis=1)
    
    print(f"📏 Размерность: {X_train.shape}")
    
    # Настройка модели
    if catboost_available:
        model_params = {
            'cat_features': cat_features,
            'random_state': 52,
            'verbose': 100,
            'iterations': 1000 if use_gpu else 500,
            'learning_rate': 0.1,
            'depth': 6
        }
        
        if use_gpu:
            model_params.update({
                'task_type': 'GPU',
                'devices': '0'
            })
            print("🚀 Используем GPU для обучения")
        else:
            model_params['task_type'] = 'CPU'
            print("💻 Используем CPU для обучения")
        
        model = CatBoostRegressor(**model_params)
    else:
        # Fallback на sklearn
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.preprocessing import LabelEncoder
        
        print("💻 Используем sklearn GradientBoosting")
        
        # Кодируем категориальные признаки
        le_dict = {}
        for col in cat_features:
            if col in X_train.columns:
                le = LabelEncoder()
                X_train[col] = le.fit_transform(X_train[col].astype(str))
                X_test[col] = le.transform(X_test[col].astype(str))
                le_dict[col] = le
        
        model = GradientBoostingRegressor(
            n_estimators=500,
            learning_rate=0.1,
            max_depth=6,
            random_state=52,
            verbose=1
        )
    
    # Обучение
    print(f"\n⏳ ОБУЧЕНИЕ МОДЕЛИ")
    print("-" * 20)
    
    model.fit(X_train, y_train)
    
    # Предсказания
    print(f"\n🔮 ПРЕДСКАЗАНИЯ")
    print("-" * 15)
    
    predictions = model.predict(X_test)
    
    # Постобработка
    print("🔧 Постобработка предсказаний...")
    predictions = np.clip(predictions, -3, 3)
    predictions = np.round(predictions)
    predictions = np.where(predictions == 0, 
                          np.where(predictions >= 0, 1, -1), 
                          predictions)
    predictions = np.clip(predictions, -3, 3)
    
    # Сохранение
    submission['target'] = predictions.astype(int)
    submission.to_csv('submission.csv', index=False)
    
    print(f"\n🎉 ГОТОВО!")
    print("=" * 20)
    print(f"✅ submission.csv сохранен")
    print(f"📊 Признаков: {X_train.shape[1]}")
    print(f"🚀 GPU: {'Да' if use_gpu else 'Нет'}")
    print(f"🤖 Модель: {'CatBoost' if catboost_available else 'sklearn'}")
    
    print(f"\n🏆 Удачи в соревновании!")

if __name__ == "__main__":
    main() 
