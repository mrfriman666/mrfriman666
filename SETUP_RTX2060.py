# setup_rtx2060.py
import os
import sys
import subprocess
import platform
import ctypes
from pathlib import Path

def run_command(cmd):
    """Выполняет команду и возвращает результат"""
    print(f"   Выполняю: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def check_windows_cuda():
    """Проверка CUDA на Windows"""
    print("\n🔍 Проверка CUDA на Windows...")
    
    # Пути к CUDA по умолчанию
    cuda_paths = [
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6",
    ]
    
    found_cuda = False
    for path in cuda_paths:
        if Path(path).exists():
            print(f"✅ Найдена CUDA: {path}")
            found_cuda = True
            
            # Проверка PATH
            cuda_bin = Path(path) / "bin"
            if str(cuda_bin) not in os.environ['PATH']:
                print(f"⚠️  CUDA не в PATH, добавляю...")
                os.environ['PATH'] = str(cuda_bin) + ";" + os.environ['PATH']
            
            # Проверка nvcc
            success, stdout, stderr = run_command(f'"{cuda_bin}\\nvcc.exe" --version')
            if success:
                print(f"✅ nvcc работает: {stdout.split('release')[-1].strip()}")
    
    if not found_cuda:
        print("❌ CUDA не найдена в стандартных путях")
    return found_cuda

def install_cuda_windows():
    """Установка CUDA на Windows"""
    print("\n📥 Установка CUDA 11.8 для Windows...")
    
    # Ссылки для скачивания
    cuda_url = "https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_522.06_windows.exe"
    cudnn_url = "https://developer.nvidia.com/downloads/compute/cudnn/secure/8.6.0/local_installers/11.8/cudnn-windows-x86_64-8.6.0.163_cuda11-archive.zip"
    
    print(f"1. Скачайте CUDA 11.8: {cuda_url}")
    print(f"2. Скачайте cuDNN 8.6: {cudnn_url}")
    print("\nИнструкция установки:")
    print("1. Запустите установщик CUDA")
    print("2. Выберите 'Custom' установку")
    print("3. Оставьте только: CUDA, NVIDIA Drivers")
    print("4. После установки, распакуйте cuDNN в C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8")
    print("5. Перезагрузите компьютер")
    
    input("\nНажмите Enter после установки...")
    return True

def fix_pytorch_installation():
    """Исправление установки PyTorch"""
    print("\n🔧 Исправление установки PyTorch...")
    
    # Удаляем все версии torch
    commands = [
        "pip uninstall torch torchvision torchaudio -y",
        "conda uninstall pytorch torchvision torchaudio -y",
        "pip cache purge"
    ]
    
    for cmd in commands:
        run_command(cmd)
    
    # Определяем какая CUDA установлена
    cuda_version = "cu118"  # По умолчанию для RTX 2060
    
    # Устанавливаем правильную версию
    pip_command = f"pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/{cuda_version}"
    
    print(f"Устанавливаю PyTorch для {cuda_version}...")
    success, stdout, stderr = run_command(pip_command)
    
    if success:
        print("✅ PyTorch установлен успешно")
    else:
        print(f"❌ Ошибка установки: {stderr}")
    
    return success

def configure_environment():
    """Настройка переменных окружения"""
    print("\n⚙️  Настройка переменных окружения...")
    
    env_vars = {
        'CUDA_PATH': r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8',
        'PATH': r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin',
        'CUDA_HOME': r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8',
        'TORCH_CUDA_ARCH_LIST': '7.5',  # Compute capability для RTX 2060
    }
    
    for key, value in env_vars.items():
        if key == 'PATH':
            current_path = os.environ.get('PATH', '')
            if value not in current_path:
                os.environ['PATH'] = value + ';' + current_path
                print(f"✅ Добавлено в PATH: {value}")
        else:
            os.environ[key] = value
            print(f"✅ Установлено {key}={value}")
    
    return True

def test_rtx2060():
    """Тестирование RTX 2060 с PyTorch"""
    print("\n🧪 Тестирование RTX 2060...")
    
    test_code = """
import torch
import sys

print("="*70)
print("ТЕСТ PYTORCH + RTX 2060")
print("="*70)

# Базовая проверка
print(f"PyTorch версия: {torch.__version__}")
print(f"CUDA доступна: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"Количество GPU: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        device_name = torch.cuda.get_device_name(i)
        print(f"\\nGPU {i}: {device_name}")
        
        # Проверка RTX 2060
        if "2060" in device_name:
            print(f"✅ Обнаружен RTX 2060!")
        
        # Информация о GPU
        props = torch.cuda.get_device_properties(i)
        print(f"   Вычислительная способность: {props.major}.{props.minor}")
        print(f"   Общая память: {props.total_memory / 1024**3:.1f} GB")
        print(f"   Мультипроцессоров: {props.multi_processor_count}")
        
        # Простой тест производительности
        print(f"\\n   Тест производительности...")
        device = torch.device(f'cuda:{i}')
        
        # Создаем тестовые тензоры
        size = 4096
        a = torch.randn(size, size, device=device)
        b = torch.randn(size, size, device=device)
        
        # Тест матричного умножения
        import time
        torch.cuda.synchronize()
        start = time.time()
        c = torch.mm(a, b)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        gflops = (2 * size**3) / (elapsed * 1e9)
        print(f"   Матричное умножение {size}x{size}: {elapsed:.3f} сек")
        print(f"   Производительность: {gflops:.1f} GFLOPS")
        
        # Проверка памяти
        memory_allocated = torch.cuda.memory_allocated(i) / 1024**2
        memory_reserved = torch.cuda.memory_reserved(i) / 1024**2
        print(f"   Использовано памяти: {memory_allocated:.1f} MB")
        print(f"   Зарезервировано памяти: {memory_reserved:.1f} MB")
        
        # Очистка
        del a, b, c
        torch.cuda.empty_cache()
else:
    print("❌ CUDA не доступна")
    print("\\nВозможные причины:")
    print("1. Не установлены драйверы NVIDIA")
    print("2. Не установлена CUDA Toolkit")
    print("3. Неправильная версия PyTorch")
    print("4. GPU не поддерживает CUDA")

print("="*70)
"""
    
    # Запускаем тест в отдельном процессе
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(test_code)
        f.flush()
        
        success, stdout, stderr = run_command(f'python "{f.name}"')
        
        if success:
            print(stdout)
        else:
            print(f"❌ Ошибка теста: {stderr}")
        
        # Удаляем временный файл
        os.unlink(f.name)
    
    return success

def main():
    """Главная функция настройки"""
    print("="*70)
    print("⚙️  НАСТРОЙКА RTX 2060 ДЛЯ PYTORCH")
    print("="*70)
    
    # Проверяем ОС
    system = platform.system()
    print(f"Операционная система: {system}")
    
    if system != "Windows":
        print("⚠️  Этот скрипт оптимизирован для Windows")
        print("   Для Linux используйте: nvidia-smi, apt install nvidia-cuda-toolkit")
        return
    
    # 1. Проверка драйверов NVIDIA
    print("\n1️⃣  ПРОВЕРКА ДРАЙВЕРОВ NVIDIA...")
    success, stdout, stderr = run_command("nvidia-smi")
    if not success:
        print("❌ NVIDIA драйверы не найдены")
        print("   Скачайте драйверы с: https://www.nvidia.com/Download/index.aspx")
        print("   Выберите RTX 2060 и вашу Windows версию")
        return
    else:
        print("✅ NVIDIA драйверы обнаружены")
    
    # 2. Проверка CUDA
    print("\n2️⃣  ПРОВЕРКА CUDA...")
    cuda_installed = check_windows_cuda()
    
    if not cuda_installed:
        print("❌ CUDA не установлена")
        if input("Установить CUDA 11.8? (y/n): ").lower() == 'y':
            install_cuda_windows()
            cuda_installed = check_windows_cuda()
    
    # 3. Настройка окружения
    configure_environment()
    
    # 4. Установка/исправление PyTorch
    print("\n3️⃣  НАСТРОЙКА PYTORCH...")
    fix_pytorch_installation()
    
    # 5. Тестирование
    print("\n4️⃣  ТЕСТИРОВАНИЕ...")
    test_rtx2060()
    
    print("\n" + "="*70)
    print("🎉 НАСТРОЙКА ЗАВЕРШЕНА!")
    print("="*70)
    print("\nСледующие шаги:")
    print("1. Перезагрузите компьютер")
    print("2. Запустите: python -c \"import torch; print(torch.cuda.is_available())\"")
    print("3. Если всё работает, запускайте вашу торговую модель")

if __name__ == "__main__":
    main()