#!/usr/bin/env python3
"""
Setup script for checking and configuring GPU (RTX 2060) for PyTorch
"""

import os
import sys
import subprocess
import platform
import torch
import yaml

def check_python_version():
    """Check Python version"""
    print("Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro} detected")
        return True
    else:
        print(f"✗ Python {version.major}.{version.minor} detected, need 3.8+")
        return False

def check_pytorch():
    """Check PyTorch installation and GPU"""
    print("\nChecking PyTorch installation...")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__} installed")
        
        # Check CUDA availability
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.version.cuda}")
            print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
            print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            return True
        else:
            print("✗ CUDA not available")
            return False
    except ImportError:
        print("✗ PyTorch not installed")
        return False

def install_pytorch_cuda():
    """Install PyTorch with CUDA support for RTX 2060"""
    print("\nInstalling PyTorch with CUDA support...")
    
    # CUDA 11.8 is stable for RTX 2060
    cmd = [
        sys.executable, "-m", "pip", "install",
        "torch==2.1.0",
        "torchvision==0.16.0",
        "torchaudio==2.1.0",
        "--index-url", "https://download.pytorch.org/whl/cu118"
    ]
    
    try:
        subprocess.check_call(cmd)
        print("✓ PyTorch with CUDA installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("✗ Failed to install PyTorch with CUDA")
        return False

def install_requirements():
    """Install required packages"""
    print("\nInstalling required packages...")
    
    requirements = [
        "python-binance",
        "pandas",
        "numpy",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "ta",  # Technical Analysis library
        "ccxt",
        "python-dotenv",
        "pyyaml",
        "tqdm",
        "optuna",  # For hyperparameter optimization
        "tensorboard",
        "joblib"
    ]
    
    for package in requirements:
        print(f"Installing {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])
            print(f"  ✓ {package} installed")
        except:
            print(f"  ✗ Failed to install {package}")
    
    return True

def configure_gpu_settings():
    """Configure GPU settings for optimal performance"""
    print("\nConfiguring GPU settings for RTX 2060...")
    
    if not torch.cuda.is_available():
        print("✗ GPU not available, skipping configuration")
        return
    
    # RTX 2060 specific optimizations
    gpu_props = torch.cuda.get_device_properties(0)
    total_memory = gpu_props.total_memory / 1e9
    
    # Set optimal memory limit (50% of total for RTX 2060)
    memory_limit = min(50, int(total_memory * 0.5))
    
    config = {
        'gpu': {
            'enabled': True,
            'device_id': 0,
            'memory_limit': memory_limit,
            'precision': 'mixed' if torch.cuda.get_device_capability(0)[0] >= 7 else 'fp32'
        },
        'cpu': {
            'thread_limit': min(4, os.cpu_count() // 2),
            'process_limit': 2
        },
        'memory': {
            'data_cache_size': 2,
            'pin_memory': True
        },
        'optimization': {
            'auto_detect': True,
            'benchmark': True,
            'fallback_cpu': True
        }
    }
    
    # Save configuration
    with open('config_gpu_cpu.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"✓ GPU configuration saved")
    print(f"  - Memory limit: {memory_limit}%")
    print(f"  - Precision: {config['gpu']['precision']}")
    print(f"  - CPU threads: {config['cpu']['thread_limit']}")
    
    # Run benchmark
    if config['optimization']['benchmark']:
        print("\nRunning CUDA benchmark...")
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        
        # Simple benchmark
        import time
        a = torch.randn(1000, 1000).cuda()
        b = torch.randn(1000, 1000).cuda()
        
        start = time.time()
        for _ in range(100):
            c = torch.matmul(a, b)
        torch.cuda.synchronize()
        end = time.time()
        
        print(f"  ✓ CUDA benchmark completed: {(end-start)*1000:.2f}ms for 100 matrix multiplications")

def main():
    """Main setup function"""
    print("""
╔══════════════════════════════════════════════════════════╗
║     Binance AI Scalping Bot - System Setup              ║
║               RTX 2060 Configuration                    ║
╚══════════════════════════════════════════════════════════╝
    """)
    
    # Check Python
    if not check_python_version():
        sys.exit(1)
    
    # Check/Install PyTorch
    if not check_pytorch():
        print("\nPyTorch with CUDA support not found.")
        choice = input("Install PyTorch with CUDA support? (y/n): ")
        if choice.lower() == 'y':
            if install_pytorch_cuda():
                print("Please restart the script after installation.")
                sys.exit(0)
            else:
                print("Failed to install PyTorch with CUDA.")
                sys.exit(1)
        else:
            print("Skipping PyTorch installation.")
    
    # Install requirements
    print("\nInstalling additional requirements...")
    install_requirements()
    
    # Configure GPU
    if torch.cuda.is_available():
        configure_gpu_settings()
    
    # Create .env template
    print("\nCreating .env template...")
    if not os.path.exists('.env'):
        with open('.env', 'w') as f:
            f.write("""# Binance API Credentials
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here

# Optional: Telegram notifications
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_chat_id
""")
        print("✓ .env template created")
    
    # Create necessary directories
    print("\nCreating project directories...")
    dirs = ['models', 'data', 'logs', 'data/historical', 'data/processed']
    for dir_name in dirs:
        os.makedirs(dir_name, exist_ok=True)
        print(f"  ✓ Created {dir_name}/")
    
    print("\n" + "="*50)
    print("✓ Setup completed successfully!")
    print("="*50)
    
    # Final summary
    if torch.cuda.is_available():
        print(f"\nGPU Status: READY")
        print(f"  - Device: {torch.cuda.get_device_name(0)}")
        print(f"  - CUDA Version: {torch.version.cuda}")
        print(f"  - Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("\n⚠️  GPU not available - running in CPU mode")
    
    print("\nNext steps:")
    print("1. Edit .env file and add your Binance API keys")
    print("2. Run 'python main.py' to start the bot")
    print("3. Collect data and train your first model")

if __name__ == "__main__":
    main()