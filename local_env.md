virtualenv .venv -p=3.10
source ENV/bin/activate
pip install -e .[base]
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
git clone lerobot 
rm -rf lerobot/.git
cd lerobot
change its version of torchvision to 0.20.1 from 0.21
pip install -e .
cd ..
pip install --no-build-isolation flash-attn==2.7.1.post4
pip install feetech-servo-sdk