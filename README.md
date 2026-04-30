# Stage_M2FESUP_SATIE_CNRS
sudo apt update --fix-missing
python3 -m venv .venv                                                                                                                      
source .venv/bin/activate
pip install --upgrade pip                                                                                                                  
pip install -r requirements.txt


pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
