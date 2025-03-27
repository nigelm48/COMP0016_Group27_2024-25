# COMP0016_Group27_2024-25


## How to Run
Operating System: Windows, macOS
Memory: 16GB recommended
Storage: 30GB free space
GPU: NVIDIA with CUDA support or Apple Silicon (for MPS acceleration) is prefered
Installation Steps (CPU and Apple MPS accelerated):

Download the software executable package of your OS.
Extract files and navigate to the directory.
Run the executable.


- Installation Steps (CUDA Accelerated):
- Download and install the latest NVIDIA driver and CUDA toolkit compatible with your GPU from: https://developer.nvidia.com/cuda-downloads
- Clone the project repository:
- git clone https://github.com/nigelm48/COMP0016_Group27_2024-25.git
- Navigate to the project directory:
cd COMP0016_Group27_2024-25
git checkout main
- Manually download the required models from Hugging Face:
Qwen2.5-1.5B
multilingual-e5-small
-Place the downloaded model directories inside the project root directory, maintaining the following structure:

COMP0016_Group27_2024-25/
├── multilingual-e5-small/       # Directory containing the multilingual-e5-small model files
├── Qwen2.5-1.5B/                # Directory containing the Qwen2.5-1.5B model files

                
# Python libraries requirement
All the libraries required are listed in requirement.txt
User can run command: 
pip install -r requirements.txt
to install all the libraries needed.

pip install tkinter
For CUDA: Ensure your system uses a Python version >= 3.9 and your environment supports CUDA.
Run the main application:
python main.py

