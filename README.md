* create environment: "python3 -m venv virEn"
* activate env:  "source virEn/bin/activate"
* install ollama,
* pull ollama gemma2: "ollama pull gemma2:9b"
* in virEn install ollama: "pip install ollama"
* in virEn install sentence-transformers: "pip install sentence-transformers"
* check server has nvidia gpu: "lspci | grep -i nvidia"
* check if its working: "nvidia-smi"
* install FAISS with gpu if CUDA working: "pip install faiss-gpu"
* downgrade numpy to an earlier version: "pip install "numpy<2"
* install util: "pip install psutil"

once creating virtual environment with installed programs and gemma2:9b-it has been downloaded and saved into the required dir, 
then run run_all.sh sequentially one at a time, to process each step of the expirment
