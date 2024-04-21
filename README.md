# absa-nlp

## Prerequisite:

- Tensordock 1x GeForce RTX 4090 PCIE 24 GB, 8x AMD EPYC 75F3 vCPUs, 26 GB RAM, 80 GB NVMe SSD, chose pre-installed Jupyter, TensorFlow, Keras, CUDA OS template.
- Check the current CUDA driver and the toolkit to see if theỉr versions are matched to each other:
    - `nvidia-smi`
    - `nvcc -V`
- Pyenv:
    - `curl https://pyenv.run | bash`
    - https://github.com/pyenv/pyenv/wiki#suggested-build-environment
    - `pyenv install 3.9.19`
-  `sudo apt-get install graphviz`
- `pip install poetry`
- Optional: `poetry env use 3.9.19`
- `poetry install`
- `poetry shell`
- Java 1.8+: `sudo apt install default-jre`
- `git clone https://github.com/vncorenlp/VnCoreNLP.git`
- Nvtop:
    ```
    sudo add-apt-repository ppa:flexiondotorg/nvtop
    sudo apt install nvtop
    ```

## After train:

- Download models folder to local:
    `scp -r -P 55400 user@207.189.112.61:/home/user/absa-nlp/models ./models`
