# Spyre plugin for vLLM

This plugin enable support for IBM Spyre accelerator in [vLLM](https://docs.vllm.ai/en/latest/).

## Installation from source with Docker

First, download vllm-spyre

```
git clone https://github.com/IBM/vllm-spyre
cd vllm-spyre
```

Build image from source

```
docker build . -f Dockerfile.spyre -t vllm-spyre
docker run -it --rm vllm-spyre bash
```
