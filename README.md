# Spyre plugin for vLLM

The vLLM Spyre plugin (`vllm-spyre`) is a dedicated backend extension that enables seamless integration of IBM Spyre Accelerator with vLLM. It follows the architecture describes in [vLLM's Plugin System](https://docs.vllm.ai/en/latest/design/plugin_system.html), making it easy to integrate IBM's advanced AI acceleration into existing vLLM workflows.

## Installation

### With Docker

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

### In a local environment

```
# Install vllm
git clone https://github.com/vllm-project/vllm.git
cd vllm
pip install -r requirements-build.txt
export VLLM_TARGET_DEVICE=empty
pip install --no-build-isolation -v -e .

# Install vllm-spyre
cd ..
git clone https://github.com/IBM/vllm-spyre.git
cd vllm-spyre
pip install --no-build-isolation -v -e .
```