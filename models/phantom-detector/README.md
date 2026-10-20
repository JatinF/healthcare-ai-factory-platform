# phantom-detector

Triton model repository for the synthetic geometric detector. `config.pbtxt` asks for one GPU instance. The Python backend repeats the same bright-pixel fraction as `hafactory.detector`.

This repository has not started Triton and has not measured GPU latency. Point Triton at this directory only on the cluster under test:

```bash
tritonserver --model-repository=/models
```

Set `HAF_TRITON_URL` only after that server answers. The API fail-closes when the URL is set and this process cannot show a measured Triton result, and it fail-closes when fault injection marks the backend down. The unit tests use the local CPU detector and label the runtime `local-cpu`.

TensorRT and ONNX Runtime are the production acceleration options for a later clinical model. They are not wired to this fixture, and there is no parity measurement. NVIDIA NIM stays disabled in the Helm values. A hosted NIM endpoint is out of policy for clinical text.
