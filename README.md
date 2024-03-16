# torch-mnist-bench

Benchmark on MNIST dataset using PyTorch, comparing with [test-candle-rs](https://github.com/yyhhenry/test-candle-rs).

## Training benchmark

Tested on a single NVIDIA RTX 3050 Laptop GPU.

For python, total time does not include the time for importing libraries.

|Python|Train/epoch|Eval/epoch|Total|
|---|---|---|---|
|`train_with_dataloader.py`|6.5s|0.95s|/|
|`train_with_dataloader.py --linear`|5s|0.85s|/|
|`train.py`|1.4s|0.07s|29.37s|
|`train.py --no_cudnn`|11.4s|0.7s|/|
|`train.py --linear`|0.04s|<0.01s|2.5s|
|`train.py --no_cudnn --linear`|~0.04s|<0.01s|2.93s|
---
|Rust|Train/epoch|Eval/epoch|Total|
|---|---|---|---|
|`train.rs` without cudnn|~9s|~0.25s|/|
|`train.rs`|~14s|~0.1s|/|
|`train.rs --linear`|/|/|~6.5s|
