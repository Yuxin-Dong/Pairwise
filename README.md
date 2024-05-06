# Towards Generalization beyond Pointwise Learning: A Unified Information-theoretic Perspective (ICML 2024)

## Replicating the experiments

1. Generate the commands for the desired experiment using the `scripts/fcmi_scripts.py` script.
2. Parse the result of the experiment using the `scripts/fcmi_parse_results.py` script.
3. Use the `scripts/pairwise_plots.py` to generate plots from the parsed results.

## Requirements

* Basic libraries such as `numpy`, `scipy`, `tqdm`, `matplotlib`, and `seaborn`.
* We used `Pytorch 1.7.0`, but higher versions should work too.

## Cite

```
@inproceedings{
    dong2024towards,
    title={Towards Generalization beyond Pointwise Learning: A Unified Information-theoretic Perspective},
    author={Yuxin Dong and Tieliang Gong and Hong Chen and Mengxiang Li and Zhongjiang He and Shuangyong Song and Chen Li},
    booktitle={International Conference on Machine Learning},
    year={2024},
    organization={PMLR}
}
```