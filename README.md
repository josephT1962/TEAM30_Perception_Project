# Set up

- pytorch
- opencv
- numpy
- scipy
- imageio
- tqdm

## Training

To set up,

```bash
mkdir exp
```

## train a model
```bash
python train.py -e resnet50_bs16_2e-5_aug -t t1
```

## Evaluation

```bash
python generate.py -c resnet50_bs16_2e-5_aug_5 -t t1
```

