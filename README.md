# Language Model

This repository provides classes to train and do inference with a GPT-2 type of language model. The trained model can autogressively generate text.

## Requirements
Create a virtual environment with `python==3.12`. You can create a predefined conda environment with

```conda env create -f utils/create_env.yml```

Open the environment (e.g. `conda activate TinyStories`) and install the required packages via ``pip``.

```pip install -r utils/requirements.txt```

## Train the model

To train a language model simply run 

```python tinystories_train.py```

This runs a predefined model. The model architecture can be directly adjusted in the main script with `model_confing` and the training with `train_config`.

The tiny story dataset is obtained from Huggingface.

## Generate text

To generate text a pretrained model is required. Simply run

```python tinystories_inference.py -c experiments/250804_1522_baseline/final_model.pth -p "Lisa and Tim"```

for a model checkpoint saved in `experiments/250804_1522_baseline/final_model.pth` and a story that is supposed to start with `Lisa and Tim`. Further arguments can be parsed in the command line that are given in the script `tinystories_inference.py`.

Alternatively, you can run the jupyter notebook `tinystory_notebook.ipynb`.

## Licence
MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.