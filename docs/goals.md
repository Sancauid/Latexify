## Data

Have a notebook that describes the following stuff
- What does the data look like:
    - the image of the formula (quality, size)
    - the latex formula corresponds to the image
    - diversity of the data (count the number of the occurances of the integral, derivative, trigonometry functions)
    - length (min, max, mean, std) of the formulas (characters)
    - number of words
    - multilined?
- How to load the data: design a dataloader class (https://pytorch.org/tutorials/beginner/basics/data_tutorial.html) that loads the images/text
    - Normalization -> only a small gain
    - Augmentations?
- Any other interesting things...

## Experimentation

### Model Impl

#### Encoder

1. ResNet
    1. GC Block?
2. ConvNext
3. SwinTransformer

#### Decoder

1. Transformer
    1. Encoder + Decoder
    2. Decoder Only

2. LSTM (if we have time)
    1. Soft Attention
    2. LSTM Only

### Loss Function

1. Cross Entropy Loss

> Perplexity: We don't need it because it is equal to exp(cross-entropy loss).

### Tokenization Strategy

1. Character-based (if we have time)
    1. May lead to better generalizability (if the token is not shown in the train set before)
2. Latex token based (prefered)

### Hyperparameters

> if we have time

1. Size/Number of the encoder layers
2. Number of transformer block...
3. Momentum of Adam Optimizer

## Analysis

### Metrics (See Paper 2 Section 3)

1. Corpus BLEU (4 grams)
2. Visual Match
    1. An exact visual match score
        1. paper 3 mentions this in section 6
        2. paper 1 mentions this in p5
    2. Image Edit Distance Accuracy (EDA)
3. Syntactically correct sequences
    1. Simply report the percentage is enough

### Ablation Study

### Visualization

> if we have time

1. Visualize where the model attends to for each token.

### Present Generated Data

1. Input => Correct/Generated Sequence Length
2. Why the generated data fails?
    1. Common Errors

### Performance Analysis

How the following affects the performance

1. Different Architecture (Encoder, Decoder)
2. Different Loss Function
3. Larger Dataset
4. Performance Improvement vs length of the formula (Paper 1)

> Can we find a model that will run fast and well on a laptop?