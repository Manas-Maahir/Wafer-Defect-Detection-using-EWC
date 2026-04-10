# Wafer Defect Classification with Continual Learning
## 👥 Contributors
- [Manas Maahir](https://github.com/Manas-Maahir)
- [Jay Krishna](https://github.com/JayKrishna05)
- [Kartheek Yadav](https://github.com/KartheekYadav87)
## What is this project?

Semiconductor manufacturing is an extremely precise process. You start with a silicon wafer, you run it through hundreds of process steps, and at the end you hope that the tiny circuits etched on it actually work. When something goes wrong during manufacturing, it does not go wrong randomly. Defects on wafers tend to cluster in specific geometric patterns that reflect the specific process failure that caused them. An edge ring defect means something went wrong at the outer edge of the wafer during a deposition step. A scratch defect looks like a straight line and is usually caused by a mechanical handling issue. A center defect concentrates right in the middle and often points to an issue with process uniformity.

The goal of this project is to look at a wafer map, which is essentially a 2D grid showing which dies on the wafer passed or failed, and automatically classify which failure pattern is present. There are 9 classes in total: Center, Donut, Edge-Loc, Edge-Ring, Loc, Random, Scratch, Near-full, and none (meaning no defect pattern detected).

If you can do this automatically and accurately, you save engineers enormous amounts of time. Instead of manually looking at thousands of wafer maps every day, you let the model do the first pass and flag the interesting ones.

## The Dataset

We use the LSWMD (Large-Scale Wafer Map Dataset), which is one of the most widely used public datasets for this problem. It contains tens of thousands of real wafer maps from actual fabrication processes. The raw data lives in a pickle file (LSWMD.pkl), and we have preprocessed it into a memory-mapped numpy array (polar_strips.npy) alongside a labels.csv.

The memory-mapped format is important here. The full dataset is multiple gigabytes. Loading all of it into RAM at once would either crash your machine or leave you waiting a long time. Memory mapping lets you access the data as if it is in memory while actually reading it from disk on demand. It is a nice trick that makes the data pipeline much smoother.

## Why Not Just Throw a CNN at It?

You could. In fact, a plain ResNet trained on raw wafer maps works reasonably well. But there are a few reasons why we went further.

The first reason is geometry. Defect patterns on a wafer are fundamentally circular. Edge-Ring defects wrap around the entire perimeter. Center defects are radially symmetric. A standard CNN sees a rectangular grid and has no built-in awareness that the important structure is actually circular. It will learn this eventually, but it takes more data and more compute to get there.

So we do something clever in the preprocessing step. We convert each wafer map from Cartesian coordinates to a polar coordinate representation. Specifically, we extract the edge ring of the wafer (the annular region near the periphery), and we unroll it into a rectangular strip. The horizontal axis becomes the angle around the wafer (theta, from 0 to 360 degrees), and the vertical axis becomes the radial depth from the edge inward. After this transform, an Edge-Ring defect that wraps around the entire wafer becomes a horizontal stripe across the full width of the strip. A localized Edge-Loc defect becomes a vertical blob in a specific angular position. The geometry that was hard to see in Cartesian coordinates becomes obvious in polar coordinates.

This preprocessing step is done once offline and saved to disk. During training, the model only ever sees these polar strips.

## The Model Architecture

Once we have the polar strip, we run it through a hybrid CNN plus Transformer model.

The CNN part is a truncated ResNet-18. We only use the first two blocks of ResNet-18, which is enough to extract low-level spatial features like edges and textures without being too deep or too slow. The input is a single-channel grayscale image (the polar strip), so we replace the first convolution layer to accept 1 channel instead of the standard 3.

The CNN output is a feature map with 128 channels. We then adapt it to 3 channels using a simple 1x1 convolution and resize it to 224x224, which is the input size that the Swin Transformer expects.

The Swin Transformer (specifically Swin Tiny, Patch 4, Window 7) then processes these adapted features. Swin Transformers are particularly good at capturing long-range dependencies in images. This matters for wafer defects because some patterns, like a Scratch, span a large portion of the image and require the model to understand relationships between regions far apart in the image. Attention mechanisms handle this naturally in a way that local convolutions cannot.

The final Swin output (a 768-dimensional feature vector) goes into a linear classifier that produces probabilities for each of the 9 defect classes.

## The Problem We Were Running Into: Catastrophic Forgetting

Here is where things get interesting, and also a bit frustrating.

In the real world, you do not train a model once and deploy it forever. Fabs change processes. New defect types emerge. You get new batches of labeled data every few months. The natural thing to do is to take your existing model and fine-tune it on the new data. This is called continual learning or lifelong learning.

The problem is something called catastrophic forgetting. When you fine-tune a neural network on new data, it tends to completely forget what it learned before. The gradient updates that push the network to learn the new task essentially overwrite the weights that were important for the old task. You end up with a model that is great at the new stuff and terrible at everything it used to know. This is not a minor degradation, it can be a near-total collapse of performance on old tasks.

For a production fab, this is a serious problem. You cannot ship a model that suddenly misclassifies your oldest and most common defect patterns just because you updated it with new data from last quarter.

## EWC: Elastic Weight Consolidation

This is where EWC comes in. It was introduced in a 2017 paper from DeepMind titled "Overcoming Catastrophic Forgetting in Neural Networks" by Kirkpatrick et al., and it is one of the more elegant ideas in continual learning.

The core intuition is this: not all parameters in a neural network are equally important to what it has already learned. Some weights are critical and sensitive, small changes to them will destroy performance on the old task. Other weights are more flexible and can be changed significantly without affecting the old behavior much. If you could identify which weights are important and protect them during fine-tuning, you could learn new tasks without forgetting old ones.

EWC operationalizes this using something called the Fisher Information Matrix. The Fisher Information tells you how sensitive the model's output is to changes in each parameter. A parameter with high Fisher Information is one that the model relies on heavily, changing it a little causes big changes in what the model predicts. A parameter with low Fisher Information is one that barely affects the outputs and can be changed freely.

The Fisher Information Matrix is technically a large square matrix (as big as the number of parameters by the number of parameters), which for a modern neural network is completely infeasible to compute or store. So the practical trick is to only compute the diagonal of this matrix. Each parameter gets its own scalar Fisher value, and we treat them as independent. This diagonal approximation is efficient to compute and works well in practice.

Here is what EWC does in training:

After training on the original task (or at any checkpoint you designate as the "prior task"), you do one forward pass over the training data. For each parameter, you compute the squared gradient of the loss with respect to that parameter, averaged over all the data. These squared gradients are your Fisher Information diagonal estimates. They tell you how much each parameter matters.

You also save a snapshot of the current parameter values, call them theta-star. These are the values the model settled on after learning the task you want to protect.

Then, when you fine-tune on new data, you add a regularization term to the loss. This term penalizes changes to important parameters. Specifically, for each parameter, you compute the squared difference between the current value and the saved theta-star, weighted by the Fisher Information for that parameter. High Fisher weight means high penalty for changing that parameter. Low Fisher weight means you can change it freely.

The full loss during fine-tuning is the task loss plus lambda times the EWC penalty, where lambda is a hyperparameter that controls how strongly you want to protect the old task. Set lambda too low and you still forget. Set it too high and you cannot learn anything new.

## How EWC is Wired Into This Codebase

The EWC logic lives in continual_learning.py. The EWC class takes the model, a dataloader, and a device. When you call register_prior_task(), it snapshots the current weights and runs one pass to compute the Fisher diagonals.

In train.py, we instantiate EWC before the training loop starts and pass it into each training step via selective_ewc_loss(). This function adds the EWC penalty to the cross-entropy loss before the backward pass.

The lambda we use is 500 by default. This is a moderate value. If you are doing aggressive continual learning across many tasks, you may need to tune this. Higher values protect old tasks more strongly but constrain new learning more heavily.

One thing worth noting is that the EWC setup here is designed for the scenario where you are protecting one prior task state. If you want to extend this to a true sequential multi-task scenario (train task 1, register, train task 2, register, train task 3, etc.), you would call register_prior_task() again after each task. The Fisher Information and means will be updated to reflect the most recent prior task.

## Other Improvements in the Training Pipeline

A few other things were fixed or added beyond EWC:

**Train/Validation Split.** Previously, the accuracy reported during training was measured on the training data itself. That is meaningless as a measure of generalization. The model could have just memorized the training data and you would never know. We now split 15% of the data aside as a validation set and report validation accuracy at the end of each epoch. This gives you a real signal.

**Class Weighting.** The LSWMD dataset is heavily imbalanced. The "none" class (no defect) is far more common than rare patterns like Donut or Near-full. Without addressing this, the model will tend to predict the majority class and get away with decent overall accuracy while being completely useless on rare defects. We compute inverse frequency class weights and pass them to CrossEntropyLoss, which pushes the model to pay proportional attention to all classes.

**Gradient Clipping.** During early training, gradients can become very large and cause unstable weight updates. We clip the gradient norm to 1.0 before each optimizer step. This costs almost nothing in compute time and makes training noticeably more stable.

**Cosine Annealing Learning Rate Schedule.** Instead of a fixed learning rate, we reduce it smoothly over the course of training following a cosine curve. This tends to give better final performance because the model can take larger steps early when it is far from a good solution and smaller, more refined steps later.

**Dynamic Dataset Shape Inference.** The dataset loader was previously hardcoded to expect exactly 33519 samples. If the dataset size ever changes, this would silently fail or crash in confusing ways. We now infer the actual number of samples from the file size, which is more robust.

**Preprocessing Robustness.** The polar conversion preprocessing step would crash outright if given a completely empty wafer map (all zeros). This can happen with corrupted or missing data entries. We added a fallback that returns a zero strip instead of raising an exception.

**Checkpoint Compatibility.** The visualization script was loading checkpoints with a bare torch.load() call that assumed a specific format. It now handles both raw state dictionaries and the fuller checkpoint format (which includes optimizer state, epoch number, and best accuracy) used by the training loop.

## File Structure

```
Wafer/
├── train.py               Main training loop with EWC
├── model.py               CNN + Swin Transformer hybrid model
├── continual_learning.py  EWC implementation
├── dataset_memmap.py      Memory-mapped dataset loader
├── data_loader.py         Original task-split data loader (for sequential continual learning)
├── preprocessing.py       Cartesian to polar transform
├── visualize_attention.py Saliency map visualization
├── polar_strips.npy       Preprocessed wafer strips (memmap)
├── labels.csv             Labels for each sample
├── checkpoints/           Saved model weights
└── wafer_gpu_env/         Virtual environment
```

## How to Run

**Step 1: Activate the environment**

PowerShell:
```
.\wafer_gpu_env\Scripts\Activate.ps1
```

CMD:
```
wafer_gpu_env\Scripts\activate.bat
```

If PowerShell blocks script execution, run this first:
```
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
```

**Step 2: Train the model**

```
python train.py
```

Training will print validation accuracy at the end of each epoch. The best model is saved to checkpoints/best_model.pth and the most recent checkpoint (for resuming) is saved to checkpoints/last_model.pth. If you stop training and restart, it picks up exactly where it left off.

**Step 3: Visualize attention (optional)**

```
python visualize_attention.py
```

This generates a wafer_attention.png showing what the model is paying attention to when it makes a prediction.

## Hyperparameters Worth Knowing

| Parameter | Default | What it does |
|---|---|---|
| BATCH_SIZE | 32 | Samples per gradient update |
| EPOCHS | 8 | Full passes over the training data |
| LR | 1e-4 | Starting learning rate |
| VAL_SPLIT | 0.15 | Fraction of data held out for validation |
| EWC_LAMBDA | 500 | How strongly to protect prior task weights |
| GRAD_CLIP | 1.0 | Maximum gradient norm |

If you are doing multi-task continual learning and finding the model forgets too fast, increase EWC_LAMBDA. If it learns new tasks too slowly, decrease it. There is always a tradeoff and it depends on how different your tasks are from each other.
