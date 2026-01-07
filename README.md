# Pixel-Play
Pixel play '26  VLG IITR Submission

[![PDF Report](https://img.shields.io/badge/Report-PDF-red)](Pixel_Play.pdf)

This repository contains my complete work for **unsupervised video anomaly detection** on the corrupted **CUHK Avenue dataset**, developed as part of the kaggle **Pixel Play ’26 competition**.

The project progressively builds from a simple convolutional autoencoder to a motion-aware  
**CNN + ConvLSTM + frame-difference** model, followed by post-processing techniques that significantly improve mean Average Precision (mAP).

**Final Best mAP: 0.6618**

##  Repository Structure
```text
├── models/
│   ├── baseline_cae.pth
│   ├── cnn_convlstm_epoch2.pth
│   ├── cnn_convlstm_epoch3.pth
│   └── cnn_convlstm_framediff_96_epoch3.pth
│
├── baseline-autoencoder.ipynb
├── cnn-cnnlstm.ipynb
├── cnn-lstm-framediff-visualisationtools.ipynb
├── cnnlstmrgb.ipynb
├── mat-cnnlstm.ipynb
│
├── Pixel_Play.pdf
└── README.md

```

**NOTE** Run all the notebooks on kaggle 2 T4 GPU
<img width="650" height="200" alt="image" src="https://github.com/user-attachments/assets/a0bb89e2-614f-4991-84f1-6e03c33089cd" />

## Problem Overview
The task is frame-level anomaly detection in surveillance videos under the following constraints:
- Only *normal* videos are available during training
- Anomalies are rare, diverse, and unlabeled
- Data has corruption TV static type noise and flipped images in the testing set

<img width="968" height="301" alt="Screenshot 2026-01-08 at 12 34 26 AM" src="https://github.com/user-attachments/assets/b2c33034-b7bf-4e44-8396-8e526e7c975a" />

## Data Preprocessing

Key preprocessing steps applied to all models:
- Grayscale conversion (motion-focused)
- Bilateral filtering to suppress TV-static noise
- Automatic vertical flip correction
- Resize frames to **192 × 192**
- Normalize pixel values to `[0, 1]`
  
<img width="721" height="214" alt="Screenshot 2026-01-08 at 12 36 15 AM" src="https://github.com/user-attachments/assets/3626c7ca-3ded-45ad-b4d0-510de5ba4fde" />

## Models
### 1. Baseline Convolutional Autoencoder (CAE)
**Notebook**: baseline-autoencoder.ipynb
**Checkpoint**: models/baseline_cae.pth

```python
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out

```
### Loss function and the Optimizer 
```python
loss = MSE(reconstructed_frame, input_frame)
optimizer = torch.optim.Adam(model.parameters(), lr=LR) #LR is 0.0001
```

### Result
Map: **0.45**

### 2. CNN + ConvLSTM
**Notebook**: cnn-cnnlstm.ipynb
**Checkpoint**: cnn_convlstm_epoch2.pth, cnn_convlstm_epoch3.pth
<img width="487" height="417" alt="Screenshot 2026-01-08 at 12 45 21 AM" src="https://github.com/user-attachments/assets/ea7b6169-c4d8-4103-856c-ec2bcb89b5f7" />

Temporal modeling is introduced using a ConvLSTM.

Pipeline:
1. Each frame is encoded using a shared CNN encoder
2. Encoded features are passed sequentially to a ConvLSTM
3. The final hidden state predicts the next frame
4. Anomaly score = prediction MSE

### Conv LSTM Cell
```python
class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch, hidden_ch):
        super().__init__()

        # number of channels in the hidden state
        self.hidden_ch = hidden_ch

        # One convolution layer is used to compute all 4 LSTM gates
        # We concatenate the current input (x) and previous hidden state (h)
        # along the channel dimension and pass them through this conv layer
        #
        # Output channels = 4 * hidden_ch because we need:
        # input gate (i), forget gate (f), output gate (o),
        # and candidate cell state (g)
        self.conv = nn.Conv2d(
            in_ch + hidden_ch,
            4 * hidden_ch,
            kernel_size=3,
            padding=1
        )

    def forward(self, x, h, c):
        # x : current input feature map (B, in_ch, H, W)
        # h : previous hidden state (B, hidden_ch, H, W)
        # c : previous cell state (B, hidden_ch, H, W)

        # concatenate input and hidden state
        combined = torch.cat([x, h], dim=1)

        # compute all gates together
        gates = self.conv(combined)

        # split the gates into 4 parts
        i, f, o, g = torch.chunk(gates, 4, dim=1)

        # apply activations
        i = torch.sigmoid(i)   # input gate
        f = torch.sigmoid(f)   # forget gate
        o = torch.sigmoid(o)   # output gate
        g = torch.tanh(g)      # candidate cell state

        # update cell state
        c = f * c + i * g

        # update hidden state
        h = o * torch.tanh(c)

        return h, c
```

**Results:**  
 Hidden size = 96 → mAP = 0.60

### 3. CNN + ConvLSTM + Frame Difference
**Notebook:** `cnn-lstm-framediff-visualisationtools.ipynb`  
**Model:** `cnn_convlstm_framediff_96_epoch3.pth`

Each timestep is represented by a two-channel input:  
raw frame + frame difference.

<img width="918" height="634" alt="Screenshot 2026-01-08 at 12 55 28 AM" src="https://github.com/user-attachments/assets/f9858246-6e86-498f-af72-9f47cabb5819" />

#### Regularization
- Dropout added in both encoder and decoder
- Prevents the model from becoming too good at reconstruction
- Controls overfitting on limited data

 **Results:**
- Hidden size = 64 → mAP = 0.54  
- Hidden size = 96 → mAP = 0.62

### 4 Strategies for Mean Average Precision optimization
### Score fusion
Fusion of:
- Baseline CAE scores
- CNN + ConvLSTM (+ FrameDiff) scores

```python

# Combine anomaly scores from the ConvLSTM model and the baseline autoencoder.
# The parameter `alpha` controls the contribution of each model.
# Higher values of alpha give more importance to motion-based anomalies
# captured by ConvLSTM, while lower values retain appearance-based reconstruction errors from the autoencoder.
alpha = 0.95
ae["Predicted"] = alpha * cl["Predicted"] + (1 - alpha) * ae["Predicted"]

```
### Exponential moving average (EMA)
Exponential Moving Average applied to frame-level anomaly scores helps in reducing noise while preserving anomaly peaks.

<img width="1046" height="345" alt="Screenshot 2026-01-08 at 1 03 52 AM" src="https://github.com/user-attachments/assets/5a2d4e42-3ec4-4904-8c92-c14b4f5eb6d3" />
notice how EMA helps to smooth out the peeks

```python
video_scores = pd.Series(video_scores).ewm(alpha=0.6).mean().values #Smooth anomaly scores using EMA to reduce noise and stabilize temporal trends
```

### Cold start prevention
First 2T (i.e 10 frames) is given score 0 to prevent the problem of cold start **Read the report for a detailed explaination**

### Global norm
```python
# Global normalization of anomaly scores:
# Rescale all scores to the range [0, 1] using the minimum and maximum
# observed across the entire test set. This makes scores comparable
# across different videos and stabilizes ranking for mAP evaluation.
mn, mx = min(all_scores), max(all_scores)
for k in score_map:
    score_map[k] = (score_map[k] - mn) / (mx - mn + 1e-8)
```

## Final Results Summary

| Model | Details | mAP |
|------|--------|-----|
| Baseline CAE | Frame reconstruction | 0.45 |
| CNN + ConvLSTM | Hidden = 96 | 0.6 |
| CNN + ConvLSTM + Frame Difference| Hidden = 96 | 0.62|
| **CNN + ConvLSTM + Frame Difference+Fusion + EMA + Warm-up** | **Final model** | **0.6618** |


## Exploratory Experiments

- RGB input degraded performance due to color noise
- Sequence length 3 and 7 underperformed vs `T = 5`
- Motion `.mat` volumes introduced strong spatial bias
- 3D CNNs and attention models avoided due to overfitting risk

  <img width="1046" height="399" alt="Screenshot 2026-01-08 at 1 15 41 AM" src="https://github.com/user-attachments/assets/0284e852-8aa4-4c86-b5ad-0924d033bc76" />





