# Pistachio-Image-Classification-ResNet-50-Transfer-Learning-
This project builds a complete image classification pipeline to distinguish between Kirmizi Pistachio and Siirt Pistachio varieties using deep learning. The dataset was sourced from Kaggle and processed, balanced, normalized, and trained using a fine‑tuned ResNet‑50 model.

Dataset
Two classes: Kirmizi_Pistachio, Siirt_Pistachio
Images extracted from ZIP and validated with PIL.Image.verify()
Balanced to 900 images per class (1600 total)

Preprocessing: Resize to 224×224

Compute dataset mean/std
Normalize using transforms.Normalize(mean, std)
Split into train, validation, and test sets

Model
Pretrained ResNet‑50 (IMAGENET1K_V2)
Backbone frozen
Custom classifier head:
Linear(2048 → 256)
ReLU
Dropout(0.5)
Linear(256 → 2)

Training
Optimizer: Adam
Loss: CrossEntropyLoss
Scheduler: StepLR
Early stopping (patience = 5)
Checkpointing saves best model automatically

Summary
A complete pipeline for binary image classification using transfer learning, covering dataset preparation, normalization, model customization, training, checkpointing, and evaluation.
