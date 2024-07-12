# Configuring Models for Pretraining, Fine-Tuning and Evaluation

Models are configured with the aid of the sacred module. All changes to model configurations are made in the renaissance/config.py file.

## Base Architecture

Two base architectures are available, one-tower and two-tower encoders. These model-tpyes
can be adjusted via:

```model_type = "two-tower" # Supports ['one-tower', 'two-tower]```

### Configuring One-Tower Encoders

#### Choose Encoder Module
One-tower models require the user to specify a single transformer encoder from the huggingface hub. Most text transformers and some vision transformers (ViT, Beit, Deit, etc.) can serve as the encoder module.

```encoder = "google/electra-small-discriminator"```

#### Choose Pooler Type
The user can choose how to pool the output of the encoder module. The 'doulbe' pooler option will concatenate the 'CLS' feature from the visual and textual inputs. The 'single' pooler option will simply take the first 'CLS' feature from both inputs.

```pooler_type = 'double' # Supports ['single', 'double']```

#### Randomly Initilize Encdoer Weights
By default, Renaissance uses pretrained weights from the the encoder downloaded from the huggingface hub. However when the random_init_encoder is set to False, the encoder's values will be randomly intitialized. This is often useful when establishing baselines. Code sample below:

```random_init_encoder = True```

#### Manually Configure Encoder Dimensions
By default, the encoder module will use the dimensions of the encoder downloaded from huggingface. However, if both the variables random_init_encoder and encoder_manual_configuration are set to true, the encoder will have randomly initialized weights with the provided model dimensions. Example below:

```
random_init_encoder = True
## Manual Configuration
encoder_manual_configuration = True
hidden_size = 192
num_heads = 4
num_layers = 12
mlp_ratio = 4
drop_rate = 0.1
embedding_size = 96
```
### Configuring Two-Tower Encoders

#### Choose Text-Encoder Module
Two-tower models require the user to specify a text transformer encoder from the huggingface hub. 

```text_encoder = "google/electra-small-discriminator"```

#### Choose Vision-Encoder Module
Two-tower models require the user to also specify a vision transformer encoder from the huggingface hub. A variety of vision transformer models will work, however using convolutional models that contain higher-dimensional layers, ResNet for example, will raise errors. 

```image_encoder = "facebook/deit-tiny-patch16-224"```

#### Configure Cross-modal-Encoder Dimensions
Finally, two-tower models use a fusion encoder with cross-attention to combine the vision and language streams. This module is always randomly intizialized in pretraining and must always be manually configured. The settings for choosing its properties are shown below:

```
# Cross Layer Settings
cross_layer_hidden_size = 256
num_cross_layers = 6
num_cross_layer_heads = 4
cross_layer_mlp_ratio = 4
cross_layer_drop_rate = 0.1
```
#### Randomly Initilize Text and/or Vision Encoder Weights
If the user would like to randomly initialize the weights of either encoder the text or vision encoder module, they can simply set the relevant variables to False.

```
## Train encoder model from scratch
random_init_vision_encoder = False
# Train Text Encoder from Sratch if True
random_init_text_encoder = False
```

#### Manually Configure Text and/or Image Encoder Dimensions
The user can manually configure the dimensions of the vision encoder if both the  random_init_vision_encoder and image_encoder_manual_configuration  variables are set to True.

```
random_init_vision_encoder = True
image_encoder_manual_configuration = True
## Manual Configuration
image_encoder_hidden_size = 192
image_encoder_num_heads = 4
image_encoder_num_layers = 12
image_encoder_mlp_ratio = 4
image_encoder_drop_rate = 0.1
image_encoder_embedding_size = 128
image_size = 224
original_image_size = 224 # Image size model is pretrained with, used in fine-tuning and testing
patch_size = 16
image_only = False
```

The same is true of the text encoder using the relevant variables:

```
random_init_text_encoder = True
text_encoder_manual_configuration = True
text_encoder_hidden_size = 192
text_encoder_num_heads = 4
text_encoder_num_layers = 12
text_encoder_mlp_ratio = 4
text_encoder_drop_rate = 0.1
text_encoder_embedding_size = 64
max_text_len = 40
vocab_size = 30522
```

## Pretraining Configuration

