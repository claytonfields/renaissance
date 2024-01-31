from sacred import Experiment

ex = Experiment("METER")


def _loss_names(d):
    ret = {
        "itm": 0,
        "mlm": 0,
        "mpp": 0,
        "vqa": 0,
        "vcr": 0,
        "vcr_qar": 0,
        "nlvr2": 0,
        "irtr": 0,
        "contras": 0,
        "snli": 0,
        "ref": 0,
        "mrpc" : 0,
        "rte" : 0,
        'wnli' : 0,
        'sst2' : 0,
        'qqp' : 0,
        'qnli' : 0,
        'mnli' : 0,
        'cola' : 0
    }
    ret.update(d)
    return ret

# ===================== Default Settings ===================== #
@ex.config
def config():
    exp_name = "meter"
    seed = 0
    # datasets = ["coco", "vg", "sbu", "gcc"]
    datasets = ["coco", "vg"]
    loss_names = _loss_names({"itm": 1, "mlm": 1})
    batch_size = 256  # this is a desired batch size; pl trainer will accumulate gradients when per step batch is smaller.

    # Image settings
    image_encoder = 'vit_deit_tiny_patch16_224'
    random_init_vision_encoder = False
    image_encoder_hidden_size = 192
    train_transform_keys = ["imagenet"]
    val_transform_keys = ["imagenet"]
    image_size = 224
    resolution_before = 224
    patch_size = 16
    draw_false_image = 1
    image_only = False

    # Text Setting
    text_encoder = "google/electra-small-discriminator"
    random_init_text_encoder = False
    text_encoder_hidden_size = 256
    max_text_len = 40
    vocab_size = 30522
    whole_word_masking = False # note that whole_word_masking does not work for RoBERTa
    mlm_prob = 0.15
    draw_false_text = 0
    vqav2_label_size = 3129
    
    # Architecture Setting
    two_tower = True
    multi_model_encoder = 'dandelin/vilt-b32-mlm'
    
    # Cross Layer Settings
    cross_layer_hidden_size = 256
    num_cross_layers = 6
    num_cross_layer_heads = 4
    # num_layers = 6
    cross_layer_mlp_ratio = 4
    cross_layer_drop_rate = 0.1

    # Optimizer Setting
    optim_type = "adamw"
    learning_rate = 1e-5
    weight_decay = 0.01
    decay_power = 1
    max_epoch = 100
    max_steps = 100000
    warmup_steps = 10000
    end_lr = 0
    lr_mult_head = 5  # multiply lr for downstream heads
    lr_mult_cross_modal = 5  # multiply lr for the cross-modal module

    # Downstream Setting
    get_recall_metric = False
    
    hugging_face = False
    # model_type = "METER"
    
    # Trainag Parameter Setting
    freeze_image_encoder = False
    freeze_text_encoder = False
    freeze_cross_modal_layers = False
    
    # PL Trainer Setting
    resume_from = None
    fast_dev_run = False
    val_check_interval = 1.0
    test_only = False

    # below params varies with the environment
    data_root = ""
    log_dir = "result"
    per_gpu_batchsize = 0  # you should define this manually with per_gpu_batch_size=#
    num_gpus = 1
    num_nodes = 1
    load_path = ""
    num_workers = 12
    precision = 32


# ===================== Task Settings ===================== #
@ex.named_config
def task_mlm_itm():
    exp_name = "mlm_itm"
    # datasets = ["coco", "vg", "sbu", "gcc"]
    datasets = ["coco", "vg"]
    loss_names = _loss_names({"itm": 1, "mlm": 1})
    batch_size = 256
    max_epoch = None
    max_steps = 100000
    warmup_steps = 0.1
    whole_word_masking = True

    # vocab_size = 30522
    max_text_len = 50
    image_size = 224
    learning_rate = 1e-5
    val_check_interval = 1.0
    lr_mult_head = 5
    lr_mult_cross_modal = 5
    num_cross_layers = 6
    # cross_layer_hidden_size = 256
    # num_cross_layer_heads = 12
    
@ex.named_config
def task_mlm_itm_deit_electra():
    exp_name = "mlm_itm_deit_electra"
    # datasets = ["coco", "vg", "sbu", "gcc"]
    datasets = ["coco", "vg"]
    loss_names = _loss_names({"itm": 1, "mlm": 1})
    batch_size = 336
    max_epoch = None
    max_steps = 100000
    warmup_steps = 0.1
    whole_word_masking = True
    # DO NOT Freeze Encoders
    freeze_image_encoder = False
    freeze_text_encoder = False
    # Image settings
    image_encoder = 'vit_deit_tiny_patch16_224'
    image_encoder_hidden_size = 192
    train_transform_keys = ["imagenet"]
    val_transform_keys = ["imagenet"]
    image_size = 224
    resolution_before = 224
    patch_size = 16
    draw_false_image = 1
    image_only = False
    # Text Setting
    text_encoder = "google/electra-small-discriminator"
    text_encoder_hidden_size = 256
    max_text_len = 50
    vocab_size = 30522
    whole_word_masking = False # note that whole_word_masking does not work for RoBERTa
    mlm_prob = 0.15
    draw_false_text = 0
    vqav2_label_size = 3129
    # Cross Layer Settings
    cross_layer_hidden_size = 256
    num_cross_layers = 6
    num_cross_layer_heads = 4
    cross_layer_mlp_ratio = 4
    cross_layer_drop_rate = 0.1
    # Optimizer Settings
    learning_rate = 1e-5
    val_check_interval = 1.0
    lr_mult_head = 5
    lr_mult_cross_modal = 5

@ex.named_config
def task_mlm_itm_deit_fr_electra():
    exp_name = "mlm_itm_deit_fr_electra"
    # datasets = ["coco", "vg", "sbu", "gcc"]
    datasets = ["coco", "vg"]
    loss_names = _loss_names({"itm": 1, "mlm": 1})
    batch_size = 256
    max_epoch = None
    max_steps = 100000
    warmup_steps = 0.1
    whole_word_masking = True
    
    # Freeze Image Encoder
    freeze_image_encoder = True
    # DO NOT Freeze Text Encoder
    freeze_text_encoder = False
    
    vocab_size = 30522
    max_text_len = 50
    image_size = 224
    train_transform_keys = ["imagenet_randaug"]
    val_transform_keys = ["imagenet_randaug"]
    learning_rate = 1e-5
    val_check_interval = 1.0
    lr_mult_head = 5
    lr_mult_cross_modal = 5
    num_cross_layers = 6
    
@ex.named_config
def task_mlm_itm_deit_electra_fr():
    exp_name = "mlm_itm_deit_electra_fr"
    # datasets = ["coco", "vg", "sbu", "gcc"]
    datasets = ["coco", "vg"]
    loss_names = _loss_names({"itm": 1, "mlm": 1})
    batch_size = 256
    max_epoch = None
    max_steps = 100000
    warmup_steps = 0.1
    whole_word_masking = True
    
    # DO NOT Freeze Image Encoder
    freeze_image_encoder = False
    # Freeze Text Encoder
    freeze_text_encoder = True
    
    vocab_size = 30522
    max_text_len = 50
    image_size = 224
    train_transform_keys = ["imagenet_randaug"]
    val_transform_keys = ["imagenet_randaug"]
    learning_rate = 1e-5
    val_check_interval = 1.0
    lr_mult_head = 5
    lr_mult_cross_modal = 5
    num_cross_layers = 6

@ex.named_config
def task_mlm_itm_deit_fr_electra_fr():
    exp_name = "mlm_itm_deit_fr_electra_fr"
    # datasets = ["coco", "vg", "sbu", "gcc"]
    datasets = ["coco", "vg"]
    loss_names = _loss_names({"itm": 1, "mlm": 1})
    batch_size = 256
    max_epoch = None
    max_steps = 100000
    warmup_steps = 0.1
    whole_word_masking = True
    
    # Freeze Encoders
    freeze_image_encoder = True
    freeze_text_encoder = True
    
    vocab_size = 30522
    max_text_len = 50
    image_size = 224
    train_transform_keys = ["imagenet_randaug"]
    val_transform_keys = ["imagenet_randaug"]
    learning_rate = 1e-5
    val_check_interval = 1.0
    lr_mult_head = 5
    lr_mult_cross_modal = 5
    num_cross_layers = 6

@ex.named_config
def task_finetune_nlvr2_clip_bert():
    exp_name = "finetune_nlvr2"
    datasets = ["nlvr2"]
    loss_names = _loss_names({"nlvr2": 1})
    batch_size = 256
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 1e-5
    lr_mult_head = 10
    lr_mult_cross_modal = 5
    text_encoder = "bert-base-uncased"
    max_text_len = 50
    text_encoder_hidden_size = 768
    image_encoder = 'ViT-B/32'
    train_transform_keys = ["clip"]
    val_transform_keys = ["clip"]
    image_encoder_hidden_size = 768
    image_size = 288

@ex.named_config
def task_finetune_vqa_clip_bert():
    exp_name = "finetune_vqa"
    datasets = ["vqa"]
    loss_names = _loss_names({"vqa": 1})
    batch_size = 512
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 5e-6
    val_check_interval = 0.1
    lr_mult_head = 50
    lr_mult_cross_modal = 5
    text_encoder = "bert-base-uncased"
    max_text_len = 50
    text_encoder_hidden_size = 768
    image_encoder = 'ViT-B/32'
    train_transform_keys = ["clip"]
    val_transform_keys = ["clip"]
    image_encoder_hidden_size = 768
    image_size = 576

@ex.named_config
def task_finetune_irtr_coco_clip_bert():
    exp_name = "finetune_irtr_coco"
    datasets = ["coco"]
    loss_names = _loss_names({"itm": 0.5, "irtr": 1})
    batch_size = 512
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    get_recall_metric = True
    draw_false_text = 15
    learning_rate = 5e-6
    lr_mult_head = 5
    lr_mult_cross_modal = 5
    text_encoder = "bert-base-uncased"
    text_encoder_hidden_size = 768
    image_encoder = 'ViT-B/32'
    train_transform_keys = ["clip"]
    val_transform_keys = ["clip"]
    image_encoder_hidden_size = 768
    image_size = 384

@ex.named_config
def task_finetune_irtr_f30k_clip_bert():
    exp_name = "finetune_irtr_f30k"
    datasets = ["f30k"]
    loss_names = _loss_names({"itm": 0.5, "irtr": 1})
    batch_size = 512
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    get_recall_metric = True
    draw_false_text = 15
    learning_rate = 5e-6
    lr_mult_head = 5
    lr_mult_cross_modal = 5
    text_encoder = "bert-base-uncased"
    text_encoder_hidden_size = 768
    image_encoder = 'ViT-B/32'
    train_transform_keys = ["clip"]
    val_transform_keys = ["clip"]
    image_encoder_hidden_size = 768
    image_size = 384

@ex.named_config
def task_finetune_snli_clip_bert():
    exp_name = "finetune_snli"
    datasets = ["snli"]
    loss_names = _loss_names({"snli": 1})
    batch_size = 64
    max_epoch = 5
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 2e-6
    lr_mult_head = 10
    lr_mult_cross_modal = 5
    text_encoder = "bert-base-uncased"
    max_text_len = 50
    text_encoder_hidden_size = 768
    image_encoder = 'ViT-B/32'
    train_transform_keys = ["clip"]
    val_transform_keys = ["clip"]
    image_encoder_hidden_size = 768
    image_size = 384

@ex.named_config
def task_finetune_snli():
    exp_name = "finetune_snli"
    datasets = ["snli"]
    loss_names = _loss_names({"snli": 1})
    batch_size = 64
    max_epoch = 5
    max_steps = 10e6
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 2e-6
    lr_mult_head = 10
    lr_mult_cross_modal = 5
    max_text_len = 50
    image_size = 384
    # DO NOT Freeze Encoders
    freeze_image_encoder = False
    freeze_text_encoder = False
    
@ex.named_config
def task_finetune_snli_vision_fr_text_fr():
    exp_name = "finetune_snli_vision_fr_text_fr"
    datasets = ["snli"]
    loss_names = _loss_names({"snli": 1})
    batch_size = 64
    max_epoch = 5
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 2e-6
    lr_mult_head = 10
    lr_mult_cross_modal = 5
    max_text_len = 50
    image_size = 384
    # Freeze Encoders
    freeze_image_encoder = True
    freeze_text_encoder = True



@ex.named_config
def task_finetune_ref():
    exp_name = "finetune_ref"
    datasets = ["coco"]
    loss_names = _loss_names({"ref": 1})
    batch_size = 4
    max_epoch = 5
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 2e-6
    lr_mult_head = 10
    lr_mult_cross_modal = 5
    text_encoder = "google/electra-small-discriminator"
    max_text_len = 40
    text_encoder_hidden_size = 128
    image_encoder = 'vit_deit_tiny_patch16_224'
    train_transform_keys = ["imagenet"]
    val_transform_keys = ["imagenet"]
    image_encoder_hidden_size = 192
    image_size = 224







# ===================== Vision Encoders ===================== #
@ex.named_config
def swin32_base224():
    image_encoder = "swin_base_patch4_window7_224_in22k"
    patch_size = 32
    image_size = 224
    train_transform_keys = ["imagenet"]
    val_transform_keys = ["imagenet"]
    image_encoder_hidden_size = 1024
    resolution_before = 224

@ex.named_config
def swin32_base384():
    image_encoder = "swin_base_patch4_window12_384_in22k"
    patch_size = 32
    image_size = 384
    train_transform_keys = ["imagenet"]
    val_transform_keys = ["imagenet"]
    image_encoder_hidden_size = 1024
    resolution_before = 384

@ex.named_config
def swin32_large384():
    image_encoder = "swin_large_patch4_window12_384_in22k"
    patch_size = 32
    image_size = 384
    train_transform_keys = ["imagenet"]
    val_transform_keys = ["imagenet"]
    image_encoder_hidden_size = 1536
    resolution_before = 384
    
# @ex.named_config
# def swin_tiny_patch224():
#     image_encoder = "swin_tiny_patch4_window7_224"
#     patch_size = 4
#     image_size = 224
#     train_transform_keys = ["imagenet"]
#     val_transform_keys = ["imagenet"]
#     image_encoder_hidden_size = 768
#     resolution_before = 224
    
@ex.named_config
def swin_tiny_patch4_window7_224():
    image_encoder = "microsoft/swin-tiny-patch4-window7-224"
    patch_size = 4
    image_size = 224
    train_transform_keys = ["imagenet"]
    val_transform_keys = ["imagenet"]
    image_encoder_hidden_size = 768
    resolution_before = 224
    
@ex.named_config
def deit_small_distilled_patch16_224():    
    image_encoder = "vit_deit_small_distilled_patch16_224"
    cross_layer_hidden_size = 384
    num_cross_layer_heads = 6
    # num_layers = 12
    cross_layer_mlp_ratio = 4
    cross_layer_drop_rate = 0.1
    train_transform_keys = ["imagenet"]
    val_transform_keys = ["imagenet"]
    
@ex.named_config
def deit_tiny_patch16_224():    
    image_encoder = "facebook/deit-tiny-patch16-224"
    image_encoder_hidden_size = 192
    image_size = 224
    resolution_before = 224
    patch_size = 16
    train_transform_keys = ["imagenet"]
    val_transform_keys = ["imagenet"]
    
@ex.named_config
def vit_deit_tiny_patch16_224():    
    image_encoder = "vit_deit_tiny_patch16_224"
    cross_layer_hidden_size = 192
    image_encoder_hidden_size = 192
    resolution_before = 224
    patch_size = 16
    num_cross_layer_heads = 3
    # num_layers = 12
    cross_layer_mlp_ratio = 4
    cross_layer_drop_rate = 0.1
    train_transform_keys = ["imagenet"]
    val_transform_keys = ["imagenet"]

@ex.named_config
def clip16():
    image_encoder = 'ViT-B/16'
    image_size = 224
    patch_size = 16
    train_transform_keys = ["clip"]
    val_transform_keys = ["clip"]
    image_encoder_hidden_size = 768

@ex.named_config
def clip32():
    image_encoder = 'ViT-B/32'
    image_size = 224
    patch_size = 32
    train_transform_keys = ["clip"]
    val_transform_keys = ["clip"]
    image_encoder_hidden_size = 768

# ===================== Text Encoders ===================== #
@ex.named_config
def text_roberta():
    text_encoder = "roberta-base"
    vocab_size = 50265
    text_encoder_hidden_size = 768

@ex.named_config
def text_roberta_large():
    text_encoder = "roberta-large"
    vocab_size = 50265
    text_encoder_hidden_size = 1024

@ex.named_config
def text_electra_small():
    text_encoder = "google/electra-small-discriminator"
    vocab_size = 30522
    text_encoder_hidden_size = 256
    num_cross_layer_heads = 4
    # num_layers = 6
    cross_layer_mlp_ratio = 4
    cross_layer_hidden_size = 256
    
@ex.named_config
def text_electra_base():
    tokenizer = "google/electra-base-discriminator"
    vocab_size = 30522
    text_encoder_hidden_size = 768
    
# ===================== Random Augmentations ===================== #
@ex.named_config
def clip_randaug():
    train_transform_keys = ["clip"]
    val_transform_keys = ["clip"]

@ex.named_config
def imagenet_randaug():
    train_transform_keys = ["imagenet"]
    val_transform_keys = ["imagenet"]
    
# =========== Freeze of Un-Freeze Encoders for Training =========== #
@ex.named_config
def freeze_image():
    freeze_image_encoder = True

@ex.named_config
def freeze_text():
    freeze_text_encoder = True
    
# task_finetune_snli text_electra_small imagenet_randaug
    
# ===================== Test Cases ===================== #

   
    
    
   
@ex.named_config
def test_case_mlm_itm_a():
    # Settings
    test_only=False
    data_root = 'data/arrow/' 
    num_gpus=1 
    num_nodes=1 
    per_gpu_batchsize=64 
    resume_from = ''
    # SNLI-VE
    exp_name = "test_case_mlm_itm"
    datasets = ["coco", "vg"]
    loss_names = _loss_names({"itm": 1, "mlm": 1})
    # Training Time
    max_epoch = 1
    max_steps = 10
    # Text Encoder
    text_encoder = "google/electra-small-discriminator"
    vocab_size = 30522
    text_encoder_hidden_size = 256
    # Cross Layer Settings
    cross_layer_hidden_size = 256
    num_cross_layers = 6
    num_cross_layer_heads = 4
    # num_layers = 6
    cross_layer_mlp_ratio = 4
    cross_layer_drop_rate = 0.1
    # Image Encoder Settings
    image_encoder = "facebook/deit-tiny-patch16-224"
    image_encoder_hidden_size = 192
    image_size = 224
    resolution_before = 224
    patch_size = 16
    train_transform_keys = ["imagenet"]
    val_transform_keys = ["imagenet"]
    # Training Settings
    batch_size = 64
    warmup_steps = 0.1
    draw_false_image = 1
    draw_false_text = 0
    learning_rate = 2e-6
    lr_mult_head = 10
    lr_mult_cross_modal = 5
    max_text_len = 50
    whole_word_masking = True
    mlm_prob = 0.15
    # Freeze or UnFreeze Encoders
    freeze_image_encoder = False
    freeze_text_encoder = False


@ex.named_config
def test_case_mlm_itm_b():
    # Settings
    test_only=False
    data_root = 'data/arrow/' 
    num_gpus=1 
    num_nodes=1 
    per_gpu_batchsize=64 
    resume_from = ''
    # SNLI-VE
    exp_name = "test_case_mlm_itm"
    datasets = ["coco", "vg"]
    loss_names = _loss_names({"itm": 1, "mlm": 1})
    # Training Time
    max_epoch = 1
    max_steps = 10
    # Text Encoder
    text_encoder = "google/electra-base-discriminator"
    vocab_size = 30522
    text_encoder_hidden_size = 768
    # Cross Layer Settings
    cross_layer_hidden_size = 256
    num_cross_layers = 6
    num_cross_layer_heads = 4
    # num_layers = 6
    cross_layer_mlp_ratio = 4
    cross_layer_drop_rate = 0.1
    # Image Encoder Settings
    image_encoder = "microsoft/swin-tiny-patch4-window7-224"
    patch_size = 4
    image_size = 224
    train_transform_keys = ["imagenet"]
    val_transform_keys = ["imagenet"]
    image_encoder_hidden_size = 768
    resolution_before = 224
    # Training Settings
    batch_size = 64
    warmup_steps = 0.1
    draw_false_image = 1
    draw_false_text = 0
    learning_rate = 2e-6
    lr_mult_head = 10
    lr_mult_cross_modal = 5
    max_text_len = 50
    whole_word_masking = True
    mlm_prob = 0.15
    # Freeze or UnFreeze Encoders
    freeze_image_encoder = True
    freeze_text_encoder = True


# SNLI
@ex.named_config
def test_case_finetune_snli_a():
    # Settings
    test_only=False
    data_root = 'data/arrow/' 
    num_gpus=1 
    num_nodes=1 
    per_gpu_batchsize=32 
    load_path = '/home/claytonfields/nlp/code/meter/result/mlm_itm_deit_fr_electra_fr_is224_ps16_bs336_pgbs84_ts100k/checkpoints/epoch=5-step=96215.ckpt'
    # SNLI-VE
    exp_name = "test_case_finetune_snli"
    datasets = ["snli"]
    loss_names = _loss_names({"snli": 1})
    # Training Time
    max_epoch = 1
    max_steps = 10
    # Text Encoder
    text_encoder = "google/electra-small-discriminator"
    vocab_size = 30522
    text_encoder_hidden_size = 256
    # Cross Layer Settings
    cross_layer_hidden_size = 256
    num_cross_layers = 6
    num_cross_layer_heads = 4
    # num_layers = 6
    cross_layer_mlp_ratio = 4
    cross_layer_drop_rate = 0.1
    # Image Encoder Settings
    image_encoder = "facebook/deit-tiny-patch16-224"
    image_encoder_hidden_size = 192
    image_size = 224
    resolution_before = 224
    patch_size = 16
    train_transform_keys = ["imagenet"]
    val_transform_keys = ["imagenet"]
    # Training Settings
    batch_size = 64
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 2e-6
    lr_mult_head = 10
    lr_mult_cross_modal = 5
    max_text_len = 50
    # Freeze or UnFreeze Encoders
    freeze_image_encoder = False
    freeze_text_encoder = False
@ex.named_config

def test_case_eval_snli_a():
    # Settings
    test_only=True
    data_root = 'data/arrow/' 
    num_gpus=1 
    num_nodes=1 
    per_gpu_batchsize=64 
    # load_path = 'result/finetune_snli_mlm_itm_deit_fr_electra_fr_is224_ps16_bs336_pgbs84_ts100k/version_0/checkpoints/epoch\=4-step\=20684.ckpt deit_tiny_patch16_224'
    # SNLI-VE
    exp_name = "test_case_finetune_snli"
    datasets = ["snli"]
    loss_names = _loss_names({"snli": 1})
    # Training Time
    max_epoch = 1
    max_steps = 1000
    # Text Encoder
    text_encoder = "google/electra-small-discriminator"
    vocab_size = 30522
    text_encoder_hidden_size = 256
    # Cross Layer Settings
    cross_layer_hidden_size = 256
    num_cross_layers = 6
    num_cross_layer_heads = 4
    # num_layers = 6
    cross_layer_mlp_ratio = 4
    cross_layer_drop_rate = 0.1
    # Image Encoder Settings
    image_encoder = "facebook/deit-tiny-patch16-224"
    image_encoder_hidden_size = 192
    image_size = 224
    resolution_before = 224
    patch_size = 16
    train_transform_keys = ["imagenet"]
    val_transform_keys = ["imagenet"]
    # Training Settings
    batch_size = 64
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 2e-6
    lr_mult_head = 10
    lr_mult_cross_modal = 5
    max_text_len = 50
    # Freeze or UnFreeze Encoders
    freeze_image_encoder = False
    freeze_text_encoder = False
    
@ex.named_config
def test_case_finetune_snli_b():
    # Settings
    test_only=False
    data_root = 'data/arrow/' 
    num_gpus=1 
    num_nodes=1 
    per_gpu_batchsize=32 
    # load_path = '/home/claytonfields/nlp/code/meter/result/mlm_itm_deit_fr_electra_fr_is224_ps16_bs336_pgbs84_ts100k/checkpoints/epoch=5-step=96215.ckpt'
    # SNLI-VE
    exp_name = "test_case_finetune_snli"
    datasets = ["snli"]
    loss_names = _loss_names({"snli": 1})
    # Training Time
    max_epoch = 1
    max_steps = 10
    # Text Encoder
    text_encoder = "google/electra-base-discriminator"
    vocab_size = 30522
    text_encoder_hidden_size = 768
    # Cross Layer Settings
    cross_layer_hidden_size = 256
    num_cross_layers = 6
    num_cross_layer_heads = 4
    # num_layers = 6
    cross_layer_mlp_ratio = 4
    cross_layer_drop_rate = 0.1
    # Image Encoder Settings
    image_encoder = "microsoft/swin-tiny-patch4-window7-224"
    patch_size = 4
    image_size = 224
    train_transform_keys = ["imagenet"]
    val_transform_keys = ["imagenet"]
    image_encoder_hidden_size = 768
    resolution_before = 224
    # Training Settings
    batch_size = 64
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 2e-6
    lr_mult_head = 10
    lr_mult_cross_modal = 5
    max_text_len = 50
    # Freeze or UnFreeze Encoders
    freeze_image_encoder = True
    freeze_text_encoder = True

@ex.named_config
def test_case_eval_snli_b():
    # Settings
    test_only=True
    data_root = 'data/arrow/' 
    num_gpus=1 
    num_nodes=1 
    per_gpu_batchsize=64 
    # load_path = 'result/finetune_snli_mlm_itm_deit_fr_electra_fr_is224_ps16_bs336_pgbs84_ts100k/version_0/checkpoints/epoch\=4-step\=20684.ckpt deit_tiny_patch16_224'
    # SNLI-VE
    exp_name = "test_case_finetune_snli"
    datasets = ["snli"]
    loss_names = _loss_names({"snli": 1})
    # Training Time
    max_epoch = 1
    max_steps = 1000
    # Text Encoder
    text_encoder = "google/electra-base-discriminator"
    vocab_size = 30522
    text_encoder_hidden_size = 768
    # Cross Layer Settings
    cross_layer_hidden_size = 256
    num_cross_layers = 6
    num_cross_layer_heads = 4
    # num_layers = 6
    cross_layer_mlp_ratio = 4
    cross_layer_drop_rate = 0.1
    # Image Encoder Settings
    image_encoder = "microsoft/swin-tiny-patch4-window7-224"
    patch_size = 4
    image_size = 224
    train_transform_keys = ["imagenet"]
    val_transform_keys = ["imagenet"]
    image_encoder_hidden_size = 768
    resolution_before = 224
    # Training Settings
    batch_size = 64
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 2e-6
    lr_mult_head = 10
    lr_mult_cross_modal = 5
    max_text_len = 50
    # Freeze or UnFreeze Encoders
    freeze_image_encoder = True
    freeze_text_encoder = True
    
# SNLI
@ex.named_config
def test_case_finetune_mrpc_a():
    # Settings
    test_only=False
    data_root = 'data/arrow/' 
    num_gpus=1 
    num_nodes=1 
    per_gpu_batchsize=32 
    load_path = '/home/claytonfields/nlp/code/meter/result/mlm_itm_deit_fr_electra_fr_is224_ps16_bs336_pgbs84_ts100k/checkpoints/epoch=5-step=96215.ckpt'
    # SNLI-VE
    exp_name = "test_case_finetune"
    datasets = ["mrpc"]
    loss_names = _loss_names({"mrpc": 1})
    # Training Time
    max_epoch = 1
    max_steps = 10e6
    # Text Encoder
    text_encoder = "google/electra-small-discriminator"
    vocab_size = 30522
    text_encoder_hidden_size = 256
    # Cross Layer Settings
    cross_layer_hidden_size = 256
    num_cross_layers = 6
    num_cross_layer_heads = 4
    # num_layers = 6
    cross_layer_mlp_ratio = 4
    cross_layer_drop_rate = 0.1
    # Image Encoder Settings
    image_encoder = "facebook/deit-tiny-patch16-224"
    image_encoder_hidden_size = 192
    image_size = 224
    resolution_before = 224
    patch_size = 16
    train_transform_keys = ["imagenet"]
    val_transform_keys = ["imagenet"]
    # Training Settings
    batch_size = 32
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 2e-6
    lr_mult_head = 10
    lr_mult_cross_modal = 5
    max_text_len = 50
    # Freeze or UnFreeze Encoders
    freeze_image_encoder = False
    freeze_text_encoder = False
    
    
# Refcoco Reference Resolution
@ex.named_config
def test_case_finetune_refcoco_a():
    exp_name = "test_finetune_ref_case_a"
    datasets = ["refcoco"]
    # "itm": 0, "mlm": 0
    loss_names = _loss_names({"ref": 1})
    # Settings
    test_only=False
    data_root = 'data/arrow/' 
    num_gpus=1 
    num_nodes=1 
    batch_size = 5
    per_gpu_batchsize=5
    load_path = '/home/claytonfields/nlp/code/meter/result/mlm_itm_deit_fr_electra_fr_is224_ps16_bs336_pgbs84_ts100k/checkpoints/epoch=5-step=96215.ckpt'
    # Image Encoder Settings
    image_encoder = "facebook/deit-tiny-patch16-224"
    image_encoder_hidden_size = 192
    image_size = 224
    resolution_before = 224
    patch_size = 16
    train_transform_keys = ["imagenet"]
    val_transform_keys = ["imagenet"]
    draw_false_image = 0
    # Text Encoder Settings
    text_encoder = "google/electra-small-discriminator"
    vocab_size = 30522
    text_encoder_hidden_size = 256
    draw_false_text = 0
    # Cross Layer Settings
    cross_layer_hidden_size = 256
    num_cross_layers = 6
    num_cross_layer_heads = 4
    # num_layers = 6
    cross_layer_mlp_ratio = 4
    cross_layer_drop_rate = 0.1

    # Optimizer and Training Settings
    warmup_steps = 0
    learning_rate = 2e-6
    lr_mult_head = 10
    lr_mult_cross_modal = 5
    max_text_len = 50
    # Freeze or UnFreeze Encoders
    freeze_image_encoder = False
    freeze_text_encoder = False

    # Downstream Setting
    get_recall_metric = False


