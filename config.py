random_seed = 10
log_dir = "tdc_drugcomb_log"
tensorboard_dirprefix = "tdc_"

batch_size=128
n_epochs=20
lr=1e-4

gpu_id=0

congfu_hyperparameters = {
    "num_layers": 5,
    "inject_layer": 3,
    "emb_dim": 300,
    "feature_dim": 512,
    "context_dim": 27606,
    "dropout_prob": 0.6,
    "gine_dropout_prob": 0
}

molformer_hyperparameters = {
    "drug_dim": 768,
    "context_dim": 27606,
    "drug_hidden_dims": [600, 300],
    "context_hidden_dims": [600, 300],
    "drug_context_dim": 300,
    "dropout_prob": 0.5
}

drug_representation_model = "ibm/MoLFormer-XL-both-10pct"
