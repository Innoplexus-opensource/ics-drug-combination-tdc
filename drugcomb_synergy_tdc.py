import numpy as np
import sys
import random
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, MLFlowLogger
from pathlib import Path
import warnings

import tdc
from utils.models import SynergyModel
from utils.dataset import get_datasets, DataModule
from utils.features import get_mol_dict, get_mol_embed_dict
from config import random_seed, gpu_id, batch_size, lr, log_dir, n_epochs, tensorboard_dirprefix, congfu_hyperparameters, molformer_hyperparameters


warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
torch.set_float32_matmul_precision("high")


group = tdc.BenchmarkGroup('drugcombo_group', path='data_new/',
                               file_format='pkl')
results_all_seeds = {}

# Accept which model architecture to use as argument from user
model_type = sys.argv[1]

for seed in [1, 2, 3, 4, 5]:
    predictions = {}
    for benchmark in group:
        # Loading train/val/test data split for respective benchmark
        train, validation = group.get_train_valid_split(seed = seed, benchmark = benchmark['name'])
        dataset, train_dataset, val_dataset, test_dataset, cell_lines = get_datasets(train, validation, benchmark['test'])

        np.random.seed(random_seed)
        random.seed(random_seed)
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_seed)
            torch.cuda.manual_seed_all(random_seed) # if you are using multi-GPU.
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Assignment of one of the two featurizers for drug molecules
        if model_type == "congfu":
            drug_featurizer = get_mol_dict
        elif model_type == "molformer":
            drug_featurizer = get_mol_embed_dict

        # Instantiate pytorch-lightning datamodule
        datamodule = DataModule(
                        train, validation, benchmark['test'], drug_featurizer=drug_featurizer,
                        batch_size=batch_size, device=device)
        
        datamodule.setup()

        # Instantiate the model
        if model_type == "congfu":
            model_hyperparameters = congfu_hyperparameters
        elif model_type == "molformer":
            model_hyperparameters = molformer_hyperparameters

        model = SynergyModel(model_type, model_hyperparameters, n_epochs, lr)
            
        # Initialize Pytorch Lightning callback for Model checkpointing
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss", # monitored quantity
            filename='{epoch:02d}-{val_loss:.2f}',
            save_top_k=1, #  save the top k models
            mode="min", # mode of the monitored quantity  for optimization
        )
        
        # Initialize loggers
        tensorboard_dirname = f"{tensorboard_dirprefix}_{model_type}"
        tensorboard_exptname = f"{benchmark['name']}_seed_{seed}" 

        tb_logger = TensorBoardLogger(Path(log_dir).joinpath(tensorboard_dirname), name=tensorboard_exptname)
        mlf_logger = MLFlowLogger(experiment_name=tensorboard_dirname, run_name=tensorboard_exptname, save_dir=Path(log_dir).joinpath("mlflow_logs"))

        mlf_logger.log_hyperparams(model_hyperparameters)
        
        # Instantiate the Model Trainer
        trainer = pl.Trainer(max_epochs = n_epochs , accelerator = "cuda", callbacks=[checkpoint_callback], logger=[tb_logger, mlf_logger], log_every_n_steps=1, devices=[gpu_id])

        # Train the Classifier Model
        trainer.fit(model, datamodule)

        # Retreive the checkpoint path for best model
        model_path = checkpoint_callback.best_model_path
        print(f"Best model path: {model_path}")

        # Evaluate the model performance on the test dataset
        model = SynergyModel.load_from_checkpoint(model_path, model_type=model_type, model_hyperparameters=model_hyperparameters)

        model = model.to(device) # moving model to cuda
        model.eval()
        test_outputs = []
        with torch.no_grad():
            for batch in datamodule.test_dataloader():
                batch = [tensor.to(device) for tensor in batch]
                drugA, drugB, cell_line, target = batch
                output = model(drugA, drugB, cell_line)
                test_outputs.append(output)

        test_outputs = torch.vstack(test_outputs).view(-1).cpu()
        predictions[benchmark['name']] = test_outputs
    
    # Evaluate model performance using tdc module
    out = group.evaluate(predictions)
    print(out)
    results_all_seeds['seed ' + str(seed)] = out

def to_submission_format(results):
    '''
    Get average and std deviation across different splits
    '''
    import pandas as pd
    df = pd.DataFrame(results)
    def get_metric(x):
        metric = []
        for i in x:
            metric.append(list(i.values())[0])
        return [round(np.mean(metric), 3), round(np.std(metric), 3)]
    return dict(df.apply(get_metric, axis = 1))

submission_results = to_submission_format(results_all_seeds)

# Final results in submission format
print(submission_results) 

