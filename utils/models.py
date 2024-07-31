import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.conv import GATConv, MessagePassing
import pytorch_lightning as pl
from utils.layers import BasicLayer, CongFuLayer, GINEConv, create_mlp


NUM_ATOM_TYPE = 119
NUM_CHIRALITY_TAG = 3
NUM_BOND_TYPE = 5
NUM_BOND_DIRECTION = 3

class mySequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs

class CongFuBasedModel(nn.Module):

    '''
    Class for CongFu model
    Modified from : https://github.com/bowang-lab/CongFu/tree/main
    '''

    def __init__(self,
                 num_layers: int = 5,
                 inject_layer: int = 3,
                 emb_dim: int = 300,
                 mlp_hidden_dims: list = [256, 128, 64],
                 feature_dim: int = 512,
                 context_dim: int = 908,
                 device = torch.device('cuda'),
                 dropout_prob: float = 0,
                 gine_dropout_prob: float = 0
                 ):
        super().__init__()
        self.emb_dim = emb_dim
        self.device = device
        self.context_dim = context_dim
        self.dropout_prob = dropout_prob
        self.gine_dropout_prob = gine_dropout_prob

        self.x_embedding1 = nn.Embedding(NUM_ATOM_TYPE, emb_dim)
        self.x_embedding2 = nn.Embedding(NUM_CHIRALITY_TAG, emb_dim)
        nn.init.xavier_uniform_(self.x_embedding1.weight)
        nn.init.xavier_uniform_(self.x_embedding2.weight)

        basic_layers_number, congfu_layers_number = inject_layer, num_layers - inject_layer
        self.basic_layers = self._generate_basic_layers_new(basic_layers_number)
        self.congfu_layers = self._generate_congfu_layers_new(congfu_layers_number)

        self.context_encoder = create_mlp(context_dim, [feature_dim], emb_dim, activation = "relu", dropout = dropout_prob)
        self.output_transformation = create_mlp(emb_dim, [feature_dim], feature_dim//2, activation = "relu", dropout = dropout_prob)

        self.reshape_drug1drug2 = nn.Linear(feature_dim//2, feature_dim)

        self.mlp = create_mlp(self.emb_dim + feature_dim, mlp_hidden_dims, 1, activation="leaky_relu", dropout = dropout_prob)

    
    def _generate_basic_layers(self, number_of_layers: int) -> list[MessagePassing]:
        basic_layers = []

        for i in range(number_of_layers):
            graph_update_gnn = GINEConv(self.emb_dim, self.emb_dim, NUM_BOND_TYPE, NUM_BOND_DIRECTION, dropout_prob=self.gine_dropout_prob).to(self.device)
            last_layer = i == number_of_layers - 1
            basic_layer = BasicLayer(self.emb_dim, graph_update_gnn, last_layer).to(self.device)
            basic_layers.append(basic_layer)
        
        return basic_layers
    
    def _generate_basic_layers_new(self, number_of_layers: int) -> list[MessagePassing]:
        basic_layers = []

        for i in range(number_of_layers):
            graph_update_gnn = GINEConv(self.emb_dim, self.emb_dim, NUM_BOND_TYPE, NUM_BOND_DIRECTION, dropout_prob=self.gine_dropout_prob).to(self.device)
            last_layer = i == number_of_layers - 1
            basic_layer = BasicLayer(self.emb_dim, graph_update_gnn, last_layer).to(self.device)
            basic_layers.append(basic_layer)
    
        basic_layers = mySequential(
            *[layer for layer in basic_layers]
        )
        return basic_layers

    def _generate_congfu_layers(self, number_of_layers: int) -> list[MessagePassing]:
        congfu_layers = []

        for i in range(number_of_layers):
            graph_update_gnn = GINEConv(self.emb_dim, self.emb_dim, NUM_BOND_TYPE, NUM_BOND_DIRECTION, dropout_prob=self.gine_dropout_prob).to(self.device)
            bottleneck_gnn = GATConv(in_channels=(-1, -1), out_channels=self.emb_dim, add_self_loops=False)
            last_layer = i == number_of_layers - 1

            congfu_layer = CongFuLayer(self.emb_dim, self.emb_dim, self.emb_dim, graph_update_gnn, bottleneck_gnn, last_layer).to(self.device)
            congfu_layers.append(congfu_layer)
        
        return congfu_layers
    
    def _generate_congfu_layers_new(self, number_of_layers: int) -> list[MessagePassing]:
        congfu_layers = []

        for i in range(number_of_layers):
            graph_update_gnn = GINEConv(self.emb_dim, self.emb_dim, NUM_BOND_TYPE, NUM_BOND_DIRECTION, dropout_prob=self.gine_dropout_prob).to(self.device)
            bottleneck_gnn = GATConv(in_channels=(-1, -1), out_channels=self.emb_dim, add_self_loops=False)
            last_layer = i == number_of_layers - 1

            congfu_layer = CongFuLayer(self.emb_dim, self.emb_dim, self.emb_dim, graph_update_gnn, bottleneck_gnn, last_layer).to(self.device)
            congfu_layers.append(congfu_layer)
        
        congfu_layers = mySequential(
            *[layer for layer in congfu_layers]
        )
        return congfu_layers
    
    def _create_context_graph_edges(self, graph: Data) -> torch.Tensor:
        return torch.cat([
            torch.arange(graph.batch.size(0)).unsqueeze(0).to(self.device),
            graph.batch.unsqueeze(0),
        ], dim=0)
    
    def _embed_x(self, graph: Data) -> Data:
        embedding_1 = self.x_embedding1(graph.x[:, 0])
        embedding_2 = self.x_embedding2(graph.x[:, 1])
        graph.x = embedding_1 + embedding_2

        return graph

    def forward(self, graphA: Data, graphB: Data, context: torch.Tensor) -> torch.Tensor:
        graphA.context_x_edges = self._create_context_graph_edges(graphA)
        graphB.context_x_edges = self._create_context_graph_edges(graphB)

        graphA = self._embed_x(graphA)
        graphB = self._embed_x(graphB)
        context = self.context_encoder(context)

        graphA, graphB = self.basic_layers(graphA, graphB)
        graphA, graphB, context = self.congfu_layers(graphA, graphB, context)
        
        graphA.x = global_mean_pool(graphA.x, graphA.batch)
        graphA.x = self.output_transformation(graphA.x)

        graphB.x = global_mean_pool(graphB.x, graphB.batch)
        graphB.x = self.output_transformation(graphB.x)

        reshaped_drug1drug2 = self.reshape_drug1drug2(graphA.x+graphB.x)
        input_ = torch.concat((reshaped_drug1drug2, context), dim=1)

        return self.mlp(input_)
    
class MolformerBasedModel(nn.Module):
    '''
    Class for pre-trained molecular language model based synergy prediction model
    '''
    def __init__(
        self,
        drug_dim,
        context_dim,
        drug_hidden_dims,
        context_hidden_dims,
        drug_context_dim,
        dropout_prob
        ):
        super().__init__()

        self.lin1 = nn.Linear(drug_dim, drug_hidden_dims[0])
        self.lin2 = nn.Linear(drug_hidden_dims[0], drug_hidden_dims[1])
        self.lin3 = nn.Linear(context_dim, context_hidden_dims[0])
        self.lin4 = nn.Linear(context_hidden_dims[0], context_hidden_dims[1])
        self.lin5 = nn.Linear(drug_hidden_dims[1]+context_hidden_dims[1], drug_context_dim)
        self.lin6 = nn.Linear(drug_context_dim, 1)

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, drugA: Data, drugB: Data, cell_line: torch.Tensor) -> torch.Tensor:
        drugA = self.dropout(self.activation(self.lin1(drugA)))
        drugB = self.dropout(self.activation(self.lin1(drugB)))

        drugA = self.dropout(self.activation(self.lin2(drugA)))
        drugB = self.dropout(self.activation(self.lin2(drugB)))

        cell_line = self.dropout(self.activation(self.lin3(cell_line)))
        cell_line = self.dropout(self.activation(self.lin4(cell_line)))

        drugs_cellline = self.dropout(self.activation(self.lin5(torch.concat((drugA+drugB, cell_line), dim=1))))
        output = self.lin6(drugs_cellline)

        return output


class SynergyModel(pl.LightningModule):
    '''
    Pytorch lightning module for synergy prediction model training 
    '''
    def __init__(
        self,
        model_type,
        model_hyperparameters,
        n_epochs,
        lr
        ):
        super().__init__()

        if model_type == "congfu":
            self.model =CongFuBasedModel(**model_hyperparameters)
        elif model_type == "molformer":
            self.model = MolformerBasedModel(**model_hyperparameters)

        self.n_epochs = n_epochs
        self.lr = lr
        self.dropout_prob = model_hyperparameters["dropout_prob"]

        self.loss_fn = nn.L1Loss() 

        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

        self.save_hyperparameters()
      
    def forward(self,drugA, drugB, cell_line):

        output = self.model(drugA, drugB, cell_line)
        return output
    
    
    def training_step(self,batch,batch_idx):
        
        batch = [tensor.to(self.device) for tensor in batch]
        drugA, drugB, cell_line, target = batch

        output = self(drugA, drugB, cell_line)
        loss = self.loss_fn(output, target)
        
        self.log('train_loss',loss , prog_bar=True,logger=True)

        self.training_step_outputs.append(torch.hstack((output, target.view(-1,1))))
        
        return {"loss" :loss, "predictions":output, "labels": target }
    
    def on_train_epoch_end(self):
        with torch.no_grad():
            model_outputs = torch.vstack(self.training_step_outputs)[:,:1]
            labels = torch.vstack(self.training_step_outputs)[:,-1]
            loss = self.loss_fn(model_outputs.view(-1),labels)
            self.log('epoch_train_loss',loss , prog_bar=True,logger=True)
            self.training_step_outputs.clear()  # free memory

    def validation_step(self,batch,batch_idx):
        with torch.no_grad():
            batch = [tensor.to(self.device) for tensor in batch]
            drugA, drugB, cell_line, target = batch
            output = self(drugA, drugB, cell_line)
            self.validation_step_outputs.append(torch.hstack((output, target.view(-1,1))))

        return output
    
    def on_validation_epoch_end(self):
        with torch.no_grad():
            model_outputs = torch.vstack(self.validation_step_outputs)[:,:1]
            labels = torch.vstack(self.validation_step_outputs)[:,-1]
            loss = self.loss_fn(model_outputs.view(-1),labels)
            self.log('val_loss',loss , prog_bar=True,logger=True)
            self.validation_step_outputs.clear()  # free memory

    def test_step(self,batch,batch_idx):
        with torch.no_grad():
            batch = [tensor.to(self.device) for tensor in batch]
            drugA, drugB, cell_line, target = batch
            output = self(drugA, drugB, cell_line)
            self.test_step_outputs.append(torch.hstack((output, target.view(-1,1))))

        return output

    def on_test_epoch_end(self):
        with torch.no_grad():
            model_outputs = torch.vstack(self.test_step_outputs)[:,:1]
            labels = torch.vstack(self.test_step_outputs)[:,-1]
            loss = self.loss_fn(model_outputs.view(-1),labels)
            self.log_dict({'test_preds':model_outputs.view(-1), 'test_loss': loss})
            self.test_step_outputs.clear()  # free memory
            
            return model_outputs.view(-1)
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return {"optimizer": optimizer}
