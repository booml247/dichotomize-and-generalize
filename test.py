@click.option('-d', '--dataset', type=str, default="breast", help="Name of the dataset to use.")
@click.option('--experiment-name', type=str, default="test", help="Name of the experiment (for logging).")
@click.option('-n', '--network', type=click.Choice(['pbgnet', 'pbgnet_ll', 'baseline', 'pbcombinet', 'pbcombinet_ll']),
              default='pbgnet', help="Name of the network architecture to use.")
@click.option('--hidden-size', type=int, default=10, help="Size of the hidden layers (number of neurons).")
@click.option('--hidden-layers', type=int, default=1, help="Number of hidden layers (depth of the network).")
@click.option('--sample-size', type=int, default=100, help="Sample size T for stochastic approximation of PBGNet.")
@click.option('--weight-decay', type=float, default=0, help="Weight decay (L2 penalty).")
@click.option('--prior', type=click.Choice(['zero', 'init', 'pretrain']), default='init', help="Prior distribution P.")
@click.option('--learning-rate', type=float, default=0.01, help="Learning rate.")
@click.option('--lr-patience', type=int, default=20, help="Learning rate scheduler patience before halving.")
@click.option('--optim-algo', type=click.Choice(['sgd', 'adam']), default='adam', help="Optimization algorithm.")
@click.option('--epochs', type=int, default=10, help="Maximum number of epochs.")
@click.option('--batch-size', type=int, default=8, help="Batch size.")
@click.option('--valid-size', type=float, default=0.2, help="Validation set size (pretrain set size when pretraining).")
@click.option('--pre-epochs', type=int, default=5, help="Pretrain number of epochs.")
@click.option('--stop-early', type=int, default=0, help="Early stopping patience.")
@click.option('--gpu-device', type=int, default=0, help="GPU device id to run on.")
@click.option('--random-seed', type=int, default=42, help="Random seed for reproducibility.")
@click.option('--logging', type=bool, default=True, help="Logging flag.")

dataset = "breast"
experiment_name = "test"
network = click.Choice(['pbgnet', 'pbgnet_ll', 'baseline', 'pbcombinet', 'pbcombinet_ll'])
hidden_size = 10
hidden_layer = 1
sample_size = 100
weight_decay = 0
prior = click.Choice(['zero', 'init', 'pretrain'])
learning_rate = 0.01
lr_patience = 20
optim_algo = click.Choice(['sgd', 'adam'])
epochs = 10
batch_size = 8
valid_size = 0.2
pre_epochs = 5
stop_early = 0
gpu_device = 0
random_seed = 42
logging = True
