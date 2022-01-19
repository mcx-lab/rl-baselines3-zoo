from functools import partial
import argparse
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
from ray import tune


DATA_DIR = os.path.join(os.getcwd(), "./blind_walking/examples/data/heightmap.npy")
single_data_shape = (7, 20)


class LinearAE(torch.nn.Module):
    def __init__(self, input_size=140, code_size=32):
        super().__init__()
        # encoder
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_size, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, code_size),
        )
        # decoder
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(code_size, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, input_size),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def load_data():
    # load dataset
    dataset_np = np.load(DATA_DIR)
    single_data_size = len(dataset_np[0])
    assert np.prod(single_data_shape) == single_data_size

    # shuffle dataset
    np.random.seed(12)
    np.random.shuffle(dataset_np)

    # split into train, test, validation sets
    train_size = int(0.8 * len(dataset_np))
    val_size = int(0.1 * len(dataset_np))
    train_dataset_np = dataset_np[:train_size, :]
    val_dataset_np = dataset_np[train_size : train_size + val_size, :]
    test_dataset_np = dataset_np[train_size + val_size :, :]

    # make tensor dataset
    train_dataset = torch.utils.data.TensorDataset(torch.Tensor(train_dataset_np))
    val_dataset = torch.utils.data.TensorDataset(torch.Tensor(val_dataset_np))
    test_dataset = torch.utils.data.TensorDataset(torch.Tensor(test_dataset_np))

    return (train_dataset, val_dataset, test_dataset), (single_data_size, single_data_shape)


def train_model(config, checkpoint_dir=None, tune=True):
    # load datasets
    dataset_and_info = load_data()
    train_dataset, val_dataset, _ = dataset_and_info[0]
    single_data_size, single_data_shape = dataset_and_info[1]
    # datasets loader used for training and validation
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
    )

    # model initialisation
    model = LinearAE(input_size=single_data_size, code_size=config["code_size"])
    # use gpu if available
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    model.to(device)
    # loss function and optimizer
    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    # load checkpoint if provided
    if checkpoint_dir:
        model_state, optimizer_state = torch.load(os.path.join(checkpoint_dir, "checkpoint"))
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    min_val_loss = 100  # artbitrary high number
    val_grace = 4  # number of grace times for training to continue
    epochs = 10000
    for epoch in range(epochs):
        train_loss = 0
        for batch_data in train_loader:
            # load mini-batch data to the active device
            batch_data = batch_data[0].view(-1, single_data_size).to(device)
            # reset the gradients back to zero
            optimizer.zero_grad()
            # compute reconstructions
            outputs = model(batch_data)
            # compute loss
            loss = loss_function(outputs, batch_data)
            # compute accumulated gradients
            loss.backward()
            # perform parameter update based on current gradients
            optimizer.step()
            # add the mini-batch training loss to epoch loss
            train_loss += loss.item()

        # compute and display the epoch training loss
        train_loss = train_loss / len(train_loader)
        if not tune and epoch % 100 == 0:
            print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, train_loss))

        # validation
        val_loss = 0
        for batch_data in val_loader:
            with torch.no_grad():
                batch_data = batch_data[0].view(-1, single_data_size).to(device)
                outputs = model(batch_data)
                loss = loss_function(outputs, batch_data)
                val_loss += loss.item()
        val_loss = val_loss / len(val_loader)
        # early stopping
        if epoch % 100 == 0 and val_loss > min_val_loss * 1.1:
            val_grace -= 1
            if val_grace < 0:
                break
        min_val_loss = min(val_loss, min_val_loss)

        # report to ray tune
        if tune:
            with tune.checkpoint_dir(epoch) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, "checkpoint")
                torch.save((model.state_dict(), optimizer.state_dict()), path)
            tune.report(loss=val_loss)

    if not tune:
        # save pytorch model
        torch.save(
            (model.state_dict(), optimizer.state_dict()),
            f"./autoenc_results/model_bs{config['batch_size']}_cs{config['code_size']}_lr{config['lr']}",
        )
    print("Finished Training")


def test_model(model, device="cpu"):
    # load datasets
    dataset_and_info = load_data()
    _, _, test_dataset = dataset_and_info[0]
    single_data_size, single_data_shape = dataset_and_info[1]
    # datasets loader used for testing
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=4,
        shuffle=False,
    )
    # loss function
    loss_function = torch.nn.MSELoss()
    # test
    test_loss = 0
    for batch_data in test_loader:
        with torch.no_grad():
            batch_data = batch_data[0].view(-1, single_data_size).to(device)
            outputs = model(batch_data)
            loss = loss_function(outputs, batch_data)
            test_loss += loss.item()
    # render some test images
    n_test_render = 5
    test_images = test_dataset[:n_test_render][0]
    recon_images = model(test_images.reshape(-1, single_data_size).to(device))
    recon_images = recon_images.detach().cpu().numpy()
    fig, axes = plt.subplots(n_test_render, 2, figsize=(6, 6))
    for i, test_image in enumerate(test_images):
        axes[i, 0].imshow(test_image.reshape(*single_data_shape))
        axes[i, 1].imshow(recon_images[i].reshape(*single_data_shape))
    plt.savefig("./autoenc_results/test_images.png")
    plt.close()
    # return loss
    return test_loss / len(test_loader)


def hyperparam_search():
    config = {
        "batch_size": tune.choice([32, 64, 128]),
        "code_size": tune.choice([16, 32, 64]),
        "lr": tune.loguniform(1e-4, 1e-1),
    }
    scheduler = tune.schedulers.ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=10000,
        grace_period=10,
        reduction_factor=2,
    )
    reporter = tune.CLIReporter(
        parameter_columns=["batch_size", "code_size", "lr"],
        metric_columns=["loss", "training_iteration"],
    )
    result = tune.run(
        partial(train_model),
        config=config,
        num_samples=100,
        scheduler=scheduler,
        progress_reporter=reporter,
        local_dir=os.path.join(os.getcwd(), "./autoenc_results"),
    )

    # print best trial results
    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(best_trial.last_result["loss"]))

    # use gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # create model with best performing hyperparameters
    best_trained_model = LinearAE(input_size=np.prod(single_data_shape), code_size=best_trial.config["code_size"]).to(device)
    # load model parameters from checkpoint
    best_checkpoint_dir = best_trial.checkpoint.value
    model_state, optimizer_state = torch.load(os.path.join(best_checkpoint_dir, "checkpoint"))
    best_trained_model.load_state_dict(model_state)
    best_trained_model.eval()
    # test model
    test_loss = test_model(best_trained_model, device)
    print("test loss = {:.6f}".format(test_loss))


def single_train_run():
    config = {
        "batch_size": 32,
        "code_size": 32,
        "lr": 1e-3,
    }

    """ Train """
    train_model(config, tune=False)

    """ Test """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LinearAE(input_size=np.prod(single_data_shape), code_size=config["code_size"]).to(device)
    # load trained model
    model_state, optimizer_state = torch.load(
        f"./autoenc_results/model_bs{config['batch_size']}_cs{config['code_size']}_lr{config['lr']}"
    )
    model.load_state_dict(model_state)
    model.eval()
    # test model
    test_loss = test_model(model, device)
    print("test loss = {:.6f}".format(test_loss))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hyper", action="store_true", default=False, help="Hyperparameter search")
    args = parser.parse_args()

    if args.hyper:
        hyperparam_search()
    else:
        single_train_run()
