def model_fn(model_dir):
    """Load model from file for SageMaker"""
    logger.info("Loading the model.")
    model = ImageClassifier()
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        model.load_state_dict(torch.load(f))
    return model.to(device)


def save_model(model, model_dir):
    """Save model to file for SageMaker"""
    logger.info("Saving the model.")
    path = os.path.join(model_dir, "model.pth")
    # Recommended way from http://pytorch.org/docs/master/notes/serialization.html
    torch.save(model.cpu().state_dict(), path)


def save_checkpoint(
    checkpoint_dir,
    run_name=None,
    checkpoint_name=None,
    model=None,
    optimizer=None,
    epoch=0,
    loss=None,
    args=None,
):
    """Save checkpoint"""
    logger.info("Saving checkpoint...")

    # Get checkpoint path
    run_checkpoint_dir = Path.cwd() / checkpoint_dir / run_name
    run_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_checkpoint_dir / checkpoint_name
    config_path = run_checkpoint_dir / "config.json"

    checkpoint = {
        "epoch": epoch,
        "loss": loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": args,
    }

    with open(config_path, "w") as f:
        json.dump(vars(args), f, sort_keys=True, indent=4, separators=(",", ": "))

    # Recommended way from http://pytorch.org/docs/master/notes/serialization.html
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(checkpoint, model, optimizer):
    """Load checkpoint
    checkpoint is the path of the checkpoint
    """
    logger.info("Loading checkpoint")
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    return model, optimizer, epoch, loss
