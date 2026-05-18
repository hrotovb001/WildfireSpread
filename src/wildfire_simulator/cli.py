def format_train_description(epoch, loss):
    return f"Epoch {epoch} train loss: {loss:.4f}"

def format_val_description(val_loss):
    return f"val loss: {val_loss:.4f}"

def format_results(summary):
    return (f"Training completed. "
            f"Best epoch: {summary['best_epoch']}, "
            f"Train loss: {summary['train_loss']:.4f}, "
            f"Val loss: {summary['val_loss']:.4f}, "
            f"Duration: {summary['duration_seconds']:.1f}s")
