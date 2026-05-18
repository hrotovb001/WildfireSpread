def format_train_description(epoch, total_epochs, loss):
    return f"Epoch {epoch+1}/{total_epochs} train loss: {loss:.4f}"

def format_val_description(epoch, total_epochs, val_loss):
    return f"Epoch {epoch+1}/{total_epochs} val loss: {val_loss:.4f}"

def format_results(train_loss, val_loss):
    return f"Training completed. Final train loss: {train_loss:.4f}, final val loss: {val_loss:.4f}"
