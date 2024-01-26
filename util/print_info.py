def print_epoch_end(current_epoch, current_lr, current_train_loss, current_accuracy, output_file):
    print("Epoch [{}], lr: {:.6f}, train_loss: {:.4f}, accuracy: {:.2f}".format(
            current_epoch, current_lr, current_train_loss, current_accuracy), file=output_file)