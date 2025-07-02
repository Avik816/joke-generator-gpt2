import matplotlib.pyplot as mplot

# Plot training vs validation loss
mplot.plot(history.history['loss'], label='Training Loss')
mplot.plot(history.history['val_loss'], label='Validation Loss')
mplot.title("Loss Curve")
mplot.xlabel("Epochs")
mplot.ylabel("Loss")
mplot.legend()
mplot.grid(True)
mplot.tight_layout()
mplot.show()
