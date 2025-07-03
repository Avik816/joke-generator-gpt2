import matplotlib.pyplot as mplot
from ..CONFIG import OUTPUT_DIR_PLOT


def plot_learning_curve(history):
        
    # Plotting training vs validation loss
    mplot.plot(history.history['loss'], label='Training Loss')
    mplot.plot(history.history['val_loss'], label='Validation Loss')
    mplot.title('Loss Curve')
    mplot.xlabel('Epochs')
    mplot.ylabel('Loss')
    mplot.legend()
    mplot.grid(True)
    mplot.tight_layout()
    
    # Saving the file
    mplot.savefig(f'{OUTPUT_DIR_PLOT}/Model learning curve_loss-plot.png')
    print(f'Plot is saved at {OUTPUT_DIR_PLOT} !')