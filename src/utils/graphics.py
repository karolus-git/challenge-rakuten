import pandas as pd
import seaborn as sns
import tensorflow as tf
import io
import matplotlib.pyplot as plt

def plot_cm(y_true, y_pred):

    cm = pd.crosstab(
            y_true, y_pred,
            rownames=['Read'], colnames=['Predicted'],
            normalize="index"
        )
    hm = sns.heatmap(cm, 
        annot=False, 
        linewidth=.5, 
        vmin=0, vmax=1,
        cmap=sns.cubehelix_palette(as_cmap=True))
    fig = hm.get_figure()
    
    return fig

def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image