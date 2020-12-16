#Original author jpcano1, all content belongs to him and used under authorization#

from . import general as gen
import numpy as np
import matplotlib.pyplot as plt

from skimage.segmentation import mark_boundaries

def get_labeled_image(img, label, outline_color=(1, 0, 0), 
                        color=(1, 0, 0), mode="outer"):
    """
    Function to get the labeled image
    :param img: The image
    :param label: The mask label
    :param outline_color: The color of outline
    :param color: The color of fill
    :return: The mask and the image merged
    """
    assert mode in ['thick', 'inner', 'outer', 'subpixel']
    img_mask = mark_boundaries(img, label, outline_color=outline_color, 
                               color=color, mode=mode)
    return img_mask

def predict(model, device, dataset, class_: str="kidney",
            random_state=1234, **kwargs):
    """
    Method to make predictions from a model
    :param model: The model that makes the prediction
    :param device: The hardware accelerator device
    :param dataset: The dataset to make the predictions
    :param class_: The class of the predictions
    :param random_state: The random seed
    """
    if class_ == "kidney":
        channel = 0
    elif class_ == "tumor":
        channel = 1
    else:
        raise Exception("No es la clase esperada")
    
    # Random seed
    np.random.seed(random_state)
    # Take the random sample
    random_sample = np.random.choice(len(dataset), 3)

    # Create the figure to plot on
    plt.figure(figsize=(12, 12))
    for i in range(len(random_sample)):
        rnd_idx = random_sample[i]
        # Take the image and the label
        X, y_true = dataset[rnd_idx]
        y_true = y_true[channel]
        # Extend dims and allocate on device
        X_t = X.unsqueeze(0).to(device)
        X = X.squeeze(0)
        # Predict
        y_pred = model(X_t)
        y_pred = y_pred.squeeze(0)
        y_pred = y_pred[channel].cpu().detach().numpy() > .5

        # Plot the results versus the originals
        plt.subplot(3, 4, 1 + i*4)
        gen.imshow(X, color=False, cmap="bone", title="Image")

        plt.subplot(3, 4, 2 + i*4)
        gen.imshow(y_pred, color=False, title=f"Predicted {class_.title()}")

        plt.subplot(3, 4, 3 + i*4)
        gen.imshow(get_labeled_image(X, y_pred, **kwargs), title="Boundary")
        
        plt.subplot(3, 4, 4 + i*4)
        gen.imshow(y_true, color=False, title="True Label")

def rugosity(image):
    if image.max() != 0:
        var = image.var()
        max = image.max() ** 2
        return 1 - (1 / (1 + var/max))
    else:
        return 0

#By Sebascal1      
def confusionMatrix(ds, model, device):
  conf_matrix = np.zeros([2,2])
  for i in range(len(ds)):
    X, y_true = ds[i]
    dim = X.shape[2]
    X_t = X.unsqueeze(0).to(device)
    X = X.squeeze(0)
    y_pred = model(X_t)

    true_positives = torch.sum(y_pred[0,1,:,:]* y_true[1,:,:].to(device))
    false_positives = abs(torch.sum(y_pred[0,1,:,:]) - true_positives)
    false_negatives = abs(torch.sum(y_true[1,:,:]).to(device) - true_positives)

    true_positives = true_positives.cpu().detach().numpy().round()
    false_positives = false_positives.cpu().detach().numpy().round()
    false_negatives = false_negatives.cpu().detach().numpy().round()

    true_negatives = (dim * dim) - true_positives - false_negatives - false_positives

    conf_matrix[0][0] += true_negatives
    conf_matrix[0][1] += false_positives
    conf_matrix[1][0] += false_negatives
    conf_matrix[1][1] += true_positives

   recall = conf_matrix[1][1]/(conf_matrix[1][0]+conf_matrix[1][1])
   precision = conf_matrix[1][1]/(conf_matrix[0][1]+conf_matrix[1][1])
   accuracy = (conf_matrix[1][1] + conf_matrix[0][0])/(conf_matrix[1][1] + conf_matrix[0][0] + conf_matrix[1][0] + conf_matrix[0][1])

  return conf_matrix, recall, precision, accuracy
