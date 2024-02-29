# 🖌️ Automatic Image Colorization

This is an enhanced implementation of the [_Let there be Color!_](http://iizuka.cs.tsukuba.ac.jp/projects/colorization/en/) model by Satoshi Iizuka, Edgar Simo-Serra, and Hiroshi Ishikawa, modified by the research team at Nazarbayev University. Our adaptation, named ColNet+, narrows down the focus from the original 365 classes to an optimized set of 12 classes, integrating a dropout layer on low-level features to prevent overfitting, and incorporating global average pooling to replace the first two fully connected layers for a more robust and generalized feature representation.

![Colorized Książ Castle, Poland](colorized/ksiaz-castle.png "Colorized Książ Castle, Poland")

[More images](colorized/colorized.md)

This project has been updated as discussed in our latest research paper from Nazarbayev University. The following files have been added to extend the capabilities of our colorization framework:

- `stats.py`: for generating statistical data about colorization results.
- `comparison.py`: to compare the performance and results of different colorization models.
- `tsne.py`: for dimensionality reduction and visualization of color spaces.
- `colorize.py`: have been updated in the plot_dl folder to colorize images without google colab, plot the results and see the classification results   
## First Run

[Places12-Standard](http://places365.csail.mit.edu/download.html) dataset will be downloaded and split into _train/dev/test_ subsets with a focused set of 12 categories. Here the original model have 365 classes but we only filtered out the below classes due to computational limitations:
BEACH
HOSPITAL_ROOM
BOOKSTORE
FIELD-CULTIVATED
ZEN_GARDEN
TOPIARY_GARDEN
BAR
DORM_ROOM
CAFETERIA
ARCADE
FOUNTAIN
TOWER

```bash
$ git clone https://github.com/antechsky/colnet-modified.git
$ cd colnet-modified
$ make dataset
'''If you want to split data only with 12 clases run the command below''':
$ make new_new_split 
'''Otherwise run normal command below''' 
$ make split
```

## Requirements

Code is written in Python 3.6.3. All requirements can be installed via:

```bash
pip3 install -r requirements.txt
```

## Network Training

To train the ColNet+ network with new configurations:

```bash
$ python3 loader.py config/places12.yaml
```

Model checkpoints are saved every epoch and training can be resumed with a saved model:

```bash
$ python3 loader.py config/places12.yaml --model model.pt
```

## Colorize!

To colorize an image with your chosen model:

```bash
$ python3 colorize.py img.jpg ./models/places12.pt
```

Please refer to the changes and methodologies discussed in our paper for a deeper understanding of these updates.

---

The research team at Nazarbayev University is dedicated to advancing the field of automatic image colorization. We appreciate contributions from the community and invite collaboration. 
