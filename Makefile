train_bayer:
	python bin/train.py data/images/train/filelist.txt output/bayer --val_data data/images/val/filelist.txt

train_bayer:
	echo "nothing yet"

setup: build download_data

build:
	pip install -r requirements.txt

# Launch a server to visualize training (port 8097)
server:
	python -m 'visdom.server'

download_data: data
	cd data && wget https://data.csail.mit.edu/graphics/demosaicnet/download_dataset.py && \
		python download_dataset.py

data:
	mkdir - p data
