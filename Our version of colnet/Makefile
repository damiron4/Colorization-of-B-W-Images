setup:
	mkdir -p data out model

download:
	@echo "Downloading dataset ..."
	wget -nc http://data.csail.mit.edu/places/places365/places365standard_easyformat.tar
	@echo "... done."
    	
extract:
	@echo "Extracting specific folders to data/ (if they don't exist) ..."
	tar --keep-old-files -xf places365standard_easyformat.tar \
		-C data/ \
		places365_standard/val/botanical_garden \
		places365_standard/val/formal_garden \
		places365_standard/val/japanese_garden \
		places365_standard/val/roof_garden \
		places365_standard/val/topiary_garden \
		places365_standard/val/vegetable_garden \
		places365_standard/val/zen_garden \
		places365_standard/val/lawn \
		places365_standard/val/house \
		places365_standard/val/cottage \
		places365_standard/train/botanical_garden \
		places365_standard/train/formal_garden \
		places365_standard/train/japanese_garden \
		places365_standard/train/roof_garden \
		places365_standard/train/topiary_garden \
		places365_standard/train/vegetable_garden \
		places365_standard/train/zen_garden \
		places365_standard/train/lawn \
		places365_standard/train/house \
		places365_standard/train/cottage
	@echo "... done."



old_split:
	@echo 'Splitting dataset ...'
	# For each category:
	# 4096 pics from train/ will go to places5_old/train
	# 128 pics from train/ will go to places5_old/val
	python3 scripting/split-dataset.py data/places365_standard/train/ data/places5_old/ scripting/places5.txt train 4096 --bname val --bsize 128
	# 96 pics from val/ will go to places5_old/test
	python3 scripting/split-dataset.py data/places365_standard/val/ data/places5_old/ scripting/places5.txt test 96
	@echo '... done.'
	@echo 'Please run `make run` to train the network.'


new_split:
	@echo 'Splitting dataset ...'
	# For training set: 4096 pics will go to places5/train
	python3 scripting/split-dataset.py data/places365_standard/train/ data/places5/ scripting/places5.txt train 4096 --bname val --bsize 128
	# For validation set: 128 pics will go to places5/val
	python3 scripting/split-dataset.py data/places365_standard/val/ data/places5/ scripting/places5.txt val 128 --bname val --bsize 128
	# For test set: 96 pics will go to places5/test
	python3 scripting/split-dataset.py data/places365_standard/train/ data/places5/ scripting/places5.txt test 96
	@echo '... done.'
	@echo 'Please run `make run` to train the network.'

new_new_split:
	@echo 'Splitting dataset ...'
	# For training set: 4096 pics will go to places12/train
	python3 scripting/split-dataset.py data/places365_standard/train/ data/places12/ scripting/places12.txt train 4096 --bname val --bsize 128
	# For validation set: 128 pics will go to places12/val
	python3 scripting/split-dataset.py data/places365_standard/val/ data/places12/ scripting/places12.txt val 128 --bname val --bsize 128
	# For test set: 96 pics will go to places12/test
	python3 scripting/split-dataset.py data/places365_standard/train/ data/places12/ scripting/places12.txt test 96
	@echo '... done.'
	@echo 'Please run `make run` to train the network.'

split:
	@echo 'Splitting dataset ...'
	# For training set: 4096 pics will go to places10/train
	python3 scripting/split-dataset.py data/places365_standard/train/ data/places10/ scripting/places10.txt train 4096 --bname val --bsize 128
	# For validation set: 128 pics will go to places10/val
	python3 scripting/split-dataset.py data/places365_standard/val/ data/places10/ scripting/places10.txt val 128 --bname val --bsize 128
	# For test set: 96 pics will go to places10/test
	python3 scripting/split-dataset.py data/places365_standard/train/ data/places10/ scripting/places10.txt test 96
	@echo '... done.'
	@echo 'Please run `make run` to train the network.'



dataset: setup download extract
	@echo "Downloading, extracting and splitting dataset."
	@echo "Run `make split` to split the dataset"

run:
	python3 loader.py config/places10.yaml


zip_out:
	zip --quiet --recurse-paths out.zip out/
	#rm -rf out/*
	@echo "Zipped all files in ./out into out.zip"

clean:
	rm -rf __pycache__/
	rm -rf src/__pycache__/
	rm colorized-*.jpg


places16:
	@echo 'Places16 - splitting dataset ...'
	# For each category:
	# 4096 pics from train/ will go to places16/train
	#  128 pics from train/ will go to places16/val
	python3 scripting/split-dataset.py data/places365_standard/train/ data/places16/ scripting/places16.txt train 4096 --bname val --bsize 128
	#   96 pics from val/   will go to places16/test
	python3 scripting/split-dataset.py data/places365_standard/val/ data/places16/ scripting/places16.txt test 96
	@echo  '... done.'
	@echo 'Please run `python3 loaader.py config/places16.yaml` to train the network.'
    
    
places13:
	@echo 'Places13 - splitting dataset ...'
	# For each category:
	# 4096 pics from train/ will go to places13/train
	#  128 pics from train/ will go to places13/val
	python3 scripting/split-dataset.py data/places365_standard/train/ data/places13/ scripting/places13.txt train 4096 --bname val --bsize 128
	#   96 pics from val/   will go to places13/test
	python3 scripting/split-dataset.py data/places365_standard/val/ data/places13/ scripting/places13.txt test 96
	@echo  '... done.'
	@echo 'Please run `python3 loaader.py config/places13.yaml` to train the network.'
