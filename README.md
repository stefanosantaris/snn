## Create virtual environment
python3 -m venv env

## Activate virtual environment
. env/bin/activate

## Install requirements
pip install -r requirements.txt

## Split file
cat .tmp/train.csv | parallel --header : --pipe -N500000 'cat > .tmp/file_{#}.csv'


## Avazu Dataset URL
https://www.kaggle.com/c/avazu-ctr-prediction

## Criteo Dataset URL
https://ailab.criteo.com/criteo-uplift-prediction-dataset/
