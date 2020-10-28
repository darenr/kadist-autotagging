# kadist-autotagging
Assign tags to Kadist artworks

```
conda create -n kadist3 python=3.6
conda activate kadist3
pip install -r requirements.txt
```

Get latest `kadist.json`

then create the assignments:

```
python assign_kadist_clusters.py

```

make sure the kernel is available to the notebook:

```
python -m ipykernel install --user --name=kadist3

```

then open the notebook:

```
jupyter notebook "Kadist Tagged Corpus-2020.ipynb"

```
