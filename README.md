# Rabbit Garden Traverser
## Usage
`python main.py`
If you want there are the following optional arguments:
- r: how many rows in the garden.
- c: how many columns in the garden.
- m: the maximum random integer valued assigned in garden.
- v: whether or not to be verbose. This will print an updated garden at each step as the rabbit traverses. Pretty interesting.

Ie, you can do:
`python main.py -r 5 -c 3 -m 10 -v True`

### Notes
At the moment there is no way to supply a matrix of your choosing to the code. Instead it will generate on randomly, however you can bound the randomness as mentioned above.

## Dependencies
Appears to work in both Python 2 and 3
```
numpy
argparse
```

Code Challenge for Trace Genomics
