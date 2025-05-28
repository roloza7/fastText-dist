## FastText-dist

An attempt at extending the [Hogwild!](https://people.eecs.berkeley.edu/~brecht/papers/hogwildTR.pdf) training algorithm used by [FastText](https://fasttext.cc) to distribute over multiple nodes.

## Method

The patch adds a periodic model "souping" step every 64 iterations, (performed by the main thread of each node). This reduces performance to some extent but at the same allows us to scale training by having more input streams.

This is not equivalent (and does not offer the same guarantees) as standard HogWild!, HogWild++, or synchronized SGD, but is more convenient if have access to several nodes.

#### Examples:

- 1 Node with 24 threads → 1 File is split into 24 input streams with fseek
- 2 Nodes with 24 threads each → 1 File is split into 48 input streams with fseek (half the number of iterations per thread)
- And so on...

## Installation

This patch has been tested with `openmpi` installed (both on my local machine and on a HPC IB-capable environment) and uses the `mpic++` compiler.

Unlike the original FastText, there are no official releases. To install it, simply:

```
git clone https://github.com/roloza7/fastText-dist
cd fastText-dist
make
```

## Usage

Identical to FastText, with the addition of a single argument `-nodes N` that triggers periodic souping between nodes.

## Benchmarks

### Dataset

To benchmark this model, I prepare a large language classification dataset from the Wikimedia download files in the original fasttext repository. The final dataset has 644M entries and contains:

| Language   | Size (Uncompressed, GB) |
| ---------- | ----------------------- |
| German     | 11.34                   |
| English    | 29.26                   |
| Spanish    | 7.06                    |
| French     | 9.72                    |
| Japanese   | 5.77                    |
| Portuguese | 3.17                    |
| Chinese    | 2.71                    |
| **Total**  | 69.03                   |

After adding labels, shuffling, and splitting the dataset, we are left with 68.61GB of train data and 7.62 of test data.

For most of the benchmarks, I randomly sampled a subset (~20GB) of the full training dataset to acommodate single-node fastthread comparisons.



### Setup

Every benchmark is tested with the command:

```bash
srun --mpi=pmix time ./fasttext supervised -input <file> -output <model> \
-thread <nthreads> -lr 0.1 -epoch 5
```

For multi-node runs the command `-nodes N` is also included. 

### Runtime

Total runtime is reported (including the overhead of building the dictionary, and saving .bin and .vec files)

For Multi-Node setups, the time of rank 0 is reported, since rank 0 is the last to exit.

> ℹ️ **Note**
> In this system, all runs spent an estimate 8 minutes on saving the final output. Benchmarks were kept as simple as possible, but a more sensible minimum frequency parameter would cut model size (and save time) down.

| Model | Elapsed Time (mm:ss) | P@1 | R@1 |
| ----- | ---------------- | --------- | ------ |
| 1x24  | 35:51            | 0.881 | 0.881  |
| 4x24  | 19:10    | 0.862 | 0.862  |
| 8x24  | 16:04            | 0.868  | 0.868  |
| 16x24 | 14:32                | 0.830  | 0.830  |

While model quality is noticeably smaller with a fixed dataset, we can offset this by increasing the amount of ingested data (since we know have scaling). If your data is small, its probably best to stick to single-node training.

> 🚧 **Coming Soon**
> Benchmarks on distributed models trained on the full (68.61GB) of data

### Benchmark Environment

All nodes are identical, containing the following relevant features:

- Dual Intel Xeon Gold 6226 CPUs @ 2.7 GHz (24 cores/node)
- DDR4-2933 MHz DRAM (192GB)
- Infiniband 100HDR interconnect