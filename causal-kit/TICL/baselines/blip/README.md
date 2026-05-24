# blip

This is the "Bayesian network Learning Improved Project" (blip), an open-source Java package that offers a wide range of structure learning algorithms.
It is developed my Mauro Scanagatta and it is distributed under the LGPL-3 by IDSIA. 

It focuses on score-based learning, mainly the BIC and the BDeu score functions, and
allows the user to learn BNs from datasets containing thousands of variables. It provides state-of-the-art algortihms for the following tasks: parent set identification ( BIC ), general structure optimization (WINASOBS-ENT), bounded treewidth structure optimization (KMAX) and structure learning on incomplete data sets (SEM-KMAX). 

An R binding is also available: (https://github.com/mauro-idsia/r.blip).

## References

This package implements the algorithms detailed in the following papers: 
* [Learning Bayesian Networks with Thousands of Variables](https://papers.nips.cc/paper/5803-learning-bayesian-networks-with-thousands-of-variables) (NIPS 2015) Mauro Scanagatta, Giorgio Corani, Cassio P. de Campos, Marco Zaffalon
* [Learning Treewidth-Bounded Bayesian Networks with Thousands of Variables](https://papers.nips.cc/paper/6232-learning-treewidth-bounded-bayesian-networks-with-thousands-of-variables) (NIPS 2016) Mauro Scanagatta, Giorgio Corani, Cassio P. de Campos, Marco Zaffalon
* [Efficient learning of bounded-treewidth Bayesian networks from complete and incomplete data sets](https://www.sciencedirect.com/science/article/pii/S0888613X17307272) (IJAR 2018) - [supplementary material](supplementary-IJAR.pdf)
* [Improved Local Search in Bayesian Networks Structure Learning](http://proceedings.mlr.press/v73/scanagatta17a.html) (AMBN 2017)
* [Approximated Structural Learning for Large Bayesian Networks](https://link.springer.com/article/10.1007/s10994-018-5701-9) (ECML PKDD 2018) [supplementary material](supplementary-ML17.pdf)


## Usage

The process of learning a bounded-treewidth BN is explained by using the "child" network as example.

### Dataset format

The format for the initial dataset has to be the same as the file "child-5000.dat", namely a space-separated file containing: 

    * First line: list of variables names, separated by space;
    * Second line: list of variables cardinalities, separated by space;
    * Following lines: list of values taken by the variables in each datapoint, separated by space.

### Parent set identification 

The first step is build the parent sets score cache. The state-of-the-art approach is to use BIC* (for the BIC score): 

```
java -jar blip.jar scorer.is -d data/child-5000.dat -j data/child-5000.jkl -t 10 -b 0 
```

Main options: 
* -d VAL : Datafile input path (.dat format)
* -j VAL : Parent set scores output file (.jkl format)
* -t N   : Maximum time limit, in seconds (default: 10)
* -b N   : Number of machine cores to use - if 0, all are used  (default: 1)

### General structure optimization 

Given the parent sets score cache, now it is time to learn the structure. The state-of-the-art approach is to use WINASOBS (Windows operator applied to ASOBS) with ENT (entropy-based) ordering: 

```
java -jar blip.jar solver.winasobs.adv -smp ent -d data/child-5000.dat -j data/child-5000.jkl -r data/child.wa.res -t 10 -b 0
```

Main options: 
* -smp VAL : Advanced sampler (possible values: std, mi, ent, r_mi, r_ent)
* -d VAL : Datafile input path (.dat format)
* -j N   : Parent set scores input file (.jkl format)
* -r VAL : Structure output file (.res format)
* -t N   : Maximum time limit, in seconds (default: 10)
* -b N   : Number of machine cores to use - if 0, all are used  (default: 1)

### Bounded-treewidth structure optimization 

Given the parent sets score cache, it is possible to learn a structure under a bounded treewidth constraints. The state-of-the-art approach is to use k-max: 

For perfoming with k-max:

```
java -jar blip.jar solver.kmax -w 4 -j data/child-5000.jkl -r data/child-5000.kmax.res -t 10 -b 0
```

Main options: 
*  -w N  : Maximum treewidth allowed
* -j N   : Parent set scores input file (.jkl format)
* -r VAL : Structure output file (.res format)
* -t N   : Maximum time limit, in seconds (default: 10)
* -b N   : Number of machine cores to use - if 0, all are used  (default: 1)

### Structure learning from incomplete data sets

To learn a structure from data containing missing values the state-of-the-art approach is to use SEM-kMAX: 

```
java -jar blip.jar imputation.sem  -d data/child-5000-missing.dat -o data/child-5000-imputed.dat -r data/child.res -t 1 -tmp data/tmp -w 6 -b 0
```

Main options: 
* -d VAL   : Datafile (with missing valus) input path (.dat format)
* -o VAL   : Datafile (with imputed values) output path (.dat format)
* -r VAL   : Structure output file (.res format)
* -t N     : Time regulation parameter (default: 1)
* -tmp VAL : Temporary directory
* -w N     : Learning treewidth (default: 6)
* -b N     : Number of machine cores to use - if 0, all are used  (default: 1)

### Interpreting the result 

The format of the ".res" file is as follows: each line indicates the parent set assigned to each variable and its score.

For example the line "4: -2797.39 (10,17,18)" indicates that to the variable with index 4 in the dataset are assgined as parents the variables with index (10,17,18). This parent set has score -2797.39 (by default the score function is the BIC). 

### Learn the parameters

Using the structure found it is possible to learn the parameters with: 

```
java -jar blip.jar parle -d data/child-5000.dat -r data/child-5000.kmax.res -n data/child-5000.kmax.uai
```

Main options: 
* -d VAL  : Datafile input path (.dat format)
* -r VAL  : Structure input file (.res format)
* -n VAL  : BN output file (.uai format) 

The final output will be a full Bayesian network in UAI format. 
 
