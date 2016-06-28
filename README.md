## Robust Fisher Linear Discriminant Analysis (<a href="https://stanford.edu/~boyd/papers/pdf/robust_FDA.pdf">Paper</a>)

This is a collaborative project by Chen LIU and Jun LU.

We use two datasets, both of which are from UCI. They are [sonar](https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data) and [ionosphere](https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data)

In addition, in order to solve the QCQP problem raised by robust LDA, we take advantage of the python library cvxopt and scipt writtern by <a href="http://pages.cs.wisc.edu/~kline/qcqp/qcqprel_py">Jeffery Kline</a>

## Code Structure
```
|-sonar              # the sonar dataset
|-ionosphere         # the ionosphere dataset
|-result             # the result file runs on 10 random samples, used in mainTest.py to show the figure
|-load.py            # load datas
|-FisherLDA.py       # general fisher LDA
|-robustFisherLDA.py # robust fisher LDA 
|-mainTest.py        # test on general fisher LDA and robust fisher LDA
|-log.py  # if you do not want to see so many log infos, please change the DEBUG to False
|-util.py # util functions
|-plotData1.py & plotData2.py       # project the dataset onto 3-d, and plot
|-requirements.txt   # contains the requried packages
```
