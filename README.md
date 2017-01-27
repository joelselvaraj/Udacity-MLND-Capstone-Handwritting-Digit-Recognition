# Machine Learning Engineer Nanodegree
## Capstone Project: Handwritten Digit Recognition using Deep Neural Network

### Install

This project requires **Python 2.7** and the following Python libraries installed:

- [Anaconda2](https://www.continuum.io/downloads)
- [Tensorflow](https://www.tensorflow.org/) or [Theano](http://www.deeplearning.net/software/theano/)
- [Keras](https://keras.io/)
- [OpenCV](http://opencv.org/)
- (Optional) A GPU with Cuda support is highly recommended for fast computation. Install respective graphics drivers, cuda toolkit, cudnn, [tensorflow-gpu](https://www.tensorflow.org/get_started/os_setup) version

You will also need to have software installed to run and execute a [Jupyter Notebook](http://ipython.org/notebook.html)

If you do not have Python installed yet, it is highly recommended that you install the [Anaconda](http://continuum.io/downloads) distribution of Python, which already has the above packages and more included. Make sure that you select the Python 2.7 installer and not the Python 3.x installer.

### Code

The code is provided in the `handwritten_digit_recognizer.ipynb` notebook file. The MNIST dataset required for this project will be downloaded and loaded in the code itself. No need to download manually.

### Run

In a terminal or command window, navigate to the top-level project directory `Udacity-MLND-Capstone-Handwritting-Digit-Recognition/` (that contains this README) and run one of the following commands:

```bash
ipython notebook handwritten_digit_recognizer.ipynb
```  
or
```bash
jupyter notebook handwritten_digit_recognizer.ipynb
```

This will open the Jupyter Notebook software and project file in your browser.

### Data

The MNIST dataset required for this project will be downloaded and loaded in the code itself. No need to download manually. This dataset has the following description:

The	MNIST	database of	handwritten	digits	has	a	training	set	of	60,000	examples,	and	a	test	
set	of	10,000	examples.	It	is	a	subset	of	a	larger	set	available	from	NIST.	The	digits	have	been	
size-normalized	and	centered	in	a	fixed-size	image.	

The	original	black	and	white	(bi-level)	images	from	NIST	were	size	normalized	to	fit	in	a	20x20	
pixel	box	while	preserving	their	aspect	ratio.	The	resulting	images	contain	grey	levels	as	a	result	
of	the	anti-aliasing	technique	used	by	the	normalization	algorithm.	the	images	were	centered	in	
a	28x28	image	by	computing	the	center	of	mass	of	the	pixels,	and	translating	the	image	so	as	to	
position	this	point	at	the	center	of	the	28x28	field.	

The	dataset	can	be	downloaded	and	loaded	using	Keras,	a	high-level	neural	networks	python	
library	using	the	following	code	
```	
from	keras.datasets	import	mnist	
(X_train,	y_train),	(X_test,	y_test)	=	mnist.load_data()	
```	
- X_train,X_test:	uint8	array	of	grayscale	image	split	as	train	and	test	data	
- y_train,y_test:	uint8	array	of	digit	labels	(integers	in	range	0-9)	split	as	train	and	test data	

Thus	we	can	use	the	above	variables	to	train	the	model	with	pixel-values	of	the	handwritten	
image	as	features(X_train)	and	its	respective	label(y_train)	as	target.	We	can	also	test	the	
model	by	evaluating	it	against	the	test	variables	X_test	and	y_test.	

