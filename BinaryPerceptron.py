from scripts.getdata import read_data
from lib.models import BinaryPerceptron
import matplotlib.pyplot as plt

def main():
	# get Fashin MNIST data 
	datapath = "data/fashion-mnist_train.csv"
	X_train, X_test, y_train, y_test = read_data(datapath)
	
	# train binary perceptron classifier
	BinaryPerceptron_model = BinaryPerceptron(iteration = 50)
	BinaryPerceptron_model.fit(X_train, y_train)
	
	# train binary passive-aggressive classifier
	BinaryPA_model = BinaryPA(iteration = 50)
	BinaryPA_model.fit(X_train, y_train)
	
	# plot results
	plt.plot(BinaryPerceptron_model.errors_list, label = 'perceptron')
	plt.plot(BinaryPA_model.errors_list, label = 'PA')
	plt.title('Binary Classification - number of iterations vs. number of mistakes')
	plt.xlabel('Number of iteration')
	plt.ylabel('Number of classification mistakes')
	plt.legend()
	plt.grid()
	plt.savefig('test.png')
	
if __name__ == '__main__':
    main()