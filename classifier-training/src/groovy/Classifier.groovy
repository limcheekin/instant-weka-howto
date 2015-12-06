import weka.core.converters.ConverterUtils.DataSource
import weka.classifiers.bayes.NaiveBayesUpdateable
import weka.classifiers.functions.SMO
import weka.classifiers.trees.J48
import weka.core.Instance
import weka.core.Instances
import weka.core.converters.ArffLoader

DataSource source = new DataSource("../dataset/titanic.arff")
Instances dataset = source.dataSet
dataset.classIndex = dataset.numAttributes() - 1

// decision trees
String[] options = new String[1]
options[0] = "-U"
J48 tree = new J48()
tree.options = options
tree.buildClassifier(dataset)	
println tree


// support vector machines
SMO svm = new SMO()
svm.buildClassifier(dataset)
println svm


//incremental Naive Bayes classifier
ArffLoader loader = new ArffLoader()
loader.file = new File("../dataset/titanic.arff")
Instances dataStructure = loader.structure
dataStructure.classIndex = dataStructure.numAttributes() - 1
NaiveBayesUpdateable nb = new NaiveBayesUpdateable()
nb.buildClassifier(dataStructure)
Instance current
while ((current = loader.getNextInstance(dataStructure)))
	nb.updateClassifier(current)
println(nb)
		