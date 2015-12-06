import weka.core.Instance
import weka.core.Instances
import weka.classifiers.functions.GaussianProcesses
import weka.classifiers.functions.LinearRegression
import weka.classifiers.functions.MultilayerPerceptron
import weka.classifiers.functions.SMOreg
import weka.classifiers.rules.ZeroR
import weka.classifiers.trees.REPTree

//load data
Instances data = new Instances(new BufferedReader(new FileReader("../dataset/house.arff")))
data.classIndex = data.numAttributes() - 1

//build model
//ZeroR zModel = new ZeroR();
//LinearRegression lrModel = new LinearRegression();
//REPTree treeModel = new REPTree();
//SMOreg svmModel = new SMOreg();
//MultilayerPerceptron mlpModel = new MultilayerPerceptron();

GaussianProcesses gpModel = new GaussianProcesses()
gpModel.buildClassifier(data) //the last instance with missing class is not used
println gpModel


//classify the last instance
Instance myHouse = data.lastInstance()
double price = gpModel.classifyInstance(myHouse)
println "My house ($myHouse): $price"