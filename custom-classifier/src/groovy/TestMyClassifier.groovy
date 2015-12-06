import weka.core.converters.ConverterUtils.DataSource
import weka.core.Instances

DataSource source = new DataSource("../dataset/titanic.arff")
Instances dataset = source.dataSet
dataset.classIndex = dataset.numAttributes() - 1

MyClassifier myClassifier = new MyClassifier()
Instances newDataset = myClassifier.buildClassifier(dataset)	
println newDataset
