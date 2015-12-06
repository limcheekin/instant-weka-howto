import weka.core.Instances
import weka.core.converters.ConverterUtils.DataSource

DataSource source = new DataSource("../dataset/titanic.arff")
Instances data = source.dataSet
println data
println "${data.numInstances()} instances loaded."