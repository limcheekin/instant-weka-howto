import weka.core.Instances
import weka.core.converters.ConverterUtils.DataSource
import weka.filters.Filter
import weka.filters.unsupervised.attribute.Remove

DataSource source = new DataSource("../dataset/titanic.arff")
Instances data = source.dataSet

String[] options = new String[2];
options[0] = "-R" // "range"
options[1] = "2"  // first attribute
Remove remove = new Remove()                          // new instance of filter
remove.options = options                              // set options
remove.inputFormat = data                             // inform filter about dataset **AFTER** setting options
Instances newData = Filter.useFilter(data, remove);   // apply filter

println data
println newData