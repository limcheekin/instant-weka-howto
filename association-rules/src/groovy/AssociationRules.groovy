import weka.core.Instances
import weka.associations.Apriori

Instances data = new Instances(new BufferedReader(new FileReader("../dataset/bank-data.arff")))

//build model
Apriori model = new Apriori()
model.buildAssociations(data) 
println model