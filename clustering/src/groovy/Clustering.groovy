import weka.core.Instance
import weka.core.Instances
import weka.core.converters.ArffLoader
import weka.clusterers.ClusterEvaluation
import weka.clusterers.Cobweb
import weka.clusterers.EM

//load data
Instances data = new Instances(new BufferedReader(new FileReader("../dataset/bank-data.arff")))

// new instance of clusterer
EM model = new EM()
// build the clusterer
model.buildClusterer(data)
println model

clusterClassify()
incrementalCluster()
evaluate()


void clusterClassify() {
	
	//load data
	Instances data = new Instances(new BufferedReader(new FileReader("../dataset/bank-data.arff")));
	Instance inst = data.instance(0)
	data.delete(0)
	
	// new instance of clusterer
	EM model = new EM()
	// build the clusterer
	model.buildClusterer(data)
	
	int cls = model.clusterInstance(inst)
	println "Cluster: $cls"
	
	double[] dist = model.distributionForInstance(inst)
	for(int i = 0; i < dist.length; i++) 
		println "Cluster ${i}.\t${dist[i]}"

}

void incrementalCluster() {
	// load data
	ArffLoader loader = new ArffLoader()
	loader.file = new File("../dataset/bank-data.arff")
	Instances data = loader.structure
	 
	 // train Cobweb
	Cobweb model = new Cobweb()
	model.buildClusterer(data)
	Instance current
	while ((current = loader.getNextInstance(data)))
	   model.updateClusterer(current)
	model.updateFinished()
	println model
}

void evaluate() {

	Instances data = new Instances(new BufferedReader(new FileReader("../dataset/bank-data.arff")));

	EM model = new EM()

	//double logLikelyhood = ClusterEvaluation.crossValidateModel(model, data, 10, new Random(1));
	//System.out.println(logLikelyhood);

	ClusterEvaluation eval = new ClusterEvaluation()
	model.buildClusterer(data)                                 // build clusterer
	eval.setClusterer(model)                                   // the cluster to evaluate
	eval.evaluateClusterer(data)                               // data to evaluate the clusterer on
	println "# of clusters: ${eval.getNumClusters()}"  // output # of clusters

}