import weka.core.Instances
import weka.core.converters.ConverterUtils.DataSource
import weka.attributeSelection.CfsSubsetEval
import weka.attributeSelection.GreedyStepwise
import weka.attributeSelection.InfoGainAttributeEval
import weka.attributeSelection.PrincipalComponents
import weka.attributeSelection.Ranker
import weka.attributeSelection.ReliefFAttributeEval
import weka.classifiers.Evaluation
import weka.classifiers.meta.AttributeSelectedClassifier
import weka.classifiers.trees.J48
import weka.filters.Filter
import weka.filters.supervised.attribute.AttributeSelection

selectExample()
infoGainExample()
principalComponentAnalysisExample()
classifierSpecificExample()

/*
The greedy search algorithm runs with the correlation-based feature selection evaluator 
to discover a subset of attributes with the highest predictive power.
*/
void selectExample() {
	println "selectExample()"

	DataSource source = new DataSource("../dataset/titanic.arff")
	Instances data = source.dataSet

	AttributeSelection filter = new AttributeSelection()
	CfsSubsetEval eval = new CfsSubsetEval()
	GreedyStepwise search = new GreedyStepwise()

	search.searchBackwards = true
	filter.evaluator = eval
	filter.search = search
	filter.inputFormat = data
	Instances newData = Filter.useFilter(data, filter)

	println "Number of attributes: ${newData.numAttributes()}"
}

void infoGainExample() {
	println "infoGainExample()"

	DataSource source = new DataSource("../dataset/titanic.arff")
	Instances data = source.dataSet

	weka.attributeSelection.AttributeSelection filter = new weka.attributeSelection.AttributeSelection()
	InfoGainAttributeEval eval = new InfoGainAttributeEval()
	Ranker search = new Ranker()

	filter.evaluator = eval
	filter.search = search
	filter.SelectAttributes(data)
	int[] indices = filter.selectedAttributes();
	
	println filter.toResultsString()
    println indices
}

void principalComponentAnalysisExample() {
	println "principalComponentAnalysisExample()"

	DataSource source = new DataSource("../dataset/titanic.arff")
	Instances data = source.dataSet

	AttributeSelection filter = new AttributeSelection()
	PrincipalComponents eval = new PrincipalComponents()
	Ranker search = new Ranker()

	filter.evaluator = eval
	filter.search = search
	filter.inputFormat = data
	Instances newData = Filter.useFilter(data, filter)

	println newData
}

void classifierSpecificExample() {
	println "classifierSpecificExample()"

	DataSource source = new DataSource("../dataset/titanic.arff")
	Instances data = source.dataSet
	data.classIndex = data.numAttributes() - 1
			
	AttributeSelectedClassifier classifier = new AttributeSelectedClassifier()
	ReliefFAttributeEval eval = new ReliefFAttributeEval()
	Ranker search = new Ranker()
	J48 baseClassifier = new J48()
	classifier.classifier = baseClassifier 
	classifier.evaluator = eval
	classifier.search = search
	
	Evaluation evaluation = new Evaluation(data);
	evaluation.crossValidateModel(classifier, data, 10, new Random(1));
	println evaluation.toSummaryString()
}