import weka.core.Instances
import weka.core.converters.ConverterUtils.DataSource
import weka.classifiers.Evaluation
import weka.classifiers.trees.J48
import javax.swing.*
import weka.core.*
import weka.classifiers.evaluation.*
import weka.gui.visualize.*

evaluate()
rocCurve()

void evaluate() { 

	DataSource source = new DataSource("../dataset/titanic.arff")
	Instances data = source.dataSet
	data.classIndex = data.numAttributes() - 1

	J48 classifier = new J48()
	Evaluation eval = new Evaluation(data)
	eval.crossValidateModel(classifier, data, 10, new Random(1))

	println eval.toSummaryString("Results", false)

	println "-------- Stats ----------"
	println eval.correct()
	println eval.pctCorrect()
	println eval.kappa()
	println eval.correct()
	println "-------------------------"
	println eval.toMatrixString()
}

void rocCurve() { 

	DataSource source = new DataSource("../dataset/titanic.arff")
	Instances data = source.dataSet
	data.classIndex = data.numAttributes() - 1

	J48 classifier = new J48()
	Evaluation eval = new Evaluation(data)
	eval.crossValidateModel(classifier, data, 10, new Random(1))

	// generate curve
	ThresholdCurve tc = new ThresholdCurve();
	int classIndex = 0;
	Instances result = tc.getCurve(eval.predictions(), classIndex);

	// plot curve
	ThresholdVisualizePanel vmc = new ThresholdVisualizePanel();
	vmc.ROCString = "(Area under ROC = ${Utils.doubleToString(tc.getROCArea(result), 4)})"
	vmc.name = result.relationName() 
	PlotData2D tempd = new PlotData2D(result)
	tempd.plotName = result.relationName() 
	tempd.addInstanceNumberAttribute()
	// specify which points are connected
	boolean[] cp = new boolean[result.numInstances()]
	for (int n = 1; n < cp.length; n++)
		cp[n] = true;
	tempd.connectPoints = cp
	// add plot
	vmc.addPlot(tempd)

	// display curve		
	JFrame frame = new javax.swing.JFrame("ROC Curve")
	frame.setSize(800, 500)
	frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE)
	frame.getContentPane().add(vmc)
	frame.setVisible(true)
}
