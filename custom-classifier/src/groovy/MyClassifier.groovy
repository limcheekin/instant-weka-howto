import weka.classifiers.Classifier
import weka.core.Capabilities
import weka.core.Capabilities.Capability
import weka.core.Instance
import weka.core.Instances

class MyClassifier extends Classifier {

	private Instances trainData

	public Capabilities getCapabilities() {

		Capabilities result = super.capabilities
		result.disableAll()

		// attributes
		result.enable(Capability.NUMERIC_ATTRIBUTES)
		result.enable(Capability.NOMINAL_ATTRIBUTES)

		// class
		result.enable(Capability.NOMINAL_CLASS)

		// instances
		result.setMinimumNumberInstances(2)

		return result
	}

	@Override
	public void buildClassifier(Instances data) throws Exception {

		// can classifier handle the data?
		getCapabilities().testWithFail(data)

		// remove instances with missing class
		data = new Instances(data)
		data.deleteWithMissingClass()

		// save train data
		trainData = new Instances(data, 0, data.numInstances())

	}

	public double classifyInstance(Instance instance) {
		
		
        double minDistance =  Double.MAX_VALUE
        double secondMinDistance =  Double.MAX_VALUE
        double distance
        double classVal = 0, minClassVal = 0
        
		Enumeration enu = trainData.enumerateInstances()
		while (enu.hasMoreElements()) {

			Instance trainInstance = (Instance) enu.nextElement()
			if (!trainInstance.classIsMissing()) {
				
				distance = distance(instance, trainInstance)

				if (distance < minDistance) {

					secondMinDistance = minDistance
					minDistance = distance
					
					classVal = minClassVal
					minClassVal = trainInstance.classValue()

				} else if (distance < secondMinDistance) {
					secondMinDistance = distance
					classVal = trainInstance.classValue()
				} 
			}
		}
        return classVal

	}

	private double distance(Instance first, Instance second) {
		double diff = 0, distance = 0
		for (int i = 0; i < trainData.numAttributes(); i++) {
			if (i == trainData.classIndex()) {
				continue
			}
			if (trainData.attribute(i).isNominal()) { // If attribute is nominal
				if(!first.isMissing(i) && !second.isMissing(i) && first.value(i) == second.value(i)) {
					diff = 0
				} else diff = 1
			}else { // If attribute is numeric
				if (first.isMissing(i) || second.isMissing(i)) {
					diff = 1
				}else {
					diff = Math.abs(first.value(i) - second.value(i))
				}
			}
			distance += diff
		}
		return distance
	}

}
