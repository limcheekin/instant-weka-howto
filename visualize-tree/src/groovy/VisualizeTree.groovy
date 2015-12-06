import weka.core.Instances
import javax.swing.JFrame;

import weka.classifiers.trees.J48;
import weka.gui.treevisualizer.PlaceNode2;
import weka.gui.treevisualizer.TreeVisualizer;

Instances data = new Instances(new BufferedReader(new FileReader("../dataset/titanic.arff")));
data.setClassIndex(data.numAttributes() - 1);
J48 classifier = new J48()
classifier.buildClassifier(data)

TreeVisualizer tv = new TreeVisualizer(null, classifier.graph(), new PlaceNode2())
		
JFrame frame = new javax.swing.JFrame("Tree Visualizer")
frame.setSize(1300, 800)
frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE)
frame.getContentPane().add(tv)
frame.setVisible(true)
tv.fitToScreen()