package main;

import java.io.IOException;
import java.util.List;

import mlsp.cs.cmu.edu.dnn.elements.Edge;
import mlsp.cs.cmu.edu.dnn.elements.MomentumEdge;
import mlsp.cs.cmu.edu.dnn.factory.CrossEntropyFFDNNFactory;
import mlsp.cs.cmu.edu.dnn.factory.DNNFactory;
import mlsp.cs.cmu.edu.dnn.factory.ReadSerializedFileDNNFactory;
import mlsp.cs.cmu.edu.dnn.factory.SigmoidNetworkFFDNNFactory;
import mlsp.cs.cmu.edu.dnn.structure.NeuralNetwork;
import mlsp.cs.cmu.edu.dnn.training.DNNTrainingModule;
import mlsp.cs.cmu.edu.dnn.training.DataInstance;
import mlsp.cs.cmu.edu.dnn.training.DataReader;
import mlsp.cs.cmu.edu.dnn.training.ReadBinaryCSVTrainingDataForCrossEntropy;
import mlsp.cs.cmu.edu.dnn.training.ReadCSVTrainingData;
import mlsp.cs.cmu.edu.dnn.util.BinaryThresholdOutput;
import mlsp.cs.cmu.edu.dnn.util.DNNUtils;
import mlsp.cs.cmu.edu.dnn.util.PruningTool;
import mlsp.cs.cmu.edu.dnn.util.OutputAdapter;

public class RShapeDriver {
  
  static OutputAdapter adapter = new BinaryThresholdOutput();
  static boolean printOut = true;
  static boolean batchUpdate = false;
  static boolean removeElements = false;
  static String sep = System.getProperty("file.separator");
  static String data = "." + sep + "data" + sep;
  
	public static void main(String[] args) throws IOException, CloneNotSupportedException {
		RShape(50,50);
	}
	
	public static void RShape(int... structure) throws IOException {
		System.out.println("---------------------------------------------------");
		System.out.println("                    RShape                         ");
		System.out.println("---------------------------------------------------");
		DataReader reader = new ReadCSVTrainingData();
		List<DataInstance> training = reader.getDataFromFile(data + "RShape-train.csv", 2, 1);
		List<DataInstance> testing = reader.getDataFromFile(data + "RShape-test.csv", 2, 1);
		DNNFactory factory = new SigmoidNetworkFFDNNFactory(training.get(0), structure);

		NeuralNetwork net = factory.getInitializedNeuralNetwork();
		DNNTrainingModule trainingModule = new DNNTrainingModule(net, training, testing);
//		trainingModule.setOutputOn(true);
//		trainingModule.setOutputAdapter(adapter);
//		trainingModule.setBatchUpdate(batchUpdate,10);
//		trainingModule.setConvergenceCriteria(-1, 0.004, true, 0);
//		trainingModule.setPrintResults(true, data + "RShape-test-results-" + DNNUtils.joinNumbers(structure, "-") + ".csv");
//		trainingModule.doTrainNetworkUntilConvergence();
//
//		System.out.println("Test:\n");
		
//		trainingModule.setOutputOn(false);
//		trainingModule.doTestTrainedNetwork();
//		trainingModule.saveNetworkToFile(data + "network.dnn");

		double remove = 0.0;
		while (remove <= 1) {
			System.out.println("\n\nWith remove: " + remove);
			factory = new ReadSerializedFileDNNFactory(data + "rshape.network.dnn");
			net = factory.getInitializedNeuralNetwork();
			trainingModule = new DNNTrainingModule(net, testing);
			trainingModule.setOutputOn(false);
			trainingModule.setOutputAdapter(adapter);
			net = PruningTool.doPruning(net, training, testing, remove);
			trainingModule.doTestTrainedNetwork();
			remove += 2;
		}

	}

}
