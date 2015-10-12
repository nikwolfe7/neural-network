package main;

import java.util.List;

import mlsp.cs.cmu.edu.dnn.factory.DNNFactory;
import mlsp.cs.cmu.edu.dnn.factory.ReadSerializedFileDNNFactory;
import mlsp.cs.cmu.edu.dnn.factory.SigmoidNetworkFFDNNFactory;
import mlsp.cs.cmu.edu.dnn.factory.TanhOutputFFDNNFactory;
import mlsp.cs.cmu.edu.dnn.structure.NeuralNetwork;
import mlsp.cs.cmu.edu.dnn.training.DNNTrainingModule;
import mlsp.cs.cmu.edu.dnn.training.DataInstance;
import mlsp.cs.cmu.edu.dnn.training.DataReader;
import mlsp.cs.cmu.edu.dnn.training.ReadCSVTrainingData;
import mlsp.cs.cmu.edu.dnn.util.BinaryThresholdOutput;
import mlsp.cs.cmu.edu.dnn.util.DNNUtils;
import mlsp.cs.cmu.edu.dnn.util.OutputAdapter;

public class ShapesDriver {
  
  static OutputAdapter adapter = new BinaryThresholdOutput();
  static boolean printOut = false;
  static String sep = System.getProperty("file.separator");
  static String data = "." + sep + "data" + sep;
  
  static int[][] configs = new int[][] {
		{2,2},
		{4,4},
		{8,8},
		{16,16},
		{32,32},
		{64,64},
	};
	static int[][] configs2 = new int[][] {
		{1},
		{2},
		{3},
		{4},
		{5},
		{6},
		{7},
	};

	public static void main(String[] args) {
		CircleDriver(4);
//		for(int[] config : configs) {
//			CircleDriver(config);
//			DiamondDriver(config);
//			RShapeDriver(config);
//			DRShapeDriver(config);
//		}
//		for(int[] config : configs2) {
//			CircleDriver(config);
//			DiamondDriver(config);
//			RShapeDriver(config);
//			DRShapeDriver(config);
//		}
	}
	
	public static void CircleDriver(int... structure) {
		System.out.println("---------------------------------------------------");
		System.out.println("                    Circle                         ");
		System.out.println("---------------------------------------------------");
		DataReader reader = new ReadCSVTrainingData();
//	    List<DataInstance> training = reader.getDataFromFile("circle-train.csv", 2, 1);
	    List<DataInstance> testing = reader.getDataFromFile(data + "circle-test.csv", 2, 1);
//	    DNNFactory factory = new SigmoidNetworkFFDNNFactory(training.get(0), structure);
//	    
//	    NeuralNetwork net = factory.getInitializedNeuralNetwork();
//	    DNNTrainingModule trainingModule = new DNNTrainingModule(net, training, testing);
//	    trainingModule.setOutputOn(printOut);
//	    trainingModule.setOutputAdapter(adapter);
//	    trainingModule.setPrintResults(true, "circle-test-results-"+DNNUtils.joinNumbers(structure, "-")+".csv");
//	    trainingModule.doTrainNetworkUntilConvergence();
//	    trainingModule.doTestTrainedNetwork();
//	    trainingModule.saveNetworkToFile("network.dnn");
	    
	    /* Test */
	    System.out.println("De-serializing the network..");
	    ReadSerializedFileDNNFactory factory = new ReadSerializedFileDNNFactory(data + "circle-network.dnn");
	    NeuralNetwork net = factory.getInitializedNeuralNetwork();
	    DNNTrainingModule trainingModule = new DNNTrainingModule(net, testing);
	    trainingModule.setOutputOn(printOut);
	    trainingModule.setOutputAdapter(adapter);
	    trainingModule.doTestTrainedNetwork();
	    
	}
	
	public static void DiamondDriver(int... structure) {
		System.out.println("---------------------------------------------------");
		System.out.println("                    Diamond                        ");
		System.out.println("---------------------------------------------------");
		DataReader reader = new ReadCSVTrainingData();
	    List<DataInstance> training = reader.getDataFromFile(data + "diamond-train.csv", 2, 1);
	    List<DataInstance> testing = reader.getDataFromFile(data + "diamond-test.csv", 2, 1);
	    DNNFactory factory = new SigmoidNetworkFFDNNFactory(training.get(0), structure);
	    
	    NeuralNetwork net = factory.getInitializedNeuralNetwork();
	    DNNTrainingModule trainingModule = new DNNTrainingModule(net, training, testing);
	    trainingModule.setOutputOn(printOut);
	    trainingModule.setOutputAdapter(adapter);
	    trainingModule.setPrintResults(true, data + "diamond-test-results-"+DNNUtils.joinNumbers(structure, "-")+".csv");
	    trainingModule.doTrainNetworkUntilConvergence();
	    trainingModule.doTestTrainedNetwork();
	}
	
	public static void RShapeDriver(int... structure) {
		System.out.println("---------------------------------------------------");
		System.out.println("                    RShape                         ");
		System.out.println("---------------------------------------------------");
		DataReader reader = new ReadCSVTrainingData();
	    List<DataInstance> training = reader.getDataFromFile(data + "RShape-train.csv", 2, 1);
	    List<DataInstance> testing = reader.getDataFromFile(data + "RShape-test.csv", 2, 1);
	    DNNFactory factory = new SigmoidNetworkFFDNNFactory(training.get(0), structure);
	    
	    NeuralNetwork net = factory.getInitializedNeuralNetwork();
	    DNNTrainingModule trainingModule = new DNNTrainingModule(net, training, testing);
	    trainingModule.setOutputOn(printOut);
	    trainingModule.setOutputAdapter(adapter);
	    trainingModule.setPrintResults(true, data + "RShape-test-results-"+DNNUtils.joinNumbers(structure, "-")+".csv");
	    trainingModule.doTrainNetworkUntilConvergence();
	    trainingModule.doTestTrainedNetwork();
	}
	
	public static void DRShapeDriver(int... structure) {
		System.out.println("---------------------------------------------------");
		System.out.println("                    DRShape                        ");
		System.out.println("---------------------------------------------------");
		DataReader reader = new ReadCSVTrainingData();
	    List<DataInstance> training = reader.getDataFromFile(data + "DRShape-train.csv", 2, 1);
	    List<DataInstance> testing = reader.getDataFromFile(data + "DRShape-test.csv", 2, 1);
	    DNNFactory factory = new TanhOutputFFDNNFactory(training.get(0), structure);
	    
	    NeuralNetwork net = factory.getInitializedNeuralNetwork();
	    DNNTrainingModule trainingModule = new DNNTrainingModule(net, training, testing);
	    trainingModule.setOutputOn(printOut);
	    trainingModule.setOutputAdapter(adapter);
	    trainingModule.setPrintResults(true, data + "DRShape-test-results-"+DNNUtils.joinNumbers(structure, "-")+".csv");
	    trainingModule.doTrainNetworkUntilConvergence();
	    trainingModule.doTestTrainedNetwork();
	}


}
