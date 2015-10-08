package main;

import java.util.List;

import mlsp.cs.cmu.edu.dnn.factory.DNNFactory;
import mlsp.cs.cmu.edu.dnn.factory.SigmoidNetworkFFDNNFactory;
import mlsp.cs.cmu.edu.dnn.structure.NeuralNetwork;
import mlsp.cs.cmu.edu.dnn.training.DNNTrainingModule;
import mlsp.cs.cmu.edu.dnn.training.DataInstance;
import mlsp.cs.cmu.edu.dnn.training.DataReader;
import mlsp.cs.cmu.edu.dnn.training.ReadCSVTrainingData;

public class ShapesDriver {

	public static void main(String[] args) {
//		CircleDriver();
//		DiamondDriver();
		RShapeDriver();
//		DRShapeDriver();
	}
	
	public static void CircleDriver() {
		DataReader reader = new ReadCSVTrainingData();
	    List<DataInstance> training = reader.getDataFromFile("circle-train.csv", 2, 1);
	    List<DataInstance> testing = reader.getDataFromFile("circle-test.csv", 2, 1);
	    DNNFactory factory = new SigmoidNetworkFFDNNFactory(training.get(0), 16, 16);
	    
	    NeuralNetwork net = factory.getInitializedNeuralNetwork();
	    DNNTrainingModule trainingModule = new DNNTrainingModule(net, training, testing);
	    trainingModule.setOutputOn(true);
	    trainingModule.doTrainNetworkUntilConvergence();
	    trainingModule.doTestTrainedNetwork();
	}
	
	public static void DiamondDriver() {
		DataReader reader = new ReadCSVTrainingData();
	    List<DataInstance> training = reader.getDataFromFile("diamond-train.csv", 2, 1);
	    List<DataInstance> testing = reader.getDataFromFile("diamond-test.csv", 2, 1);
	    DNNFactory factory = new SigmoidNetworkFFDNNFactory(training.get(0), 16, 16);
	    
	    NeuralNetwork net = factory.getInitializedNeuralNetwork();
	    DNNTrainingModule trainingModule = new DNNTrainingModule(net, training, testing);
	    trainingModule.setOutputOn(true);
	    trainingModule.doTrainNetworkUntilConvergence();
	    trainingModule.doTestTrainedNetwork();
	}
	
	public static void RShapeDriver() {
		DataReader reader = new ReadCSVTrainingData();
	    List<DataInstance> training = reader.getDataFromFile("RShape-train.csv", 2, 1);
	    List<DataInstance> testing = reader.getDataFromFile("RShape-test.csv", 2, 1);
	    DNNFactory factory = new SigmoidNetworkFFDNNFactory(training.get(0), 32, 16);
	    
	    NeuralNetwork net = factory.getInitializedNeuralNetwork();
	    DNNTrainingModule trainingModule = new DNNTrainingModule(net, training, testing);
	    trainingModule.setOutputOn(true);
	    trainingModule.doTrainNetworkUntilConvergence();
	    trainingModule.doTestTrainedNetwork();
	}
	
	public static void DRShapeDriver() {
		DataReader reader = new ReadCSVTrainingData();
	    List<DataInstance> training = reader.getDataFromFile("DRShape-train.csv", 2, 1);
	    List<DataInstance> testing = reader.getDataFromFile("DRShape-test.csv", 2, 1);
	    DNNFactory factory = new SigmoidNetworkFFDNNFactory(training.get(0), 16, 16);
	    
	    NeuralNetwork net = factory.getInitializedNeuralNetwork();
	    DNNTrainingModule trainingModule = new DNNTrainingModule(net, training, testing);
	    trainingModule.setOutputOn(true);
	    trainingModule.doTrainNetworkUntilConvergence();
	    trainingModule.doTestTrainedNetwork();
	}


}
