package main;

import java.io.IOException;
import java.util.List;

import mlsp.cs.cmu.edu.dnn.factory.DNNFactory;
import mlsp.cs.cmu.edu.dnn.factory.ReadSerializedFileDNNFactory;
import mlsp.cs.cmu.edu.dnn.structure.NeuralNetwork;
import mlsp.cs.cmu.edu.dnn.training.DNNTrainingModule;
import mlsp.cs.cmu.edu.dnn.training.DataInstance;
import mlsp.cs.cmu.edu.dnn.training.DataReader;
import mlsp.cs.cmu.edu.dnn.training.ReadCSVTrainingData;
import mlsp.cs.cmu.edu.dnn.util.BinaryThresholdOutput;
import mlsp.cs.cmu.edu.dnn.util.OutputAdapter;
import mlsp.cs.cmu.edu.dnn.util.PruningTool;

public class TestPruning {

	public static void main(String[] args) throws IOException {
		RShapeDriver(0);
	}

	public static void RShapeDriver(int... structure) throws IOException {
		System.out.println("---------------------------------------------------");
		System.out.println("                    RShape                         ");
		System.out.println("---------------------------------------------------");
		DataReader reader = new ReadCSVTrainingData();
		List<DataInstance> training = reader.getDataFromFile(PruningTool.data + "RShape-train.csv", 2, 1);
		List<DataInstance> testing = reader.getDataFromFile(PruningTool.data + "RShape-test.csv", 2, 1);
		
		double remove = 0.0;
		while (remove <= 1) {
			System.out.println("\n\nWith remove: " + remove);
			System.out.println("Deserializing stored network " + PruningTool.modDnnFile);
			DNNFactory factory = new ReadSerializedFileDNNFactory(PruningTool.modDnnFile);
			NeuralNetwork net = factory.getInitializedNeuralNetwork();
			DNNTrainingModule trainingModule = new DNNTrainingModule(net, testing);
			trainingModule.setOutputOn(false);
			trainingModule.setOutputAdapter(PruningTool.adapter);
			trainingModule.doTestTrainedNetwork();
			net = PruningTool.doPruning(net, training, testing, remove);
			remove += 2;
		}
	}

}
