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

	static OutputAdapter adapter = new BinaryThresholdOutput();
	static boolean printOut = true;
	static boolean batchUpdate = false;
	static boolean removeElements = false;
	static String sep = System.getProperty("file.separator");
	static String data = "." + sep + "data" + sep;

	public static void main(String[] args) throws IOException {
		RShapeDriver(0);
	}

	public static void RShapeDriver(int... structure) throws IOException {
		System.out.println("---------------------------------------------------");
		System.out.println("                    RShape                         ");
		System.out.println("---------------------------------------------------");
		DataReader reader = new ReadCSVTrainingData();
		List<DataInstance> training = reader.getDataFromFile(data + "RShape-train.csv", 2, 1);
		List<DataInstance> testing = reader.getDataFromFile(data + "RShape-test.csv", 2, 1);
		String dnnFile = data + "rshape.network.dnn";
		
		double remove = 0.0;
		while (remove <= 1) {
			System.out.println("\n\nWith remove: " + remove);
			System.out.println("Deserializing stored network " + dnnFile);
			DNNFactory factory = new ReadSerializedFileDNNFactory(dnnFile);
			NeuralNetwork net = factory.getInitializedNeuralNetwork();
			DNNTrainingModule trainingModule = new DNNTrainingModule(net, testing);
			trainingModule.setOutputOn(false);
			trainingModule.setOutputAdapter(adapter);
			trainingModule.doTestTrainedNetwork();
			net = PruningTool.doPruning(net, training, remove, removeElements);
			remove += 2;
		}
	}

}
