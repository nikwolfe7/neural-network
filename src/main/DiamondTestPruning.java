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

public class DiamondTestPruning {
	
  public static String sep = System.getProperty("file.separator");
	public static String dnnFile = "diamond.network.dnn";

	public static void main(String[] args) throws IOException {
		TestPruning(0);
	}

	public static void TestPruning(int... structure) throws IOException {
		System.out.println("---------------------------------------------------");
		System.out.println("                    Diamond                        ");
		System.out.println("---------------------------------------------------");
		DataReader reader = new ReadCSVTrainingData();
		List<DataInstance> training = reader.getDataFromFile(PruningTool.data + "diamond-train.csv", 2, 1);
		List<DataInstance> testing = reader.getDataFromFile(PruningTool.data + "diamond-test.csv", 2, 1);
		
		System.out.println("Deserializing stored network " + dnnFile);
		DNNFactory factory = new ReadSerializedFileDNNFactory("models" + sep + dnnFile);
		NeuralNetwork net = factory.getInitializedNeuralNetwork();
		DNNTrainingModule trainingModule = new DNNTrainingModule(net, testing);
		trainingModule.setOutputOn(false);
		trainingModule.setOutputAdapter(new BinaryThresholdOutput());
		trainingModule.doTestTrainedNetwork();
		PruningTool.setOutputAdapter(new BinaryThresholdOutput());
		net = PruningTool.runPruningExperiment(dnnFile, true, net, training, testing, 1.0);
	}

}
