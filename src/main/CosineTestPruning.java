package main;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import mlsp.cs.cmu.edu.dnn.factory.DNNFactory;
import mlsp.cs.cmu.edu.dnn.factory.ReadSerializedFileDNNFactory;
import mlsp.cs.cmu.edu.dnn.structure.NeuralNetwork;
import mlsp.cs.cmu.edu.dnn.training.CosineGenerator;
import mlsp.cs.cmu.edu.dnn.training.DNNTrainingModule;
import mlsp.cs.cmu.edu.dnn.training.DataInstance;
import mlsp.cs.cmu.edu.dnn.training.DataInstanceGenerator;
import mlsp.cs.cmu.edu.dnn.training.DataReader;
import mlsp.cs.cmu.edu.dnn.training.ReadCSVTrainingData;
import mlsp.cs.cmu.edu.dnn.training.XORGenerator;
import mlsp.cs.cmu.edu.dnn.util.BinaryThresholdOutput;
import mlsp.cs.cmu.edu.dnn.util.OutputAdapter;
import mlsp.cs.cmu.edu.dnn.util.PruningTool;

public class CosineTestPruning {

  public static String sep = System.getProperty("file.separator");
	public static String dnnFile = "cos.big.network.dnn";
	static DataInstanceGenerator dataGen = new CosineGenerator();

	public static void main(String[] args) throws IOException {
		TestPruning(0);
	}

	public static void TestPruning(int... structure) throws IOException {
		System.out.println("---------------------------------------------------");
		System.out.println("                    COS                            ");
		System.out.println("---------------------------------------------------");
		DataReader reader = new ReadCSVTrainingData();
		List<DataInstance> training = getData(10000);
		List<DataInstance> testing = getData(10000);

		System.out.println("Deserializing stored network " + dnnFile);
		DNNFactory factory = new ReadSerializedFileDNNFactory("models" + sep + dnnFile);
		NeuralNetwork net = factory.getInitializedNeuralNetwork();
		DNNTrainingModule trainingModule = new DNNTrainingModule(net, testing);
		trainingModule.setOutputOn(false);
		trainingModule.setOutputAdapter(new BinaryThresholdOutput());
		trainingModule.doTestTrainedNetwork();
		net = PruningTool.runPruningExperiment(dnnFile, true, net, training, testing, 1.0);
	}
	
	private static List<DataInstance> getData(int numInstances) {
    List<DataInstance> data = new ArrayList<>();
    for (int i = 0; i < numInstances; i++)
      data.add(dataGen.getNewDataInstance());
    return data;
  }

}
