package mnist.digit2;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import mlsp.cs.cmu.edu.dnn.factory.DNNFactory;
import mlsp.cs.cmu.edu.dnn.factory.ReadSerializedFileDNNFactory;
import mlsp.cs.cmu.edu.dnn.structure.NeuralNetwork;
import mlsp.cs.cmu.edu.dnn.training.CosineGenerator;
import mlsp.cs.cmu.edu.dnn.training.DNNTrainingModule;
import mlsp.cs.cmu.edu.dnn.training.DataInstance;
import mlsp.cs.cmu.edu.dnn.training.DataInstanceFactory;
import mlsp.cs.cmu.edu.dnn.training.DataInstanceGenerator;
import mlsp.cs.cmu.edu.dnn.training.DataReader;
import mlsp.cs.cmu.edu.dnn.training.ReadCSVTrainingData;
import mlsp.cs.cmu.edu.dnn.training.XORGenerator;
import mlsp.cs.cmu.edu.dnn.util.BinaryThresholdOutput;
import mlsp.cs.cmu.edu.dnn.util.DNNUtils;
import mlsp.cs.cmu.edu.dnn.util.MaxBinaryThresholdOutput;
import mlsp.cs.cmu.edu.dnn.util.OutputAdapter;
import mlsp.cs.cmu.edu.dnn.util.PruningTool;

public class MNISTDigitTestPruning {

	static int numToLearn = 2;
	static String dnnFile = "mnist-test-single-digit-" + numToLearn + "-in-784-out-2-struct-100.dnn";

	public static void main(String[] args) throws IOException {
		TestPruning();
	}

	public static void TestPruning() throws IOException {
		System.out.println("---------------------------------------------------");
		System.out.println("               MNIST Digit Pruning                 ");
		System.out.println("---------------------------------------------------");
		
		OutputAdapter adapter = new MaxBinaryThresholdOutput();
		DataInstanceFactory dataInstanceFactory = new MNISTSingleDigitInstanceFactory(numToLearn);
		List<DataInstance> training = dataInstanceFactory.getTrainingInstances();
		List<DataInstance> testing = dataInstanceFactory.getTestingInstances();

		System.out.println("Deserializing stored network " + dnnFile);
		DNNFactory factory = new ReadSerializedFileDNNFactory("models" + DNNUtils.sep + dnnFile);
		NeuralNetwork net = factory.getInitializedNeuralNetwork();
		DNNTrainingModule trainingModule = new DNNTrainingModule(net, testing);
		trainingModule.setOutputOn(false);
		trainingModule.setOutputAdapter(adapter);
		trainingModule.doTestTrainedNetwork();
		net = PruningTool.runPruningExperiment(dnnFile, true, net, training, testing, 1.0);
	}

}
