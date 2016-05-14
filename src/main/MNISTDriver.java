package main;

import java.util.List;

import mlsp.cs.cmu.edu.dnn.factory.CrossEntropyFFDNNFactory;
import mlsp.cs.cmu.edu.dnn.factory.DNNFactory;
import mlsp.cs.cmu.edu.dnn.structure.NeuralNetwork;
import mlsp.cs.cmu.edu.dnn.training.DNNTrainingModule;
import mlsp.cs.cmu.edu.dnn.training.DataInstance;
import mlsp.cs.cmu.edu.dnn.training.DataInstanceFactory;
import mlsp.cs.cmu.edu.dnn.training.MNISTDataInstanceFactory;
import mlsp.cs.cmu.edu.dnn.util.DNNUtils;
import mlsp.cs.cmu.edu.dnn.util.MaxBinaryThresholdOutput;
import mlsp.cs.cmu.edu.dnn.util.OutputAdapter;

public class MNISTDriver {

	static OutputAdapter adapter = new MaxBinaryThresholdOutput();
	static boolean printout = true;
	static boolean batchUpdate = false;
	static String data = "." + DNNUtils.sep + "data" + DNNUtils.sep;
	static String networkFile = "mnist.network.dnn";

	public static void main(String[] args) {
		RunMNISTDriver(28,56,16);
	}

	public static void RunMNISTDriver(int... structure) {
		System.out.println("---------------------------------------------------");
		System.out.println("              MNIST Digit Recognition              ");
		System.out.println("---------------------------------------------------");
		
		/* Data stuff */
		DataInstanceFactory dataFactory = new MNISTDataInstanceFactory();
		List<DataInstance> training = dataFactory.getTrainingInstances();
		List<DataInstance> testing = dataFactory.getTestingInstances();

		/* Build network */
		System.out.print("Building network... ");
		DNNFactory factory = new CrossEntropyFFDNNFactory(training.get(0), structure);
		NeuralNetwork net = factory.getInitializedNeuralNetwork();
		DNNTrainingModule trainingModule = new DNNTrainingModule(net, training, testing);
		System.out.println("Done!");

		/* Set up stuff... */
		System.out.print("Setting up training... ");
		trainingModule.setOutputOn(printout);
		trainingModule.setOutputAdapter(adapter);
		trainingModule.setBatchUpdate(batchUpdate, 600);
		trainingModule.setConvergenceCriteria(1.0e-8, -1, 0, 10);
		trainingModule.setPrintResults(true, "mnist-test-results-" + DNNUtils.joinNumbers(structure, "-") + ".csv");
		System.out.println("Done!");
		
		trainingModule.doTrainNetworkUntilConvergence();
		
		/* Test and shit */
		System.out.println("Test:\n");
		trainingModule.doTestTrainedNetwork();
		trainingModule.saveNetworkToFile("models" + DNNUtils.sep + networkFile);

		/* Finished! */
		System.out.println("Done!");
	}

}
