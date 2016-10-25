package main;

import java.util.ArrayList;
import java.util.List;

import mlsp.cs.cmu.edu.dnn.factory.DNNFactory;
import mlsp.cs.cmu.edu.dnn.factory.SigmoidNetworkFFDNNFactory;
import mlsp.cs.cmu.edu.dnn.structure.NeuralNetwork;
import mlsp.cs.cmu.edu.dnn.training.CosineGenerator;
import mlsp.cs.cmu.edu.dnn.training.DNNTrainingModule;
import mlsp.cs.cmu.edu.dnn.training.DataInstance;
import mlsp.cs.cmu.edu.dnn.training.DataInstanceGenerator;

public class CosineFactoryDriver {

	/* Useful to have to port across OS... */
	static String sep = System.getProperty("file.separator");
	
	/* Step 1: Get data generator for Cosines */ 
	static DataInstanceGenerator dataGen = new CosineGenerator();
	
	/* Step 2: Get a DNN Builder Factory, specify input/output dimensions and structure */
	static DNNFactory factory = new SigmoidNetworkFFDNNFactory(dataGen.getNewDataInstance(), 5, 5);

	public static void main(String[] args) {

		/* Step 3: Build the network */
		NeuralNetwork net = factory.getInitializedNeuralNetwork();

		/* Step 4: Generate training and test data */
		List<DataInstance> training = getData(10000);
		List<DataInstance> testing = getData(1000);

		/* Step 5: Train the network */
		DNNTrainingModule trainingModule = new DNNTrainingModule(net, training, testing);
		
		/* Step 6: Set some convergence criteria and other flags */
		trainingModule.setOutputOn(true);
		trainingModule.setConvergenceCriteria(1.0e-9, -1, 1, 1000);
		
		/* Step 7: Train the network using the specified convergence criteria */
		trainingModule.doTrainNetworkUntilConvergence();

		/* Step 8: Test the network */
		trainingModule.doTestTrainedNetwork();
		
		/* Step 9: Save the network to a file to use later */
		trainingModule.saveNetworkToFile("models" + sep + "cos.network.dnn");
	}

	/* Generate Cosine Data Instances */
	private static List<DataInstance> getData(int numInstances) {
		List<DataInstance> data = new ArrayList<>();
		for (int i = 0; i < numInstances; i++)
			data.add(dataGen.getNewDataInstance());
		return data;
	}
}
