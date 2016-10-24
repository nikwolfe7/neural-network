package mlsp.cs.cmu.edu.dnn.cascor;

import java.util.ArrayList;
import java.util.List;

import mlsp.cs.cmu.edu.dnn.factory.DNNFactory;
import mlsp.cs.cmu.edu.dnn.structure.NeuralNetwork;
import mlsp.cs.cmu.edu.dnn.training.DNNTrainingModule;
import mlsp.cs.cmu.edu.dnn.training.DataInstance;
import mlsp.cs.cmu.edu.dnn.training.DataInstanceGenerator;
import mlsp.cs.cmu.edu.dnn.training.XORGenerator;
import mlsp.cs.cmu.edu.dnn.util.BinaryThresholdOutput;
import mlsp.cs.cmu.edu.dnn.util.DNNUtils;

public class XORDriver {
	
	/* Useful to have for file separators, like python... */
	static String sep = DNNUtils.sep;
	
	/* Step 1: Get a data instance generator for XOR data 
	 * Alteratively, get data from a dataset...
	 */
	static DataInstanceGenerator xorGenerator = new XORGenerator();
	
	/* Step 2: Get a DNN Network factory to buid a factory of specified structure with 
	 * inputs & outputs matching the data type...
	 */
	static DNNFactory factory = new CustomDNNFactory(xorGenerator.getNewDataInstance(), new int[] { 2 });
	
	/* XOR example... */
	public static void main(String[] args) {
		
		/* Step 3: Generate network */
		NeuralNetwork net = factory.getInitializedNeuralNetwork();
	
		/* Step 4: Generate training & test data... */
		List<DataInstance> training = getData(1000);
		List<DataInstance> testing = getData(1000);
		
		/* Step 5: Initialize a training module */
		DNNTrainingModule tm = new DNNTrainingModule(net, training, testing);
		
		/* Step 6: Set an output adapter */
		tm.setOutputAdapter(new BinaryThresholdOutput());
		
		/* Step 7: Set some useful parameters, e.g. convergence criteria */
		tm.setConvergenceCriteria(1.0e-9, -1, 1, 1000);
		tm.setOutputOn(true);
		
		/* Step 8: Run training */
		tm.doTrainNetworkUntilConvergence();
		
		/* Step 9: Do test training result */
		tm.doTestTrainedNetwork();
		
		/* Step 10: Save the network to a file */
		tm.saveNetworkToFile("models" + sep + "xor.text.dnn");
	}
	
	private static List<DataInstance> getData(int numInstances) {
	    List<DataInstance> data = new ArrayList<>();
	    for (int i = 0; i < numInstances; i++)
	      data.add(xorGenerator.getNewDataInstance());
	    return data;
	  }

}
