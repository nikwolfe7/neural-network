package test;

import java.io.IOException;
import java.util.List;

import mlsp.cs.cmu.edu.dnn.factory.CrossEntropyNetworkElementFactory;
import mlsp.cs.cmu.edu.dnn.factory.DNNFactory;
import mlsp.cs.cmu.edu.dnn.factory.ReadSerializedFileDNNFactory;
import mlsp.cs.cmu.edu.dnn.factory.SigmoidNetworkElementFactory;
import mlsp.cs.cmu.edu.dnn.factory.SigmoidNetworkFFDNNFactory;
import mlsp.cs.cmu.edu.dnn.structure.NeuralNetwork;
import mlsp.cs.cmu.edu.dnn.training.DNNTrainingModule;
import mlsp.cs.cmu.edu.dnn.training.DataInstance;
import mlsp.cs.cmu.edu.dnn.training.DataInstanceFactory;
import mlsp.cs.cmu.edu.dnn.util.OutputAdapter;

public class TrainNetworkThread extends Thread {

	static int idNum = 0;
	static boolean printOut = true;
	private DNNTrainingModule trainingModule;
	private DNNFactory dnnFactory;
	private NeuralNetwork net;
	private List<DataInstance> testingSet;
	private List<DataInstance> trainingSet;
	private OutputAdapter adapter;
	private String netFile;

	public TrainNetworkThread
			(/* output */
			String netFile, 
			OutputAdapter adapter, 
			DataInstanceFactory dataFactory, 
			List<DataInstance> training,
			List<DataInstance> testing, 
			DNNFactory dnnFactory,
			/* convergence criteria */
			double minDiff, 
			double minSquaredError, 
			boolean allowNegativeIterations, 
			int numMinChangeIterations,
			int maxIterations,
			boolean batchUpdate,
			int... structure) throws IOException {

		this.adapter = adapter;
		this.trainingSet = training;
		this.testingSet = testing;
		DataInstance ex = testing.get(0);
		this.net = dnnFactory.getInitializedNeuralNetwork();
		this.trainingModule = new DNNTrainingModule(net, trainingSet, testingSet);
		trainingModule.setOutputAdapter(adapter);
		trainingModule.setOutputOn(printOut);
		trainingModule.setBatchUpdate(batchUpdate);
		trainingModule.setConvergenceCriteria(minDiff, minSquaredError, allowNegativeIterations,
				numMinChangeIterations, maxIterations);

		String[] arr = new String[] { netFile, "in-" + ex.getInputDimension(), "out-" + ex.getOutputDimension(),
				"struct-" + str(structure) + "-id-" + (++idNum)};
		this.netFile = String.join("-", arr) + ".dnn";
		System.out.println("Training network for file: " + this.netFile);
	}

	private String str(int... vals) {
		String[] arr = new String[vals.length];
		for (int i = 0; i < arr.length; i++) {
			arr[i] = "" + vals[i];
		}
		return String.join("-", arr);
	}

	private void trainNetwork() {
		trainingModule.doTrainNetworkUntilConvergence();
	}

	private void testAndSaveNetwork() {
		trainingModule.doTestTrainedNetwork();
		trainingModule.saveNetworkToFile(netFile);
	}

	private void verifySavedNetwork() {
		System.out.println("De-serializing network...");
		dnnFactory = new ReadSerializedFileDNNFactory(netFile);
		net = dnnFactory.getInitializedNeuralNetwork();
		trainingModule = new DNNTrainingModule(net, testingSet);
		trainingModule.setOutputAdapter(adapter);
		trainingModule.setOutputOn(printOut);
		trainingModule.doTestTrainedNetwork();
	}

	@Override
	public void run() {
		trainNetwork();
		testAndSaveNetwork();
		verifySavedNetwork();
	}
}
