package mlsp.cs.cmu.edu.dnn.training;

import java.io.IOException;
import java.util.List;

import mlsp.cs.cmu.edu.dnn.factory.CrossEntropyNetworkElementFactory;
import mlsp.cs.cmu.edu.dnn.factory.DNNFactory;
import mlsp.cs.cmu.edu.dnn.factory.ReadSerializedFileDNNFactory;
import mlsp.cs.cmu.edu.dnn.factory.SigmoidNetworkElementFactory;
import mlsp.cs.cmu.edu.dnn.factory.SigmoidNetworkFFDNNFactory;
import mlsp.cs.cmu.edu.dnn.structure.NeuralNetwork;
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
	private String networkFile;

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
			boolean snapshot,
			int snapshotInterval,
			int... structure) throws IOException {

		/* Training setup... */
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

		/* Create file name */
		String[] arr = new String[] { netFile, "in-" + ex.getInputDimension(), "out-" + ex.getOutputDimension(),
				"struct-" + str(structure) + "-id-" + (++idNum) };
		this.networkFile = String.join("-", arr) + ".dnn";

		/* Set snapshot criteria... */
		trainingModule.setSnapshotInterval(snapshot, snapshotInterval, networkFile);
	}

	private String str(int... vals) {
		String[] arr = new String[vals.length];
		for (int i = 0; i < arr.length; i++) {
			arr[i] = "" + vals[i];
		}
		return String.join("-", arr);
	}

	private void trainNetwork() {
		System.out.println("Training network for file: " + networkFile);
		trainingModule.doTrainNetworkUntilConvergence();
	}

	private void testAndSaveNetwork() {
		System.out.println("Testing network for file: " + networkFile);
		trainingModule.doTestTrainedNetwork();
		trainingModule.saveNetworkToFile(networkFile);
	}

	private void verifySavedNetwork() {
		System.out.println("De-serializing network file: " + networkFile);
		dnnFactory = new ReadSerializedFileDNNFactory(networkFile);
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
