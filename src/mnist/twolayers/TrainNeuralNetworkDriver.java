package mnist.twolayers;

import java.io.IOException;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import mlsp.cs.cmu.edu.dnn.factory.DNNFactory;
import mlsp.cs.cmu.edu.dnn.factory.WeightInitSigmoidDNNFactory;
import mlsp.cs.cmu.edu.dnn.structure.NeuralNetwork;
import mlsp.cs.cmu.edu.dnn.training.DNNTrainingModule;
import mlsp.cs.cmu.edu.dnn.training.DataInstance;
import mlsp.cs.cmu.edu.dnn.training.DataInstanceFactory;
import mlsp.cs.cmu.edu.dnn.training.MNISTAltSmallDataInstanceFactory;
import mlsp.cs.cmu.edu.dnn.training.MNISTSmallDataInstanceFactory;
import mlsp.cs.cmu.edu.dnn.training.TrainNetworkThread;
import mlsp.cs.cmu.edu.dnn.util.DNNUtils;
import mlsp.cs.cmu.edu.dnn.util.MaxBinaryThresholdOutput;
import mlsp.cs.cmu.edu.dnn.util.OutputAdapter;

public class TrainNeuralNetworkDriver {

	static int numCores = Runtime.getRuntime().availableProcessors();
	static ExecutorService pool = Executors.newFixedThreadPool(numCores * 2);
	static String o = "mnist-acc99";
	static boolean batch = false;
	static int batchDivisions = 100;
	static boolean saveSnapshots = true;
	static int snapshotInterval = 10;
	static int iterations = 1;
	static double minDiff = 1.0e-6;
	static double minError = 0.02;

	private static void runDNN(String o, int maxIter, boolean batch, int batchDiv, boolean ss, int si, int... structure)
			throws IOException {
		
		OutputAdapter adapter = new MaxBinaryThresholdOutput();
		String[] matrixFiles = {
				"data" + DNNUtils.sep + "mat-50x401.csv", 
				"data" + DNNUtils.sep + "mat-50x51.csv", 
				"data" + DNNUtils.sep + "mat-10x51.csv"
				};
		DataInstanceFactory dataInstanceFactory = new MNISTAltSmallDataInstanceFactory();
		List<DataInstance> training = dataInstanceFactory.getTrainingInstances();
		List<DataInstance> testing = dataInstanceFactory.getTrainingInstances();
		DNNFactory factory = new WeightInitSigmoidDNNFactory(training.get(0), true, matrixFiles);
		NeuralNetwork net = factory.getInitializedNeuralNetwork();
		DNNTrainingModule testingModule = new DNNTrainingModule(net, training, testing);
		testingModule.setOutputOn(true);
		testingModule.setOutputAdapter(new MaxBinaryThresholdOutput());
		testingModule.doTestTrainedNetwork();
		
		TrainNetworkThread dnnThread = new TrainNetworkThread(o, adapter, dataInstanceFactory, training, testing, factory,
				minDiff, minError, 1, maxIter, batch, batchDiv, ss, si, structure);
		dnnThread.testAndSaveNetwork();
		//pool.execute(dnnThread);
	}
 
	public static void main(String[] args) throws IOException {
		runDNN(o, iterations, batch, batchDivisions, saveSnapshots, snapshotInterval, new int[] { 50, 50 });
		pool.shutdown();
	}

}
