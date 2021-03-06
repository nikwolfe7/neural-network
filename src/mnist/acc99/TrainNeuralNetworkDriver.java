package mnist.acc99;

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
	static int iterations = 1000;
	static double minDiff = 1.0e-6;
	static double minError = 0.02;

	private static void runDNN(String o, int maxIter, boolean batch, int batchDiv, boolean ss, int si, int... structure)
			throws IOException {
		OutputAdapter adapter = new MaxBinaryThresholdOutput();
		String[] matrixFiles = {"data" + DNNUtils.sep + "mat-100x401.csv", "data" + DNNUtils.sep + "mat-10x101.csv"};
		DataInstanceFactory dataInstanceFactory = new MNISTAltSmallDataInstanceFactory();
		List<DataInstance> training = dataInstanceFactory.getTrainingInstances();
		List<DataInstance> testing = dataInstanceFactory.getTrainingInstances();
		DNNFactory dnnFactory = new WeightInitSigmoidDNNFactory(training.get(0), true, matrixFiles);
		Thread dnnThread = new TrainNetworkThread(o, adapter, dataInstanceFactory, training, testing, dnnFactory,
				minDiff, minError, 1, maxIter, batch, batchDiv, ss, si, structure);
		pool.execute(dnnThread);
	}
 
	public static void main(String[] args) throws IOException {
		runDNN(o, iterations, batch, batchDivisions, saveSnapshots, snapshotInterval, new int[] { 100 });
		pool.shutdown();
	}
	
	/*
	 * runDNN(o, iterations, batch, new int[] {100}); 			// 79400 -- 1.)	991s **
		runDNN(o, iterations, batch, new int[] {72,28}); 		// 58744 -- 2.)	754s
		runDNN(o, iterations, batch, new int[] {56,28,16});		// 46080 -- 3.)	593s
		runDNN(o, iterations, batch, new int[] {50,50});		// 42200 -- 4.)	522s
		runDNN(o, iterations, batch, new int[] {28,72}); 		// 24688 -- 5.)	207s **
		runDNN(o, iterations, batch, new int[] {28,56,16});		// 24576 -- 6.)	99s
		runDNN(o, iterations, batch, new int[] {16,56,28});		// 15288 -- 7.)	76s
		runDNN(o, iterations, batch, new int[] {16}); 			// 12704 -- 8.) 44s
	 * */

}
