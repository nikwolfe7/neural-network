package example;

import java.io.IOException;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import mlsp.cs.cmu.edu.dnn.factory.DNNFactory;
import mlsp.cs.cmu.edu.dnn.training.DataInstance;
import mlsp.cs.cmu.edu.dnn.training.DataInstanceFactory;
import mlsp.cs.cmu.edu.dnn.training.MNISTDataInstanceFactory;
import mlsp.cs.cmu.edu.dnn.training.TrainNetworkThread;
import mlsp.cs.cmu.edu.dnn.util.MaxBinaryThresholdOutput;
import mlsp.cs.cmu.edu.dnn.util.OutputAdapter;

public class TrainNeuralNetworkDriver {
	
	static int numCores = Runtime.getRuntime().availableProcessors();
	static ExecutorService pool = Executors.newFixedThreadPool(numCores);
	static String o = "mnist-test";
	static boolean batch = false;
	static boolean allow = true;
	static boolean saveSnapshots = true;
	static int snapshotInterval = 5; 
	static int iterations = 1000;
	static double minDiff = 1.0e-8;
	
	private static void runDNN(String o, int maxIter, boolean batch, boolean ss, int si, int... structure) throws IOException {
		OutputAdapter adapter = new MaxBinaryThresholdOutput();
		DataInstanceFactory dataInstanceFactory = new MNISTDataInstanceFactory();
		List<DataInstance> training = dataInstanceFactory.getTrainingInstances();
 		List<DataInstance> testing = dataInstanceFactory.getTestingInstances();
 		DNNFactory dnnFactory = new CustomDNNFactory(testing.get(0), structure);
		Thread dnnThread = new TrainNetworkThread(o, adapter, dataInstanceFactory, training, testing, dnnFactory, minDiff, -1, allow, 1, maxIter, batch, ss, si, structure);
		pool.execute(dnnThread);
	}
	
	public static void main(String[] args) throws IOException {
		boolean ss; int si; 
		ss = saveSnapshots;
		si = snapshotInterval; 
		runDNN(o, iterations, batch, ss, si, new int[] {16});
		runDNN(o, iterations, batch, ss, si, new int[] {100});
		runDNN(o, iterations, batch, ss, si, new int[] {28,72});
		runDNN(o, iterations, batch, ss, si, new int[] {72,28});
		runDNN(o, iterations, batch, ss, si, new int[] {50,50});
		runDNN(o, iterations, batch, ss, si, new int[] {56,28,16});
		runDNN(o, iterations, batch, ss, si, new int[] {16,56,28});
		runDNN(o, iterations, batch, ss, si, new int[] {28,56,16});
		pool.shutdown();
	}
	
}
