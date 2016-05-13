package test;

import java.io.IOException;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import mlsp.cs.cmu.edu.dnn.factory.DNNFactory;
import mlsp.cs.cmu.edu.dnn.training.DataInstance;
import mlsp.cs.cmu.edu.dnn.training.DataInstanceFactory;
import mlsp.cs.cmu.edu.dnn.training.MNISTDataInstanceFactory;
import mlsp.cs.cmu.edu.dnn.util.MaxBinaryThresholdOutput;
import mlsp.cs.cmu.edu.dnn.util.OutputAdapter;

public class TrainNeuralNetworkDriver {
	
	static int numCores = Runtime.getRuntime().availableProcessors();
	static ExecutorService pool = Executors.newFixedThreadPool(numCores);
	static String o = "mnist-test-output";
	static boolean batch = false;
	static boolean allow = true;
	static int iterations = 1000;
	static double minDiff = 1.0e-8;
	
	private static void runDNN(String o, int numFrames, int maxIter, boolean batch, int... structure) throws IOException {
		OutputAdapter adapter = new MaxBinaryThresholdOutput();
		DataInstanceFactory dataInstanceFactory = new MNISTDataInstanceFactory();
		List<DataInstance> training = dataInstanceFactory.getTrainingInstances();
 		List<DataInstance> testing = dataInstanceFactory.getTestingInstances();
 		DNNFactory dnnFactory = new CustomDNNFactory(testing.get(0), structure);
		Thread dnnThread = new TrainNetworkThread(o, adapter, dataInstanceFactory, training, testing, dnnFactory, minDiff, -1, allow, 1, maxIter, batch, structure);
		pool.execute(dnnThread);
	}
	
	public static void main(String[] args) throws IOException {
		runDNN(o, 1, iterations, batch, new int[] {5});
		runDNN(o, 2, iterations, batch, new int[] {5,5});
		runDNN(o, 3, iterations, batch, new int[] {5,5,5});
		pool.shutdown();
	}
	
}
