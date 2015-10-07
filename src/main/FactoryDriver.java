package main;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import mlsp.cs.cmu.edu.factory.DNNFactory;
import mlsp.cs.cmu.edu.factory.FeedForwardDNNFactory;
import mlsp.cs.cmu.edu.factory.NeuralNetwork;
import mlsp.cs.cmu.edu.util.DNNTrainingModule;
import mlsp.cs.cmu.edu.util.DNNUtils;
import training.DataInstance;

public class FactoryDriver {

	public static void main(String[] args) {
		DNNFactory factory = new FeedForwardDNNFactory(1, 1, 10);
		NeuralNetwork net = factory.getInitializedNeuralNetwork();
		List<DataInstance> training = getData(100000);
		List<DataInstance> testing = getData(100);
		DNNTrainingModule trainingModule = new DNNTrainingModule(net, training, testing);
		trainingModule.setOutputOn(true);
		trainingModule.doTrainNetworkUntilConvergence();
		trainingModule.doTestTrainedNetwork();
	}
	
	private static List<DataInstance> getData(int numInstances) {
	    Random r = new Random();
	    List<DataInstance> data = new ArrayList<>();
	    for(int i = 0; i < numInstances; i++) {
	      double x, y, z;
	      x = r.nextInt(360);
	      z = Math.cos(x);
	      double[] d = new double[] {z, x};
	      DataInstance instance = new DataInstance(1, 1, d);
	      data.add(instance);
	    }
//	    data = DNNUtils.zScoreNormalizeInputs(data);
	    return data;
	  }
}
