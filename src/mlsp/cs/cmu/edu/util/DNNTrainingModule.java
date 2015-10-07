package mlsp.cs.cmu.edu.util;

import java.text.DecimalFormat;
import java.util.List;

import mlsp.cs.cmu.edu.factory.NeuralNetwork;
import training.DataInstance;

public class DNNTrainingModule {

	private NeuralNetwork net;
	private List<DataInstance> training;
	private List<DataInstance> testing;
	private double minDifference = 1.0e-5;
	private DecimalFormat f = new DecimalFormat("##.#####");
	private boolean outputOn = false;

	public DNNTrainingModule(NeuralNetwork network, List<DataInstance> trainingSet, List<DataInstance> testingSet) {
		this.net = network;
		this.training = trainingSet;
		this.testing = testingSet;
	}
	
	public NeuralNetwork getNetwork() {
		return net;
	}
	
	public void setTrainingData(List<DataInstance> trainingSet) {
		training = trainingSet;
	}
	
	public void setTestingData(List<DataInstance> testingSet) {
		testing = testingSet;
	}

	public void setOutputOn(boolean b) {
		outputOn = b;
	}
	
	public void doTrainNetworkUntilConvergence() {
		double prevSumSqError = Double.POSITIVE_INFINITY;
		/* Train on training data */
		while (true) {
			double sumOfSquaredErrors = 0;
			for (DataInstance x : training)
				sumOfSquaredErrors += net.trainOnInstance(x);
			/* Should never be negative */
			double diff = prevSumSqError - sumOfSquaredErrors; 
			if (outputOn)
				System.out.println("Squared Error: " + f.format(sumOfSquaredErrors) + "\tDiff: " + diff);
			prevSumSqError = sumOfSquaredErrors;
			if (diff <= minDifference)
				break;
		}
		/* Converged! Now test... */
		System.out.println(
				"==========================\n" + 
				"NETWORK WEIGHTS CONVERGED!\n" + 
				"==========================\n");
	}

	public void doTestTrainedNetwork() {
		System.out.println("Now testing network...");
		double sumOfSquaredErrors = 0;
		for(DataInstance x : testing) {
			double[] output = net.getPrediction(x.getInputVector());
			double[] truth = x.getOutputTruthValue();
			double error = CostFunction.meanSqError(output, truth);
			System.out.println("Network:  " + DNNUtils.printVector(output));
			System.out.println("Truth  :  " + DNNUtils.printVector(truth));
			System.out.println("-----------------------------------------");
			sumOfSquaredErrors += error;
		}
		System.out.println("Squared Error:  " + f.format(sumOfSquaredErrors));
		System.out.println("Mean Sq Error:  " + f.format(sumOfSquaredErrors/testing.size()));
	}

}
