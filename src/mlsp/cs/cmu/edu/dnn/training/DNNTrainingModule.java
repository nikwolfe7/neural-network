package mlsp.cs.cmu.edu.dnn.training;

import java.text.DecimalFormat;
import java.util.List;

import mlsp.cs.cmu.edu.dnn.structure.NeuralNetwork;
import mlsp.cs.cmu.edu.dnn.util.CostFunction;
import mlsp.cs.cmu.edu.dnn.util.DNNUtils;
import mlsp.cs.cmu.edu.dnn.util.DefaultOutput;
import mlsp.cs.cmu.edu.dnn.util.OutputAdapter;

public class DNNTrainingModule {

	private NeuralNetwork net;
	private List<DataInstance> training;
	private List<DataInstance> testing;
	private double minDifference = 0.01;
	private int numMinIterations = 5;
	private DecimalFormat f = new DecimalFormat("##.###");
	private boolean outputOn = false;
	private OutputAdapter adapter = new DefaultOutput();

	public DNNTrainingModule(NeuralNetwork network, List<DataInstance> trainingSet, List<DataInstance> testingSet) {
		this.net = network;
		this.training = trainingSet;
		this.testing = testingSet;
	}
	
  public void setOutputAdapter(OutputAdapter adapter) {
    this.adapter = adapter;
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
		System.out.println("Now training network...");
		double prevSumSqError = Double.POSITIVE_INFINITY;
		int countDown = numMinIterations;
		int epoch = 0;
		/* Train on training data */
		while (true) {
			double sumOfSquaredErrors = 0;
			for (DataInstance x : training)
				sumOfSquaredErrors += net.trainOnInstance(x);
			/* Should never be negative */
			double diff = prevSumSqError - sumOfSquaredErrors; 
			if (outputOn)
				System.out.println("Epoch " + (epoch++) + " | Squared Error: " + f.format(sumOfSquaredErrors) + "\tDiff: " + diff);
			prevSumSqError = sumOfSquaredErrors;
			if (diff <= minDifference) {
			  if(countDown-- <= 0)
			    break;
			}
				
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
		double numCorrect = 0;
		for(DataInstance x : testing) {
			double[] output = (double[]) net.getSmoothedPrediction(x.getInputVector(), adapter);
			double[] truth = x.getOutputTruthValue();
			if(adapter.isCorrect(output, truth))
			  numCorrect++;
			double error = CostFunction.meanSqError(output, truth);
			if(outputOn) {
				System.out.println("Network:  " + DNNUtils.printVector(output));
				System.out.println("Truth  :  " + DNNUtils.printVector(truth));
				System.out.println("-----------------------------------------");
			}
			sumOfSquaredErrors += error;
		}
		System.out.println("Squared Error:  " + f.format(sumOfSquaredErrors));
		System.out.println("Mean Sq Error:  " + f.format(sumOfSquaredErrors/testing.size()));
		System.out.println("     Accuracy:  " + f.format(numCorrect/testing.size()));
	}

}
