package mlsp.cs.cmu.edu.dnn.training;

import java.io.File;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.text.DecimalFormat;
import java.util.ArrayList;
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
	private int maxEpochs = Integer.MAX_VALUE;
	private double minDifference = 0.001;
	private int numMinChangeIterations = 1;
	private double minError = Double.NEGATIVE_INFINITY;
	private DecimalFormat f = new DecimalFormat("##.#######");
	private boolean outputOn = false;
	private boolean printResults = false;
	private boolean batchUpdate = false;
	private int batchDivisions = 1;
	private File outputFile; 
	private OutputAdapter adapter = new DefaultOutput();
	
	/* snapshots */
	private boolean snapshot = false;
	private int snapshotInterval = 1; 
	private String snapshotFile = "snapshot.dnn";

	public DNNTrainingModule(NeuralNetwork network, List<DataInstance> trainingSet, List<DataInstance> testingSet) {
		this.net = network;
		this.training = trainingSet;
		this.testing = testingSet;
		this.outputFile = new File("testing-output.csv");
	}
	
	/* Just for Testing */
	public DNNTrainingModule(NeuralNetwork network, List<DataInstance> testingSet) {
		this.training = new ArrayList<>();
		this.net = network;
		this.testing = testingSet;
		this.outputFile = new File("testing-output.csv");
	}
	
	public void setConvergenceCriteria(double minDiff, double minError, int numMinChangeIterations, int maxEpochs) {
		this.minDifference = minDiff;
		this.minError = minError;
		this.numMinChangeIterations = numMinChangeIterations;
		this.maxEpochs = maxEpochs;
	}
	
	public void setConvergenceCriteria(double minDiff, double minError, int numMinChangeIterations) {
		this.minDifference = minDiff;
		this.minError = minError;
		this.numMinChangeIterations = numMinChangeIterations;
	}
	
	public void setBatchUpdate(boolean b, int batchDivisions) {
		this.batchUpdate = b;
		this.batchDivisions = (b) ? batchDivisions : 1;
		net.setBatchUpdate(b);
	}
	
	public void setBatchUpdate(boolean b) {
		this.batchUpdate = b;
		net.setBatchUpdate(b);
	}
	
	public void setOutputAdapter(OutputAdapter adapter) {
		this.adapter = adapter;
	}
	
	public void setSnapshotInterval(boolean b, int interval, String fileName) {
		this.snapshot = b;
		this.snapshotInterval = (b) ? interval : 1;
		this.snapshotFile = (b) ? fileName : "snapshot.dnn";
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
	
	public void setPrintResults(boolean b, String fileName) {
		if(printResults = b) 
			this.outputFile = new File(fileName);
	}
	
	public void doTrainNetworkUntilConvergence() {
		System.out.println("Now training network...");
		double prevSumError = Double.POSITIVE_INFINITY;
		int countDown = numMinChangeIterations;
		int batchSize = Math.floorDiv(training.size(), batchDivisions);
		int epoch = 1;
		long startEpoch, endEpoch, startTrain, endTrain;
		startTrain = System.currentTimeMillis();
		double divisor = 1.0/training.size();

		/* Train on training data */
		while (true) {
			startEpoch = System.currentTimeMillis();
			double sumError = 0;
			/* One epoch through training data... */
			if (batchUpdate) {
				int count = 1;
				for (DataInstance x : training) {
					sumError += net.trainOnInstance(x);
					if (count++ >= batchSize) {
						net.batchUpdate();
						count = 1;
					}
				}
				/* final update for any leftovers */
				net.batchUpdate();
			} else { /* Stochastic training */
				for (DataInstance x : training) {
					sumError += net.trainOnInstance(x);
				}
			}
			endEpoch = System.currentTimeMillis();
			double duration = Math.abs(endEpoch-startEpoch) * 0.001;
			/* Mean squared error */
			sumError = sumError * divisor;
			
			/* Should ideally never be negative, but no guarantees */
			double diff = prevSumError - sumError;

			if (outputOn)
				System.out.println("Epoch " + (epoch++) + " | Avg Error: " + f.format(sumError) + "\t\t| Diff: " + diff + "\t\t| Time: " + f.format(duration) + "s");

			/* Output current network to file */
			if (snapshot && epoch % snapshotInterval == 1) {
				saveNetworkToFile(snapshotFile);
			}
				
			/* Tests for convergence... */
			prevSumError = sumError;

			/* Evaluate stopping criteria */
			boolean converged = false;

			/* min difference reached */
			if (Math.abs(diff) <= minDifference) {
				if (countDown-- <= 0)
					converged = true;
			}
			/* min squared error criteria override */
			if (minError > 0) {
				if (sumError <= minError)
					converged = true;
			}
			/* num epochs */
			if (epoch >= maxEpochs) {
				converged = true;
			}
			/* convergence criteria met */
			if (converged)
				break;
		}
		/* Converged! Now test... */
		endTrain = System.currentTimeMillis();
		double duration = Math.abs(endTrain-startTrain) * 0.001;
		System.out.println(
				"===========================\n" + 
				"NETWORK WEIGHTS CONVERGED! \n" + 
				"===========================\n" +
				"Elapsed Time: " + f.format(duration) + "s\n");
	}

	public double doTestTrainedNetwork() {
		System.out.println("Now testing network...");
		double sumOfSquaredErrors = 0;
		double numCorrect = 0;
		List<String> results = new ArrayList<>();
		for(DataInstance x : testing) {
			double[] output = net.getPrediction(x.getInputVector());
			double[] truth = x.getOutputTruthValue();
			double error = CostFunction.meanSqError(output, truth);
			sumOfSquaredErrors += error;
			boolean correct = adapter.isCorrect(output, truth); 
			if(correct) {
			  numCorrect++;
			}
			if(outputOn) {
				System.out.println("Network:  " + DNNUtils.printVector(output));
				System.out.println("Truth  :  " + DNNUtils.printVector(truth));
				System.out.println("Label  :  " + ((correct) ? "Correct!" : "Incorrect"));
				System.out.println("-----------------------------------------");
			}
			if (printResults) {
				double[] vec = new double[x.getInputDimension() + x.getOutputDimension()];
				System.arraycopy(output, 0, vec, 0, output.length);
				System.arraycopy(x.getInputVector(), 0, vec, output.length, x.getInputDimension());
				results.add(DNNUtils.csvPrintVector(vec) + "\n");
			}
		}
		if(printResults) {
			try {
				FileWriter writer = new FileWriter(outputFile, true);
				for(String s : results) 
					writer.write(s);
				writer.close();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		System.out.println("Total Error:  " + f.format(sumOfSquaredErrors));
		System.out.println(" Mean Error:  " + f.format(sumOfSquaredErrors/testing.size()));
		System.out.println("   Accuracy:  " + f.format(numCorrect/testing.size()));
		return sumOfSquaredErrors;
	}
	
	public void saveNetworkToFile(String fileName) {
		OutputStream writer;
		ObjectOutputStream outputStream;
		try {
			writer = new FileOutputStream(new File(fileName));
			outputStream = new ObjectOutputStream(writer);
			outputStream.writeObject(net);
			outputStream.close();
			writer.close();
			System.out.println("Network written to file: " + fileName);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
}
