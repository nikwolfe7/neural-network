package test2;

import java.util.List;

import mlsp.cs.cmu.edu.dnn.training.DataInstance;
import mlsp.cs.cmu.edu.dnn.training.DataInstanceFactory;
import mlsp.cs.cmu.edu.dnn.training.DataReader;
import mlsp.cs.cmu.edu.dnn.training.ReadCSVDataOneHotVectors;
import mlsp.cs.cmu.edu.dnn.training.ReadCSVTrainingData;
import mlsp.cs.cmu.edu.dnn.util.DNNUtils;

public class MNISTSingleDigitInstanceFactory implements DataInstanceFactory {

	String data = "." + DNNUtils.sep + "data" + DNNUtils.sep;
	List<DataInstance> training;
	List<DataInstance> testing;
	int numToExtract;
	
	public MNISTSingleDigitInstanceFactory(int n) {
		this.numToExtract = n; 
		System.out.println("Reading MNIST data... ");
		DataReader reader = new ReadCSVTrainingData();
		this.training = reader.getDataFromFile(data + "mnist-train.csv", 784, 1);
		this.testing = reader.getDataFromFile(data + "mnist-test.csv", 784, 1);
		for(DataInstance d : training) {
			double[] out = d.getOutputTruthValue();
	        if(out[0] == (double) numToExtract)
	        	out = new double[] {1,0};
	        else
	        	out = new double[] {0,1};
	        d.replaceOutputTruthValue(out);
		}
		for(DataInstance d : testing) {
			double[] out = d.getOutputTruthValue();
	        if(out[0] == (double) numToExtract)
	        	out = new double[] {1,0};
	        else
	        	out = new double[] {0,1};
	        d.replaceOutputTruthValue(out);
		}
	}

	@Override
	public List<DataInstance> getTrainingInstances() {
		return training;
	}

	@Override
	public List<DataInstance> getTestingInstances() {
		return testing;
	}
	
}
